"""Repo map — codebase structure awareness for the coding agent.

Builds a compact summary of the workspace's file structure and key
definitions (functions, classes, exports) so the agent knows what
exists without reading every file. Injected into the system prompt
before each act phase iteration.

Strategy:
1. `find` + `wc -l` in container for file listing with line counts
2. For top files: `grep -n` for definition patterns (def, class, function, export)
3. Rank by relevance to the user's query (simple keyword matching)
4. Budget-fit into a token limit (~2000 tokens)

No tree-sitter dependency — uses grep patterns that work in any container.
Fast: two shell commands, pure text processing.
"""
from __future__ import annotations

import re

from augmentum.utils.logging import get_logger

if __name__ != "__main__":
    log = get_logger(__name__)


# Patterns to extract definitions from source files
_DEF_PATTERNS = {
    "py": r"^(class |def |async def )",
    "js": r"^(export |function |class |const \w+ = |module\.exports)",
    "ts": r"^(export |function |class |interface |type |const \w+ = )",
    "jsx": r"^(export |function |class |const \w+ = )",
    "tsx": r"^(export |function |class |interface |type |const \w+ = )",
    "rs": r"^(pub |fn |struct |enum |impl |trait |mod )",
    "go": r"^(func |type |var |const )",
    "rb": r"^(class |module |def )",
    "java": r"^(public |private |protected |class |interface |enum )",
    "c": r"^(\w+\s+\w+\s*\(|struct |enum |typedef )",
    "cpp": r"^(\w+\s+\w+\s*\(|class |struct |namespace |template)",
    "h": r"^(\w+\s+\w+\s*\(|class |struct |typedef |#define )",
}

# Patterns to extract import/require statements
_IMPORT_PATTERNS = {
    "py": r"^(from |import )",
    "js": r"^(import |const .* = require\(|require\()",
    "ts": r"^(import )",
    "jsx": r"^(import |const .* = require\()",
    "tsx": r"^(import )",
    "rs": r"^(use |extern crate )",
    "go": r"^\t\"",  # Go imports inside import() block
    "rb": r"^(require |require_relative )",
    "java": r"^import ",
    "c": r"^#include ",
    "cpp": r"^#include ",
    "h": r"^#include ",
}

# Extensions to scan
_CODE_EXTENSIONS = (
    "py", "js", "ts", "jsx", "tsx", "rs", "go", "rb",
    "java", "c", "cpp", "h", "cs", "php", "swift", "kt",
    "sh", "bash", "yaml", "yml", "toml", "json",
    "html", "css", "scss", "sql", "md",
)

# Directories to skip
_SKIP_DIRS = (
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "target", "vendor",
    ".tox", ".mypy_cache", ".pytest_cache", "coverage",
    ".cargo", "pkg", "bin", "obj",
)


async def build_repo_map(
    container_manager,
    workspace_id: str,
    query: str = "",
    max_files: int = 30,
    max_tokens: int = 2000,
    *,
    skip_file_listing: bool = False,
) -> str:
    """Build a compact repo map for the agent's context.

    Args:
        container_manager: ContainerManager instance
        workspace_id: Workspace to scan
        query: User's current request (for relevance ranking)
        max_files: Max files to include definitions for
        max_tokens: Approximate token budget (~4 chars per token)
        skip_file_listing: When True, omit the "## Workspace Files"
            section and return only "## Key Definitions" +
            "## Dependencies". Callers that inject
            ``WorkspaceSnapshot`` (the authoritative, auto-refreshed
            file tree) should set this True to avoid duplicating the
            file listing across two separate views — observed
            2026-04-20 as ~500-750 tokens of redundant context per
            turn on medium repos.

    Returns:
        Formatted string for injection into system prompt.
    """
    # Step 1: Get file listing with line counts
    skip_prune = " ".join(f"-path '*/{d}' -prune -o" for d in _SKIP_DIRS)
    ext_filter = " -o ".join(f"-name '*.{e}'" for e in _CODE_EXTENSIONS)
    find_cmd = (
        f"find /workspace {skip_prune} "
        f"\\( {ext_filter} \\) -print "
        f"2>/dev/null | head -200 | xargs wc -l 2>/dev/null"
    )

    try:
        file_output = await container_manager._run_command(
            workspace_id, ["bash", "-c", find_cmd], timeout=10.0,
        )
    except Exception:
        return ""

    # Parse file listing: "  42 /workspace/src/main.py"
    files: list[tuple[int, str]] = []
    for line in file_output.strip().splitlines():
        line = line.strip()
        if not line or "total" in line.lower():
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            try:
                count = int(parts[0])
                path = parts[1]
                if path.startswith("/workspace"):
                    files.append((count, path))
            except ValueError:
                continue

    if not files:
        return ""

    # Step 2: Rank files by relevance to query
    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))

    def _relevance(path: str) -> float:
        """Score file relevance to the query. Higher = more relevant."""
        p = path.lower()
        score = 0.0
        # Filename keyword match
        basename = p.rsplit("/", 1)[-1] if "/" in p else p
        name_words = set(re.findall(r'\w+', basename.replace(".", " ")))
        overlap = query_words & name_words
        score += len(overlap) * 3.0
        # Path keyword match (weaker)
        path_words = set(re.findall(r'\w+', p))
        score += len(query_words & path_words) * 0.5
        # Config/main files get a boost
        if any(k in basename for k in ("main", "index", "app", "server", "config", "setup")):
            score += 1.0
        # Test files get a slight boost if query mentions test
        if "test" in query_lower and "test" in basename:
            score += 2.0
        return score

    # Sort by relevance (desc), then by line count (desc) as tiebreaker
    files.sort(key=lambda f: (_relevance(f[1]), f[0]), reverse=True)

    # Step 3: Build the file tree
    char_budget = max_tokens * 4
    sections: list[str] = []
    chars_used = 0

    # File listing section. Omitted when WorkspaceSnapshot is the
    # authoritative tree — duplicating it just wastes prompt budget
    # AND risks the model picking the wrong source if they diverge
    # (query-ranked vs. state-complete). We still BUILD the ranked
    # ``files`` list above because definitions/imports ranking uses
    # it — we just skip emitting this section.
    if not skip_file_listing:
        tree_lines = ["## Workspace Files"]
        for line_count, path in files[:50]:
            rel = path.replace("/workspace/", "")
            entry = f"  {rel} ({line_count}L)"
            tree_lines.append(entry)
        tree_section = "\n".join(tree_lines)
        sections.append(tree_section)
        chars_used += len(tree_section)

    # Step 4: Extract definitions from top-ranked files
    top_files = files[:max_files]
    def_files = [
        (count, path) for count, path in top_files
        if any(path.endswith(f".{ext}") for ext in _DEF_PATTERNS)
    ]

    if def_files and chars_used < char_budget:
        # Build grep command for all files at once
        grep_patterns = []
        for _, path in def_files[:15]:  # Limit to top 15 for speed
            ext = path.rsplit(".", 1)[-1] if "." in path else ""
            pattern = _DEF_PATTERNS.get(ext)
            if pattern:
                grep_patterns.append(f"grep -n '{pattern}' {path} 2>/dev/null")

        if grep_patterns:
            grep_cmd = " && echo '---' && ".join(grep_patterns)
            try:
                defs_output = await container_manager._run_command(
                    workspace_id, ["bash", "-c", grep_cmd], timeout=10.0,
                )
            except Exception:
                defs_output = ""

            if defs_output.strip():
                defs_section = "\n## Key Definitions\n" + _format_definitions(
                    defs_output, char_budget - chars_used,
                )
                sections.append(defs_section)
                chars_used += len(defs_section)

    # Step 5: Extract imports from top-ranked files (dependency awareness)
    if chars_used < char_budget:
        import_files = [
            (c, p) for c, p in files[:15]
            if any(p.endswith(f".{ext}") for ext in _IMPORT_PATTERNS)
        ]
        if import_files:
            import_cmds = []
            for _, path in import_files[:10]:
                ext = path.rsplit(".", 1)[-1] if "." in path else ""
                pat = _IMPORT_PATTERNS.get(ext)
                if pat:
                    import_cmds.append(f"grep -m 10 '{pat}' {path} 2>/dev/null")
            if import_cmds:
                imp_cmd = " && echo '---' && ".join(import_cmds)
                try:
                    imp_output = await container_manager._run_command(
                        workspace_id, ["bash", "-c", imp_cmd], timeout=10.0,
                    )
                except Exception:
                    imp_output = ""
                if imp_output.strip():
                    imp_section = "\n## Dependencies\n" + _format_imports(
                        imp_output, char_budget - chars_used,
                    )
                    sections.append(imp_section)

    return "\n\n".join(sections)


def _format_imports(raw_output: str, char_budget: int) -> str:
    """Format grep output into a compact import summary.

    Groups imports by file, deduplicates, and shows which modules each
    file depends on — giving the agent a dependency graph.
    """
    lines = []
    chars = 0
    current_file = ""

    for line in raw_output.splitlines():
        if line.strip() == "---" or not line.strip():
            continue
        parts = line.split(":", 1)
        if len(parts) >= 2:
            filepath = parts[0].replace("/workspace/", "")
            content = parts[1].strip()
            if filepath != current_file:
                if chars + len(filepath) + 10 > char_budget:
                    break
                header = f"  {filepath}: "
                # Collect all imports for this file on one line
                current_file = filepath
                lines.append(header)
                chars += len(header)
            # Compact: just show the import target, not the full syntax
            compact = content.strip()
            if len(compact) > 60:
                compact = compact[:57] + "..."
            entry = f"    {compact}"
            if chars + len(entry) > char_budget:
                break
            lines.append(entry)
            chars += len(entry)

    return "\n".join(lines)


def _format_definitions(raw_output: str, char_budget: int) -> str:
    """Format grep output into a clean definitions summary."""
    lines = []
    chars = 0
    current_file = ""

    for line in raw_output.splitlines():
        if line.strip() == "---":
            continue
        if not line.strip():
            continue

        # grep -n output: "/workspace/src/main.py:42:def hello():"
        parts = line.split(":", 2)
        if len(parts) >= 3:
            filepath = parts[0].replace("/workspace/", "")
            line_num = parts[1]
            content = parts[2].strip()

            # New file header
            if filepath != current_file:
                if chars + len(filepath) + 10 > char_budget:
                    break
                file_header = f"\n  {filepath}:"
                lines.append(file_header)
                chars += len(file_header)
                current_file = filepath

            # Definition line
            entry = f"    L{line_num}: {content}"
            if chars + len(entry) > char_budget:
                break
            lines.append(entry)
            chars += len(entry)

    return "\n".join(lines)
