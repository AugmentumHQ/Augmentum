"""Export all Augmentum tool and action-verb schemas for training data.

Outputs a single JSON file mapping each surface/tag to its available
tools with full JSON Schema definitions, descriptions, and metadata.

Usage:
    python scripts/export_tool_schemas.py [--out FILE]

Default output: docs/training/tool-schemas.json
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Surface tag mapping ──────────────────────────────────────────────
# Maps SurfaceExposure fields to training tags.
SURFACE_TO_TAG = {
    "chat": ":C",
    "voice": ":V",
    "coder": ":-",
    "companion": ":B",
    "artifact_studio": ":W",
}


@dataclass
class ToolSchema:
    name: str
    description: str
    category: str
    input_schema: dict
    surfaces: dict[str, Any]
    voice_level: str | None = None
    voice_capability_line: str = ""
    core_verb: dict | None = None
    produces: list[str] = field(default_factory=lambda: ["text"])
    model_hint: str = ""


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _extract_surface_dict(tool: Any) -> dict[str, Any]:
    s = _safe_getattr(tool, "surfaces", None)
    if s is None:
        return {"chat": True, "coder": True}
    return {
        "chat": getattr(s, "chat", True),
        "voice": getattr(s, "voice", None),
        "coder": getattr(s, "coder", True),
        "companion": getattr(s, "companion", False),
        "artifact_studio": getattr(s, "artifact_studio", False),
        "voice_capability_line": getattr(s, "voice_capability_line", ""),
    }


def _extract_core_verb(tool: Any) -> dict | None:
    cv = _safe_getattr(tool, "core_verb", None)
    if cv is None:
        return None
    return {
        "safety_class": getattr(cv, "safety_class", "READ"),
        "autonomy_class": getattr(cv, "autonomy_class", "EXPLICIT"),
    }


# ── Runtime tool extraction ──────────────────────────────────────────

def _try_instantiate(cls: type) -> Any | None:
    """Try to instantiate a Tool class with minimal/None args."""
    import inspect
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.values())[1:]  # skip self

    args = {}
    for p in params:
        if p.default is not inspect.Parameter.empty:
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        args[p.name] = None

    try:
        return cls(**args)
    except Exception:
        return None


def extract_chat_tools() -> list[ToolSchema]:
    """Extract tools from augmentum/tools/*.py."""
    tools_dir = ROOT / "augmentum" / "tools"
    results = []
    seen_names = set()

    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name in (
            "__init__.py", "base.py", "registry.py", "parsing.py",
            "cache.py", "circuit_breaker.py", "events.py", "intent.py",
            "filter.py", "turn_search_dedup.py", "result_processing.py",
            "chain.py", "background_chain.py", "synthesize.py",
            "synthesis_loop.py", "query_formulator.py",
            "auto_routes.py", "constraint_compiler.py",
            "constraint_schema.py", "artifact_sanitize.py",
            "artifact_normalize.py", "artifact_validate.py",
            "artifact_pipeline.py", "artifact_templates.py",
            "artifact_storage.py", "application_server.py",
            "application_sandbox.py",
        ):
            continue

        module_name = f"augmentum.tools.{py_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"  SKIP import {module_name}: {e}", file=sys.stderr)
            continue

        from augmentum.tools.base import Tool
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if (
                isinstance(obj, type)
                and issubclass(obj, Tool)
                and obj is not Tool
                and not attr_name.startswith("_")
            ):
                instance = _try_instantiate(obj)
                if instance is None:
                    continue

                name = _safe_getattr(instance, "name", None)
                if not name or name in seen_names:
                    continue
                seen_names.add(name)

                raw_cat = _safe_getattr(instance, "category", None)
                cat_str = raw_cat.value if hasattr(raw_cat, "value") else str(raw_cat or "execute")
                results.append(ToolSchema(
                    name=name,
                    description=_safe_getattr(instance, "description", "") or "",
                    category=cat_str,
                    input_schema=_safe_getattr(instance, "input_schema", {}) or {},
                    surfaces=_extract_surface_dict(instance),
                    voice_level=_extract_surface_dict(instance).get("voice"),
                    voice_capability_line=_extract_surface_dict(instance).get("voice_capability_line", ""),
                    core_verb=_extract_core_verb(instance),
                    produces=_safe_getattr(instance, "produces", ["text"]) or ["text"],
                    model_hint=_safe_getattr(instance, "model_hint", "") or "",
                ))

    return results


def extract_coder_tools() -> list[ToolSchema]:
    """Extract coder-specific tools from augmentum/coder/tools.py."""
    results = []
    try:
        mod = importlib.import_module("augmentum.coder.tools")
    except Exception as e:
        print(f"  SKIP augmentum.coder.tools: {e}", file=sys.stderr)
        return results

    from augmentum.tools.base import Tool
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name, None)
        if (
            isinstance(obj, type)
            and issubclass(obj, Tool)
            and obj is not Tool
            and not attr_name.startswith("_")
        ):
            instance = _try_instantiate(obj)
            if instance is None:
                continue

            name = _safe_getattr(instance, "name", None)
            if not name:
                continue

            raw_cat = _safe_getattr(instance, "category", None)
            cat_str = raw_cat.value if hasattr(raw_cat, "value") else str(raw_cat or "code")
            results.append(ToolSchema(
                name=name,
                description=_safe_getattr(instance, "description", "") or "",
                category=cat_str,
                input_schema=_safe_getattr(instance, "input_schema", {}) or {},
                surfaces={"chat": False, "voice": None, "coder": True,
                           "companion": False, "artifact_studio": False},
                produces=_safe_getattr(instance, "produces", ["text"]) or ["text"],
                model_hint=_safe_getattr(instance, "model_hint", "") or "",
            ))

    return results


def extract_intent_verbs() -> list[dict]:
    """Extract action verbs from the intent registry."""
    results = []
    try:
        from augmentum.intent import REGISTRY
    except Exception as e:
        print(f"  SKIP intent registry: {e}", file=sys.stderr)
        return results

    for action in REGISTRY.all():
        results.append({
            "id": action.id,
            "summary": action.summary,
            "examples": action.examples[:5],
            "arg_schema": action.arg_schema,
            "required_args": action.required_args,
            "surfaces": action.surfaces,
            "modes": action.modes,
            "tier3_exposed": action.fanout.tier3,
            "companion_initiatable": action.companion_initiatable,
        })

    return results


# ── Group by surface tag ─────────────────────────────────────────────

def group_by_tag(
    chat_tools: list[ToolSchema],
    coder_tools: list[ToolSchema],
    verbs: list[dict],
) -> dict:
    """Organize everything by training surface tag."""
    tag_groups: dict[str, dict] = {}

    for tag_key, tag_name in SURFACE_TO_TAG.items():
        tag_groups[tag_name] = {
            "surface": tag_key,
            "tag": tag_name,
            "tools": [],
            "verbs": [],
        }

    # Add extra tags that don't map to SurfaceExposure fields
    for extra in [":A", ":N", ":G", ":X", ":R", ":S", ":Vp"]:
        tag_groups[extra] = {
            "surface": extra,
            "tag": extra,
            "tools": [],
            "verbs": [],
        }

    all_tools = chat_tools + coder_tools

    for tool in all_tools:
        entry = {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "input_schema": tool.input_schema,
            "produces": tool.produces,
            "model_hint": tool.model_hint,
        }
        if tool.voice_level:
            entry["voice_level"] = tool.voice_level
        if tool.voice_capability_line:
            entry["voice_capability_line"] = tool.voice_capability_line
        if tool.core_verb:
            entry["core_verb"] = tool.core_verb

        s = tool.surfaces
        if s.get("chat"):
            tag_groups[":C"]["tools"].append(entry)
            tag_groups[":A"]["tools"].append(entry)
            tag_groups[":W"]["tools"].append(entry)
        if s.get("voice"):
            tag_groups[":V"]["tools"].append(entry)
            tag_groups[":Vp"]["tools"].append(entry)
        if s.get("coder"):
            tag_groups[":-"]["tools"].append(entry)
        if s.get("companion"):
            tag_groups[":B"]["tools"].append(entry)

    for verb in verbs:
        entry = {
            "id": verb["id"],
            "summary": verb["summary"],
            "examples": verb["examples"],
            "arg_schema": verb["arg_schema"],
            "required_args": verb["required_args"],
            "companion_initiatable": verb["companion_initiatable"],
        }

        verb_surfaces = verb.get("surfaces", [])
        if not verb_surfaces or "voice" in verb_surfaces:
            tag_groups[":V"]["verbs"].append(entry)
            tag_groups[":B"]["verbs"].append(entry)
        if not verb_surfaces or "chat" in verb_surfaces:
            tag_groups[":C"]["verbs"].append(entry)

    return tag_groups


def build_flat_catalog(
    chat_tools: list[ToolSchema],
    coder_tools: list[ToolSchema],
    verbs: list[dict],
) -> dict:
    """Build a flat catalog of every tool/verb with full metadata."""
    catalog = {"tools": {}, "verbs": {}}

    for tool in chat_tools + coder_tools:
        catalog["tools"][tool.name] = {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "input_schema": tool.input_schema,
            "surfaces": tool.surfaces,
            "produces": tool.produces,
            "model_hint": tool.model_hint,
            "core_verb": tool.core_verb,
        }

    for verb in verbs:
        catalog["verbs"][verb["id"]] = verb

    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tool schemas for training")
    parser.add_argument("--out", default=str(ROOT / "docs" / "training" / "tool-schemas.json"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Extracting chat/general tools...")
    chat_tools = extract_chat_tools()
    print(f"  Found {len(chat_tools)} tools")

    print("Extracting coder tools...")
    coder_tools = extract_coder_tools()
    print(f"  Found {len(coder_tools)} coder tools")

    print("Extracting intent verbs...")
    verbs = extract_intent_verbs()
    print(f"  Found {len(verbs)} verbs")

    by_tag = group_by_tag(chat_tools, coder_tools, verbs)
    flat = build_flat_catalog(chat_tools, coder_tools, verbs)

    output = {
        "_meta": {
            "description": "Augmentum tool & verb schemas for training data generation",
            "total_tools": len(chat_tools) + len(coder_tools),
            "total_verbs": len(verbs),
            "surfaces": list(by_tag.keys()),
        },
        "by_surface": by_tag,
        "catalog": flat,
    }

    # Summary
    print("\n--- Summary ---")
    for tag, group in by_tag.items():
        t_count = len(group["tools"])
        v_count = len(group["verbs"])
        if t_count or v_count:
            print(f"  {tag:4s} ({group['surface']:16s}): {t_count:3d} tools, {v_count:3d} verbs")

    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten to {out_path}")
    print(f"Total: {output['_meta']['total_tools']} tools + {output['_meta']['total_verbs']} verbs")


if __name__ == "__main__":
    main()
