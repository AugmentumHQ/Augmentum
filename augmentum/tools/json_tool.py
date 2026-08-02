"""JSON tool — parse, validate, query, transform, and format JSON data."""

from __future__ import annotations

import json
import re

from augmentum.tools.base import Tool, ToolCategory, ToolResult


def _jsonpath_query(data: object, path: str) -> object:
    """Simple JSONPath-like query.

    Supports: $.key, $.key.nested, $.array[0], $.array[*].field
    """
    if not path or path == "$":
        return data

    # Strip leading $. or $
    path = re.sub(r"^\$\.?", "", path)
    if not path:
        return data

    current = data
    parts = re.split(r"\.(?![^[]*\])", path)  # Split on . but not inside []

    for part_idx, part in enumerate(parts):
        if not part:
            continue

        # Handle array index: key[0] or [0]
        array_match = re.match(r"(\w*)\[(\d+|\*)\]", part)
        if array_match:
            key, index = array_match.groups()
            if key:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None

            if current is None:
                return None

            if not isinstance(current, list):
                return None

            if index == "*":
                # Return all elements — if there are further path parts, apply them
                remaining_parts = parts[part_idx + 1:]
                if remaining_parts:
                    remaining_path = ".".join(remaining_parts)
                    return [_jsonpath_query(item, remaining_path) for item in current]
                return current

            idx = int(index)
            if idx < len(current):
                current = current[idx]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None

        if current is None:
            return None

    return current


def _validate_json(text: str) -> tuple[bool, str, object | None]:
    """Validate JSON and return (is_valid, error_message, parsed_data)."""
    try:
        data = json.loads(text)
        return True, "", data
    except json.JSONDecodeError as e:
        return False, f"JSON parse error at line {e.lineno}, column {e.colno}: {e.msg}", None


class JsonTool(Tool):
    """Parse, validate, query, and format JSON data."""

    @property
    def name(self) -> str:
        return "json_tool"

    @property
    def description(self) -> str:
        return (
            "JSON operations. Actions: 'validate' (check if valid JSON), "
            "'format' (pretty-print JSON), 'minify' (compact JSON), "
            "'query' (extract data using JSONPath-like syntax e.g. $.key.nested[0]), "
            "'keys' (list top-level keys), 'type' (show JSON structure types), "
            "'diff' (compare two JSON objects), 'merge' (merge two JSON objects)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["validate", "format", "minify", "query", "keys", "type", "diff", "merge"],
                },
                "json_text": {"type": "string", "description": "JSON string to process"},
                "json_text2": {"type": "string", "description": "Second JSON string (for diff/merge)"},
                "path": {"type": "string", "description": "JSONPath query (for query action, e.g. $.data[0].name)"},
                "indent": {"type": "integer", "description": "Indentation level for format (default 2)"},
            },
            "required": ["action", "json_text"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "validate")
        json_text = kwargs.get("json_text", "")

        if not json_text:
            return ToolResult(success=False, error="No JSON text provided")

        try:
            if action == "validate":
                return self._validate(json_text)
            if action == "format":
                return self._format(json_text, kwargs.get("indent", 2))
            if action == "minify":
                return self._minify(json_text)
            if action == "query":
                return self._query(json_text, kwargs.get("path", "$"))
            if action == "keys":
                return self._keys(json_text)
            if action == "type":
                return self._type_info(json_text)
            if action == "diff":
                return self._diff(json_text, kwargs.get("json_text2", ""))
            if action == "merge":
                return self._merge(json_text, kwargs.get("json_text2", ""))
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _validate(self, text: str) -> ToolResult:
        valid, error, data = _validate_json(text)
        if valid:
            type_name = type(data).__name__
            size = len(text)
            return ToolResult(
                success=True,
                output=f"Valid JSON ({type_name}, {size} bytes)",
                metadata={"valid": True, "type": type_name, "size": size},
            )
        return ToolResult(
            success=True,
            output=f"Invalid JSON: {error}",
            metadata={"valid": False, "error": error},
        )

    def _format(self, text: str, indent: int) -> ToolResult:
        data = json.loads(text)
        formatted = json.dumps(data, indent=int(indent), ensure_ascii=False, sort_keys=False)
        return ToolResult(success=True, output=formatted)

    def _minify(self, text: str) -> ToolResult:
        data = json.loads(text)
        minified = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        return ToolResult(
            success=True,
            output=minified,
            metadata={"original_size": len(text), "minified_size": len(minified)},
        )

    def _query(self, text: str, path: str) -> ToolResult:
        data = json.loads(text)
        result = _jsonpath_query(data, path)
        if result is None:
            return ToolResult(success=True, output="null (path not found)", metadata={"path": path, "found": False})
        return ToolResult(
            success=True,
            output=json.dumps(result, indent=2, ensure_ascii=False),
            metadata={"path": path, "found": True},
        )

    def _keys(self, text: str) -> ToolResult:
        data = json.loads(text)
        if isinstance(data, dict):
            keys = list(data.keys())
            return ToolResult(
                success=True,
                output=json.dumps(keys),
                metadata={"count": len(keys), "keys": keys},
            )
        if isinstance(data, list):
            return ToolResult(
                success=True,
                output=f"Array with {len(data)} elements (use query to inspect)",
                metadata={"type": "array", "length": len(data)},
            )
        return ToolResult(success=True, output=f"Primitive value: {type(data).__name__}")

    def _type_info(self, text: str) -> ToolResult:
        data = json.loads(text)

        def describe(obj: object, depth: int = 0) -> str:
            indent = "  " * depth
            if isinstance(obj, dict):
                if not obj:
                    return "{}"
                lines = ["{"]
                for k, v in obj.items():
                    lines.append(f"{indent}  {k}: {describe(v, depth + 1)}")
                lines.append(f"{indent}}}")
                return "\n".join(lines)
            if isinstance(obj, list):
                if not obj:
                    return "[]"
                return f"[{describe(obj[0], depth)}] (×{len(obj)})"
            return type(obj).__name__

        return ToolResult(success=True, output=describe(data))

    def _diff(self, text1: str, text2: str) -> ToolResult:
        if not text2:
            return ToolResult(success=False, error="Second JSON text required for diff")
        data1 = json.loads(text1)
        data2 = json.loads(text2)

        diffs: list[str] = []

        def compare(a: object, b: object, path: str = "$") -> None:
            if type(a) is not type(b):
                diffs.append(f"{path}: type changed {type(a).__name__} → {type(b).__name__}")
                return
            if isinstance(a, dict) and isinstance(b, dict):
                all_keys = set(a.keys()) | set(b.keys())
                for k in sorted(all_keys):
                    if k not in a:
                        diffs.append(f"{path}.{k}: added")
                    elif k not in b:
                        diffs.append(f"{path}.{k}: removed")
                    else:
                        compare(a[k], b[k], f"{path}.{k}")
            elif isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    diffs.append(f"{path}: length {len(a)} → {len(b)}")
                for i in range(min(len(a), len(b))):
                    compare(a[i], b[i], f"{path}[{i}]")
            elif a != b:
                diffs.append(f"{path}: {json.dumps(a)} → {json.dumps(b)}")

        compare(data1, data2)
        if not diffs:
            return ToolResult(success=True, output="No differences found", metadata={"identical": True})
        return ToolResult(
            success=True,
            output="\n".join(diffs),
            metadata={"identical": False, "diff_count": len(diffs)},
        )

    def _merge(self, text1: str, text2: str) -> ToolResult:
        if not text2:
            return ToolResult(success=False, error="Second JSON text required for merge")
        data1 = json.loads(text1)
        data2 = json.loads(text2)

        def deep_merge(a: dict, b: dict) -> dict:
            result = dict(a)
            for k, v in b.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = deep_merge(result[k], v)
                else:
                    result[k] = v
            return result

        if isinstance(data1, dict) and isinstance(data2, dict):
            merged = deep_merge(data1, data2)
            return ToolResult(
                success=True,
                output=json.dumps(merged, indent=2, ensure_ascii=False),
            )
        if isinstance(data1, list) and isinstance(data2, list):
            merged_list = data1 + data2
            return ToolResult(
                success=True,
                output=json.dumps(merged_list, indent=2, ensure_ascii=False),
            )
        return ToolResult(success=False, error="Cannot merge: both inputs must be objects or both arrays")
