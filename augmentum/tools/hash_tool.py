"""Hash tool — compute cryptographic and non-cryptographic hashes."""

from __future__ import annotations

import hashlib
import hmac

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult

_SUPPORTED_ALGORITHMS = {
    "md5", "sha1", "sha224", "sha256", "sha384", "sha512",
    "sha3_224", "sha3_256", "sha3_384", "sha3_512",
    "blake2b", "blake2s",
}


def compute_hash(text: str, algorithm: str = "sha256") -> str:
    """Compute hash of text using the specified algorithm."""
    algo = algorithm.lower().replace("-", "_")
    if algo not in _SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_ALGORITHMS))}"
        )
    h = hashlib.new(algo)
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def compute_hmac(text: str, key: str, algorithm: str = "sha256") -> str:
    """Compute HMAC of text using the specified key and algorithm."""
    algo = algorithm.lower().replace("-", "_")
    if algo not in _SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return hmac.new(
        key.encode("utf-8"),
        text.encode("utf-8"),
        algo,
    ).hexdigest()


class HashTool(Tool):
    """Compute cryptographic hashes of text."""

    @property
    def name(self) -> str:
        return "hash"

    @property
    def description(self) -> str:
        return (
            "Compute hash digests of text. Actions: 'hash' (compute hash), "
            "'hmac' (compute HMAC with a key), 'compare' (check if text matches a hash). "
            "Algorithms: md5, sha1, sha256, sha384, sha512, sha3_256, blake2b, etc."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True,
            voice="core",
            coder=True,
            voice_capability_line="compute a hash or checksum (hash)",
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["hash", "hmac", "compare"],
                    "description": "Operation to perform",
                },
                "text": {"type": "string", "description": "Text to hash"},
                "algorithm": {
                    "type": "string",
                    "description": "Hash algorithm (default: sha256)",
                },
                "key": {"type": "string", "description": "HMAC key (for hmac action)"},
                "expected": {"type": "string", "description": "Expected hash to compare against"},
            },
            "required": ["action", "text"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "hash")
        text = kwargs.get("text", "")
        algorithm = kwargs.get("algorithm", "sha256")

        if not text:
            return ToolResult(success=False, error="No text provided")

        try:
            if action == "hash":
                digest = compute_hash(text, algorithm)
                return ToolResult(
                    success=True,
                    output=digest,
                    metadata={"algorithm": algorithm, "digest": digest, "length": len(digest)},
                )

            if action == "hmac":
                key = kwargs.get("key", "")
                if not key:
                    return ToolResult(success=False, error="HMAC key required")
                digest = compute_hmac(text, key, algorithm)
                return ToolResult(
                    success=True,
                    output=digest,
                    metadata={"algorithm": algorithm, "digest": digest},
                )

            if action == "compare":
                expected = kwargs.get("expected", "")
                if not expected:
                    return ToolResult(success=False, error="Expected hash required for comparison")
                digest = compute_hash(text, algorithm)
                match = hmac.compare_digest(digest, expected.lower())
                return ToolResult(
                    success=True,
                    output=f"Match: {match}",
                    metadata={
                        "match": match,
                        "computed": digest,
                        "expected": expected.lower(),
                        "algorithm": algorithm,
                    },
                )

            return ToolResult(success=False, error=f"Unknown action: {action}")

        except ValueError as e:
            return ToolResult(success=False, error=str(e))
