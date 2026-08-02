"""Toy ring-buffer reader."""

from __future__ import annotations


def read_first_n(items: list[str], n: int) -> list[str]:
    out: list[str] = []
    # BUG: range stops at len(items)+1 instead of len(items). When the
    # caller passes n >= len(items), the last iteration tries items[len(items)]
    # which is IndexError.
    for i in range(min(n, len(items) + 1)):
        out.append(items[i])
    return out


def demo() -> list[str]:
    return read_first_n(["a", "b", "c"], 5)
