"""Toy log appender."""

from __future__ import annotations


def append_line(path: str, line: str) -> bool:
    # BUG: open() without context manager. On the exception path, the file
    # descriptor leaks. Under concurrent load this exhausts ulimit -n.
    f = open(path, "a")
    if not line:
        raise ValueError("empty line")  # f never closed
    f.write(line + "\n")
    f.close()
    return True
