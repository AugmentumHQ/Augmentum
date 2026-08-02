"""Boot-smoke error-locus extraction — the location-rich failure detail that
makes a self-heal repair possible (2026-07-02).

The bug this locks: boot-smoke used to keep only the LAST line of a traceback
("IndentationError: unexpected indent") and throw the location away, so a
self-heal repair was asked to fix an error in an 8,600-line file blind. The locus
extractor keeps file:line + the offending source line — the structured capture
the coder's parse-checkers do.
"""

from __future__ import annotations

from augmentum.selfedit.bootsmoke import _error_locus

# a real SyntaxError/IndentationError traceback (import-time)
_INDENT_TB = '''\
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/data/selfedit/repo/augmentum/proxy/server.py", line 8635
    from augmentum.proxy.mobile_pair_routes import router as mobile_pair_router
    ^^^^
IndentationError: unexpected indent'''

# a runtime error during import (deepest frame is the site)
_RUNTIME_TB = '''\
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/app/augmentum/proxy/foo_routes.py", line 42, in <module>
    x = undefined_name + 1
NameError: name 'undefined_name' is not defined'''


def test_locus_keeps_file_line_for_indentation_error():
    d = _error_locus(_INDENT_TB)
    assert "server.py:8635" in d                     # the LOCATION, not just the message
    assert "unexpected indent" in d
    assert "mobile_pair_routes" in d                 # the offending source line
    assert "<string>" not in d                       # the -c wrapper frame is skipped


def test_locus_keeps_deepest_frame_for_runtime_error():
    d = _error_locus(_RUNTIME_TB)
    assert "foo_routes.py:42" in d
    assert "NameError" in d and "undefined_name" in d


def test_locus_falls_back_to_last_line_without_a_frame():
    assert _error_locus("SomeError: no traceback here") == "SomeError: no traceback here"
    assert _error_locus("") == "no output"


def test_locus_uses_basename_not_full_path():
    d = _error_locus(_INDENT_TB)
    assert "/data/selfedit/repo" not in d            # basename only — compact + portable
