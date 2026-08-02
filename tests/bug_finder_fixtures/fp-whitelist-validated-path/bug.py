"""FP-bait: looks like path traversal, but the input is whitelisted."""

from __future__ import annotations

from pathlib import Path

LOCALE_DIR = Path("/var/data/locale")
ALLOWED_LOCALES = frozenset({"en", "fr", "de", "es", "ja", "zh"})


def read_locale(locale: str) -> str:
    # `locale` is user input but is rejected if not in a fixed whitelist.
    # `..` / `../etc/passwd` / `en/../..` all fail the membership check.
    if locale not in ALLOWED_LOCALES:
        raise ValueError(f"unsupported locale: {locale!r}")
    target = LOCALE_DIR / f"{locale}.json"
    return target.read_text(encoding="utf-8")
