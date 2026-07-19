from __future__ import annotations

import re
import unicodedata
from typing import Any


_WHITESPACE = re.compile(r"\s+")
_DASHES = str.maketrans({"—": "-", "–": "-", "−": "-"})
_QUOTES = str.maketrans({"«": '"', "»": '"', "„": '"', "“": '"', "”": '"'})


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text)).strip()


def normalize_lexical(text: str, config: dict[str, Any]) -> str:
    value = unicodedata.normalize(config.get("unicode_form", "NFKC"), str(text))
    if config.get("normalize_quotes", True):
        value = value.translate(_QUOTES)
    if config.get("normalize_dashes", True):
        value = value.translate(_DASHES)
    if config.get("lowercase", True):
        value = value.lower()
    if config.get("replace_yo", True):
        value = value.replace("ё", "е")
    return normalize_whitespace(value)


def normalize_natural(text: str) -> str:
    return normalize_whitespace(unicodedata.normalize("NFKC", str(text)))

