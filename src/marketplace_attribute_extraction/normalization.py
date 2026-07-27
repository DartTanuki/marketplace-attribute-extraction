"""Text and identifier normalization helpers."""

from __future__ import annotations

import html
import re
from typing import Any


def normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def normalize_query_for_fasttext(text: Any) -> str:
    text = html.unescape(str(text or ""))
    text = text.lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    return text
