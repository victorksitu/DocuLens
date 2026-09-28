import re
from collections.abc import Sequence

from backend.app.extraction_models import ExtractedField
from backend.app.extraction_text import group_blocks_into_lines, line_text
from backend.app.field_candidates import (
    choose_candidate,
    labeled_values,
    make_candidate,
)
from backend.app.ocr_models import OCRBlock

_LABEL = re.compile(r"(?:vendor|seller|sold by|from)(?:\s*:\s*|\s+|$)(?P<value>.*)", re.I)
_OTHER_FIELD = re.compile(
    r"^(?:invoice|receipt|synthetic|date|due|bill to|ship to|customer|subtotal|sub total|"
    r"total|tax|vat|gst|hst|amount|balance|item|description|qty|quantity|thank|not a real|"
    r"phone|tel|email|address|vendor|seller|sold by|from)\b", re.I
)


def _looks_like_name(text: str) -> bool:
    return (
        any(c.isalpha() for c in text)
        and not text[0].isdigit()
        and not _OTHER_FIELD.search(text)
        and not re.search(r"@|https?://|www\.|\d{2,}", text, re.I)
    )


def extract_vendor(blocks: Sequence[OCRBlock]) -> ExtractedField | None:
    candidates = []
    for value, source, score in labeled_values(blocks, _LABEL):
        if _looks_like_name(value):
            candidates.append(make_candidate("vendor", value, source, score))
    if candidates:
        return choose_candidate(candidates)
    # Header inference is deliberately weak and always requires review.
    for line in group_blocks_into_lines(blocks)[:5]:
        text = line_text(line)
        if re.match(r"^(?:bill to|ship to|customer|item|description|subtotal|total)\b", text, re.I):
            break
        if _looks_like_name(text):
            return make_candidate("vendor", text, line, 0.6)
    return None
