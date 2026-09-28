import re
from collections.abc import Sequence

from backend.app.extraction_models import ExtractedField
from backend.app.field_candidates import (
    choose_candidate,
    labeled_values,
    make_candidate,
)
from backend.app.ocr_models import OCRBlock

_LABEL = re.compile(
    r"(?:invoice|inv\.?)\s*(?:#\s*:?\s*|(?:number|no\.?|id)(?:\s*[:#]\s*|\s+|$))(?P<value>.*)",
    re.I,
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")


def extract_invoice_number(blocks: Sequence[OCRBlock]) -> ExtractedField | None:
    candidates = []
    for value, source, score in labeled_values(blocks, _LABEL):
        if _IDENTIFIER.fullmatch(value) and any(char.isdigit() for char in value):
            candidates.append(make_candidate("invoice_number", value, source, score))
    return choose_candidate(candidates)
