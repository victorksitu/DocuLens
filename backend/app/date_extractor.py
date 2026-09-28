import re
from collections.abc import Sequence
from datetime import date

from backend.app.extraction_models import ExtractedField
from backend.app.extraction_text import group_blocks_into_lines, line_text
from backend.app.field_candidates import (
    choose_candidate,
    labeled_values,
    make_candidate,
)
from backend.app.ocr_models import OCRBlock

_LABEL = re.compile(
    r"(?:invoice date|receipt date|date)(?:\s*:\s*|\s+|$)(?P<value>.*)", re.I
)
_MONTHS = {
    name: index for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
    )
}
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def parse_document_date(text: str) -> tuple[str, bool] | None:
    """Return ISO date and ambiguity flag; preserve ambiguous numeric text."""
    text = text.strip()
    iso = re.fullmatch(r"(\d{4})([-/])(\d{1,2})\2(\d{1,2})", text)
    numeric = re.fullmatch(r"(\d{1,2})([-/])(\d{1,2})\2(\d{4})", text)
    possibilities: set[str] = set()
    if iso:
        parts = [(int(iso[1]), int(iso[3]), int(iso[4]))]
    elif numeric:
        first, second, year = int(numeric[1]), int(numeric[3]), int(numeric[4])
        parts = [(year, first, second), (year, second, first)]
    else:
        words = text.replace(",", "").split()
        if len(words) != 3:
            return None
        if words[0].isdigit():
            day, month, year = words
        else:
            month, day, year = words
        month = month.lower()
        if month not in _MONTH_NAMES and month not in _MONTHS:
            return None
        if not day.isdigit() or not re.fullmatch(r"\d{4}", year):
            return None
        parts = [(int(year), _MONTHS[month[:3]], int(day))]
    for year, month, day in parts:
        try:
            possibilities.add(date(year, month, day).isoformat())
        except ValueError:
            continue
    if not possibilities:
        return None
    if len(possibilities) > 1:
        return text, True
    return possibilities.pop(), False


def extract_date(blocks: Sequence[OCRBlock]) -> ExtractedField | None:
    candidates = []
    for value, source, score in labeled_values(blocks, _LABEL):
        parsed = parse_document_date(value)
        if parsed:
            normalized, ambiguous = parsed
            candidates.append(make_candidate("date", normalized, source, 0.5 if ambiguous else score))
    if candidates:
        return choose_candidate(candidates)
    # Only standalone date lines qualify for fallback; due dates are excluded.
    for line in group_blocks_into_lines(blocks):
        parsed = parse_document_date(line_text(line))
        if parsed:
            value, ambiguous = parsed
            candidates.append(make_candidate("date", value, line, 0.5 if ambiguous else 0.65))
    return choose_candidate(candidates)
