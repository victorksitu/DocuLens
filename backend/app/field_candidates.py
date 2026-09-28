import re
from collections.abc import Iterator, Sequence
from dataclasses import replace

from backend.app.extraction_models import ExtractedField
from backend.app.extraction_text import group_blocks_into_lines, line_text
from backend.app.ocr_models import OCRBlock


def labeled_values(
    blocks: Sequence[OCRBlock], pattern: re.Pattern[str]
) -> Iterator[tuple[str, list[OCRBlock], float]]:
    """Yield same-line values, or the next aligned line for an empty label.

    Joining each line supports both word-level and whole-line OCR blocks.
    Patterns must capture the value in a group named 'value'.
    """
    lines = group_blocks_into_lines(blocks)
    for index, line in enumerate(lines):
        match = pattern.fullmatch(line_text(line).strip())
        if match is None:
            continue
        value = match["value"].strip()
        if value:
            yield value, line, 0.9
        elif index + 1 < len(lines):
            below = lines[index + 1]
            gap = min(b.y for b in below) - max(b.y + b.height for b in line)
            if 0 <= gap <= 60 and abs(below[0].x - line[0].x) <= 40:
                yield line_text(below), [*line, *below], 0.75


def make_candidate(
    name: str, value: str, source: Sequence[OCRBlock], score: float
) -> ExtractedField:
    known = [b.confidence for b in source if b.confidence is not None]
    confidence = min(known) if len(known) == len(source) else None
    return ExtractedField(
        field_name=name,
        value=value,
        source_text=" ".join(b.text for b in source),
        ocr_confidence=confidence,
        extraction_confidence=score,
        needs_review=score < 0.8 or confidence is None or confidence < 0.8,
    )


def choose_candidate(candidates: Sequence[ExtractedField]) -> ExtractedField | None:
    if not candidates:
        return None
    best = max(candidates, key=lambda field: (
        field.extraction_confidence or 0,
        field.ocr_confidence if field.ocr_confidence is not None else -1,
    ))
    if len({field.value for field in candidates}) > 1:
        return replace(best, extraction_confidence=0.5, needs_review=True)
    return best
