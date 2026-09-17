import re
from collections.abc import Sequence

from backend.app.extraction_models import ExtractedField
from backend.app.extraction_text import (
    find_blocks_below,
    find_blocks_to_right,
    group_blocks_into_lines,
    line_text,
    normalize_text,
)
from backend.app.ocr_models import OCRBlock

_LABELS = {"total", "grand total", "amount due", "balance due", "total due"}
_AMOUNT = re.compile(r"\$?\s*([0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)\.([0-9]{2})")


def extract_total(blocks: Sequence[OCRBlock]) -> ExtractedField | None:
    """Find an English total label and a nearby nonnegative decimal amount.

    Amounts must occupy one OCR block; a separate dollar sign is allowed.
    Scores are rule strengths, not probabilities. Layout tolerances are in
    pixels; below-label searches stop at the next text line within 60px.
    """
    candidates: list[ExtractedField] = []
    lines = group_blocks_into_lines(blocks)
    for index, line in enumerate(lines):
        label = _find_label(line)
        if not label:
            continue
        amounts = _find_amounts(find_blocks_to_right(label[-1], line))
        score = 0.9
        if not amounts and index + 1 < len(lines):
            nearby = find_blocks_below(
                _label_box(label), lines[index + 1], max_vertical_gap=60
            )
            amounts = _find_amounts(nearby)
            score = 0.75
        for amount, value in amounts:
            source = [*label, amount]
            known = [b.confidence for b in source if b.confidence is not None]
            confidence = min(known) if len(known) == len(source) else None
            candidates.append(
                ExtractedField(
                    field_name="total",
                    value=value,
                    source_text=" ".join(b.text for b in source),
                    ocr_confidence=confidence,
                    extraction_confidence=score,
                    needs_review=score < 0.8 or confidence is None or confidence < 0.8,
                )
            )
    if not candidates:
        return None
    # Prefer same-line matches, then OCR confidence; ties keep reading order.
    best = max(
        candidates,
        key=lambda field: (
            field.extraction_confidence,
            field.ocr_confidence if field.ocr_confidence is not None else -1,
        ),
    )
    conflicting = len({field.value for field in candidates}) > 1
    return ExtractedField(
        field_name="total", value=best.value, source_text=best.source_text,
        ocr_confidence=best.ocr_confidence,
        extraction_confidence=0.5 if conflicting else best.extraction_confidence,
        needs_review=best.needs_review or conflicting,
    )


def _find_label(line: Sequence[OCRBlock]) -> list[OCRBlock]:
    # Longest first keeps "Total Due" together instead of matching only "Total".
    for count in range(min(3, len(line)), 0, -1):
        prefix = line[:count]
        if normalize_text(line_text(prefix)).rstrip(" :") in _LABELS:
            return list(prefix)
    return []


def _find_amounts(blocks: Sequence[OCRBlock]) -> list[tuple[OCRBlock, str]]:
    amounts = []
    for block in blocks:
        if block.text.strip() == "$":
            continue
        match = _AMOUNT.fullmatch(block.text.strip())
        # Extra words may be a different label, such as "Total Items" or "Tax".
        if not match:
            return []
        value = f"{match[1].replace(',', '')}.{match[2]}"
        amounts.append((block, value))
    return amounts


def _label_box(blocks: Sequence[OCRBlock]) -> OCRBlock:
    left = min(b.x for b in blocks)
    top = min(b.y for b in blocks)
    right = max(b.x + b.width for b in blocks)
    bottom = max(b.y + b.height for b in blocks)
    return OCRBlock(line_text(blocks), left, top, right - left, bottom - top)
