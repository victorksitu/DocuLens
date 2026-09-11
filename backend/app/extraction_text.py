import re
from collections.abc import Sequence

from backend.app.ocr_models import OCRBlock


def normalize_text(text: str) -> str:
    """Normalize OCR text for case-insensitive matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def line_text(blocks: Sequence[OCRBlock]) -> str:
    """Join OCR blocks into one left-to-right text line."""
    return " ".join(block.text for block in sorted(blocks, key=lambda block: block.x))


def group_blocks_into_lines(
    blocks: Sequence[OCRBlock],
    y_tolerance: int = 10,
) -> list[list[OCRBlock]]:
    """Group OCR blocks into approximate text lines from top to bottom."""
    _validate_tolerance(y_tolerance, "y_tolerance")

    lines: list[list[OCRBlock]] = []

    for block in sorted(blocks, key=lambda item: (_vertical_center(item), item.x)):
        matching_line = _find_matching_line(lines, block, y_tolerance)

        if matching_line is None:
            lines.append([block])
        else:
            matching_line.append(block)
            matching_line.sort(key=lambda item: item.x)

    return sorted(lines, key=lambda line: (_line_top(line), line[0].x))


def sort_blocks_reading_order(
    blocks: Sequence[OCRBlock],
    y_tolerance: int = 10,
) -> list[OCRBlock]:
    """Sort OCR blocks top-to-bottom, then left-to-right inside each line."""
    return [block for line in group_blocks_into_lines(blocks, y_tolerance) for block in line]


def blocks_on_same_line(
    reference: OCRBlock,
    blocks: Sequence[OCRBlock],
    y_tolerance: int = 10,
) -> list[OCRBlock]:
    """Find blocks that sit on the same approximate line as a reference block."""
    _validate_tolerance(y_tolerance, "y_tolerance")

    same_line = [
        block
        for block in blocks
        if block is not reference
        and abs(_vertical_center(block) - _vertical_center(reference)) <= y_tolerance
    ]
    return sorted(same_line, key=lambda block: block.x)


def find_blocks_to_right(
    reference: OCRBlock,
    blocks: Sequence[OCRBlock],
    y_tolerance: int = 10,
) -> list[OCRBlock]:
    """Find same-line blocks positioned to the right of a reference block."""
    reference_right = reference.x + reference.width
    return [
        block
        for block in blocks_on_same_line(reference, blocks, y_tolerance)
        if block.x >= reference_right
    ]


def find_blocks_below(
    reference: OCRBlock,
    blocks: Sequence[OCRBlock],
    x_tolerance: int = 40,
    max_vertical_gap: int | None = None,
) -> list[OCRBlock]:
    """Find blocks below a reference block with roughly aligned horizontal centers."""
    _validate_tolerance(x_tolerance, "x_tolerance")

    if max_vertical_gap is not None:
        _validate_tolerance(max_vertical_gap, "max_vertical_gap")

    reference_bottom = reference.y + reference.height
    candidates: list[OCRBlock] = []

    for block in blocks:
        if block is reference:
            continue

        vertical_gap = block.y - reference_bottom

        if vertical_gap < 0:
            continue

        if max_vertical_gap is not None and vertical_gap > max_vertical_gap:
            continue

        if abs(_horizontal_center(block) - _horizontal_center(reference)) <= x_tolerance:
            candidates.append(block)

    return sorted(candidates, key=lambda block: (block.y, block.x))


def _find_matching_line(
    lines: Sequence[list[OCRBlock]],
    block: OCRBlock,
    y_tolerance: int,
) -> list[OCRBlock] | None:
    for line in lines:
        if abs(_vertical_center(block) - _line_vertical_center(line)) <= y_tolerance:
            return line

    return None


def _vertical_center(block: OCRBlock) -> float:
    return block.y + block.height / 2


def _horizontal_center(block: OCRBlock) -> float:
    return block.x + block.width / 2


def _line_vertical_center(line: Sequence[OCRBlock]) -> float:
    return sum(_vertical_center(block) for block in line) / len(line)


def _line_top(line: Sequence[OCRBlock]) -> int:
    return min(block.y for block in line)


def _validate_tolerance(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")
