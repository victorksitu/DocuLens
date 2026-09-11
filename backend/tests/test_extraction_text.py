import pytest

from backend.app.extraction_text import (
    blocks_on_same_line,
    find_blocks_below,
    find_blocks_to_right,
    group_blocks_into_lines,
    line_text,
    normalize_text,
    sort_blocks_reading_order,
)
from backend.app.ocr_models import OCRBlock


def block(text: str, x: int, y: int, width: int = 20, height: int = 10) -> OCRBlock:
    return OCRBlock(text=text, x=x, y=y, width=width, height=height, confidence=0.9)


def test_normalize_text_collapses_whitespace_and_lowercases() -> None:
    assert normalize_text("  TOTAL\tAmount  ") == "total amount"


def test_line_text_joins_blocks_left_to_right() -> None:
    blocks = [block("22.55", 180, 100), block("TOTAL", 60, 100)]

    assert line_text(blocks) == "TOTAL 22.55"


def test_group_blocks_into_lines_groups_y_jitter_and_sorts_lines() -> None:
    blocks = [
        block("22.55", 180, 203),
        block("TOTAL", 60, 200),
        block("Store", 120, 50),
        block("Sample", 40, 52),
    ]

    lines = group_blocks_into_lines(blocks, y_tolerance=6)

    assert [[item.text for item in line] for line in lines] == [
        ["Sample", "Store"],
        ["TOTAL", "22.55"],
    ]


def test_sort_blocks_reading_order_flattens_grouped_lines() -> None:
    blocks = [
        block("22.55", 180, 203),
        block("TOTAL", 60, 200),
        block("Store", 120, 50),
        block("Sample", 40, 52),
    ]

    sorted_blocks = sort_blocks_reading_order(blocks, y_tolerance=6)

    assert [item.text for item in sorted_blocks] == [
        "Sample",
        "Store",
        "TOTAL",
        "22.55",
    ]


def test_blocks_on_same_line_finds_blocks_within_vertical_tolerance() -> None:
    total = block("TOTAL", 60, 200, width=50)
    value = block("22.55", 500, 203)
    above = block("12.00", 500, 150)

    same_line = blocks_on_same_line(total, [total, value, above], y_tolerance=6)

    assert same_line == [value]


def test_find_blocks_to_right_allows_far_right_same_line_values() -> None:
    total = block("TOTAL", 60, 200, width=50)
    value = block("22.55", 900, 202)
    left_value = block("99.99", 10, 202)
    below_value = block("10.00", 900, 240)

    candidates = find_blocks_to_right(
        total,
        [total, value, left_value, below_value],
        y_tolerance=6,
    )

    assert candidates == [value]


def test_find_blocks_below_finds_value_under_label() -> None:
    total = block("TOTAL", 60, 200, width=50)
    value = block("22.55", 65, 230, width=50)
    far_right = block("12.00", 300, 230)
    same_line = block("8.50", 70, 202)

    candidates = find_blocks_below(
        total,
        [total, value, far_right, same_line],
        x_tolerance=30,
    )

    assert candidates == [value]


def test_find_blocks_below_respects_max_vertical_gap() -> None:
    total = block("TOTAL", 60, 200, width=50)
    nearby = block("22.55", 65, 230, width=50)
    too_far = block("99.99", 65, 400, width=50)

    candidates = find_blocks_below(
        total,
        [total, nearby, too_far],
        x_tolerance=30,
        max_vertical_gap=40,
    )

    assert candidates == [nearby]


def test_group_blocks_into_lines_rejects_negative_y_tolerance() -> None:
    with pytest.raises(ValueError, match="y_tolerance"):
        group_blocks_into_lines([block("TOTAL", 60, 200)], y_tolerance=-1)


def test_blocks_on_same_line_rejects_negative_y_tolerance() -> None:
    total = block("TOTAL", 60, 200)

    with pytest.raises(ValueError, match="y_tolerance"):
        blocks_on_same_line(total, [total], y_tolerance=-1)


def test_find_blocks_below_rejects_negative_x_tolerance() -> None:
    total = block("TOTAL", 60, 200)

    with pytest.raises(ValueError, match="x_tolerance"):
        find_blocks_below(total, [total], x_tolerance=-1)


def test_find_blocks_below_rejects_negative_max_vertical_gap() -> None:
    total = block("TOTAL", 60, 200)

    with pytest.raises(ValueError, match="max_vertical_gap"):
        find_blocks_below(total, [total], max_vertical_gap=-1)
