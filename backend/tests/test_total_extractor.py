import pytest

from backend.app.ocr_models import OCRBlock
from backend.app.total_extractor import extract_total


def block(text: str, x: int = 20, y: int = 100, confidence: float | None = 0.95) -> OCRBlock:
    return OCRBlock(text, x, y, 60, 10, confidence)


@pytest.mark.parametrize("label", ["TOTAL", " Total: ", "Amount Due", "Balance Due", "Grand Total", "Total Due"])
def test_labels_and_metadata(label: str) -> None:
    result = extract_total([block("$1,234.56", x=900), block(label)])
    assert result is not None
    assert result.field_name == "total"
    assert result.value == "1234.56"
    assert result.source_text == f"{label} $1,234.56"
    assert result.ocr_confidence == 0.95
    assert result.extraction_confidence == 0.9
    assert result.needs_review is False


def test_split_label_and_symbol() -> None:
    result = extract_total([block("Amount"), block("Due:", x=90), block("$", x=170), block("22.55", x=240)])
    assert result is not None
    assert result.value == "22.55"
    assert result.source_text == "Amount Due: 22.55"


def test_below_fallback() -> None:
    result = extract_total([block("TOTAL"), block("unknown", x=200), block("22.55", y=130)])
    assert result is not None
    assert result.value == "22.55"
    assert result.extraction_confidence == 0.75
    assert result.needs_review is True


def test_right_before_below() -> None:
    result = extract_total([block("TOTAL"), block("22.55", x=900), block("99.99", y=130)])
    assert result is not None
    assert result.value == "22.55"


@pytest.mark.parametrize("label", ["SUBTOTAL", "Sub Total", "Total Items", "Tax", "Not a total"])
def test_other_labels(label: str) -> None:
    assert extract_total([block(label), block("22.55", x=200)]) is None


@pytest.mark.parametrize("amount", ["abc", "12", "12.3", "12.345", "1,23.45", "-22.55", "22,55", "USD22.55", "12.00abc"])
def test_unsupported_amounts(amount: str) -> None:
    assert extract_total([block("TOTAL"), block(amount, x=200)]) is None


def test_missing_data() -> None:
    assert extract_total([]) is None
    assert extract_total([block("TOTAL")]) is None
    assert extract_total([block("22.55")]) is None


@pytest.mark.parametrize("x,y", [(20, 200), (500, 130), (20, 60)])
def test_unrelated_positions(x: int, y: int) -> None:
    assert extract_total([block("TOTAL"), block("22.55", x=x, y=y)]) is None


def test_does_not_skip_line() -> None:
    assert extract_total([block("TOTAL"), block("Tax", y=125), block("2.55", y=150)]) is None


@pytest.mark.parametrize("confidence", [0.4, None])
def test_weak_ocr(confidence: float | None) -> None:
    result = extract_total([block("TOTAL", confidence=confidence), block("22.55", x=200)])
    assert result is not None
    assert result.ocr_confidence == confidence
    assert result.needs_review is True


def test_conflicts_rank_by_ocr() -> None:
    result = extract_total([block("TOTAL"), block("22.55", x=200, confidence=0.5), block("99.99", x=400)])
    assert result is not None
    assert result.value == "99.99"
    assert result.extraction_confidence == 0.5
    assert result.needs_review is True


def test_same_line_ranks_first_across_labels() -> None:
    result = extract_total([block("TOTAL"), block("99.99", y=130), block("Amount Due", y=200), block("22.55", x=400, y=200)])
    assert result is not None
    assert result.value == "22.55"
    assert result.needs_review is True


def test_subtotal_and_total() -> None:
    result = extract_total([block("Subtotal", y=50), block("20.00", x=200, y=50), block("TOTAL"), block("22.55", x=200)])
    assert result is not None
    assert result.value == "22.55"
    assert result.needs_review is False


def test_split_total_items_is_not_a_total() -> None:
    blocks = [block("Total"), block("Items", x=90), block("22.55", x=200)]
    assert extract_total(blocks) is None


def test_below_amount_with_other_label_is_rejected() -> None:
    blocks = [block("TOTAL"), block("Tax", x=0, y=130), block("2.55", x=70, y=130)]
    assert extract_total(blocks) is None


def test_zero_is_a_valid_amount() -> None:
    result = extract_total([block("TOTAL"), block("0.00", x=200)])
    assert result is not None
    assert result.value == "0.00"
