import pytest

from backend.app.extraction_models import ExtractedField


def test_extracted_field_stores_value_source_confidence_and_review_flag() -> None:
    field = ExtractedField(
        field_name="total",
        value="22.55",
        source_text="TOTAL 22.55",
        ocr_confidence=0.95,
        extraction_confidence=0.85,
        needs_review=True,
    )

    assert field.field_name == "total"
    assert field.value == "22.55"
    assert field.source_text == "TOTAL 22.55"
    assert field.ocr_confidence == 0.95
    assert field.extraction_confidence == 0.85
    assert field.needs_review is True


def test_extracted_field_allows_missing_confidence_values() -> None:
    field = ExtractedField(
        field_name="vendor",
        value="Sample Store",
        source_text="Sample Store",
    )

    assert field.ocr_confidence is None
    assert field.extraction_confidence is None
    assert field.needs_review is False


def test_extracted_field_rejects_empty_field_name() -> None:
    with pytest.raises(ValueError, match="field_name"):
        ExtractedField(
            field_name=" ",
            value="22.55",
            source_text="TOTAL 22.55",
        )


def test_extracted_field_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="value"):
        ExtractedField(
            field_name="total",
            value=" ",
            source_text="TOTAL 22.55",
        )


def test_extracted_field_rejects_empty_source_text() -> None:
    with pytest.raises(ValueError, match="source_text"):
        ExtractedField(
            field_name="total",
            value="22.55",
            source_text=" ",
        )


def test_extracted_field_rejects_ocr_confidence_outside_normalized_range() -> None:
    with pytest.raises(ValueError, match="ocr_confidence"):
        ExtractedField(
            field_name="total",
            value="22.55",
            source_text="TOTAL 22.55",
            ocr_confidence=95.0,
        )


def test_extracted_field_rejects_extraction_confidence_outside_normalized_range() -> None:
    with pytest.raises(ValueError, match="extraction_confidence"):
        ExtractedField(
            field_name="total",
            value="22.55",
            source_text="TOTAL 22.55",
            extraction_confidence=-0.1,
        )


def test_extracted_field_rejects_non_boolean_review_flag() -> None:
    with pytest.raises(TypeError, match="needs_review"):
        ExtractedField(
            field_name="total",
            value="22.55",
            source_text="TOTAL 22.55",
            needs_review="yes",
        )
