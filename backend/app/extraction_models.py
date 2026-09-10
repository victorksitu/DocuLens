from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedField:
    """A structured field found from OCR text."""

    field_name: str
    value: str
    source_text: str
    ocr_confidence: float | None = None
    extraction_confidence: float | None = None
    needs_review: bool = False

    def __post_init__(self) -> None:
        if not self.field_name.strip():
            raise ValueError("field_name must not be empty")

        if not self.value.strip():
            raise ValueError("value must not be empty")

        if not self.source_text.strip():
            raise ValueError("source_text must not be empty")

        _validate_confidence(self.ocr_confidence, "ocr_confidence")
        _validate_confidence(self.extraction_confidence, "extraction_confidence")

        if not isinstance(self.needs_review, bool):
            raise TypeError("needs_review must be a boolean")


def _validate_confidence(confidence: float | None, name: str) -> None:
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
