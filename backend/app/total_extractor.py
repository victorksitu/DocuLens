from collections.abc import Sequence

from backend.app.amount_extractor import extract_amount
from backend.app.extraction_models import ExtractedField
from backend.app.ocr_models import OCRBlock


def extract_total(blocks: Sequence[OCRBlock]) -> ExtractedField | None:
    """Extract the stated total using the shared amount rules."""
    labels = {"total", "grand total", "amount due", "balance due", "total due"}
    return extract_amount(blocks, "total", labels)
