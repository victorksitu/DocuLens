from collections.abc import Sequence

from backend.app.amount_extractor import extract_subtotal, extract_tax
from backend.app.date_extractor import extract_date
from backend.app.extraction_models import ExtractedField
from backend.app.invoice_extractor import extract_invoice_number
from backend.app.ocr_models import OCRBlock
from backend.app.total_extractor import extract_total
from backend.app.vendor_extractor import extract_vendor


def extract_fields(blocks: Sequence[OCRBlock]) -> dict[str, ExtractedField | None]:
    """Extract the six Milestone 3 fields without inferring missing values."""
    return {
        "vendor": extract_vendor(blocks),
        "invoice_number": extract_invoice_number(blocks),
        "date": extract_date(blocks),
        "subtotal": extract_subtotal(blocks),
        "tax": extract_tax(blocks),
        "total": extract_total(blocks),
    }
