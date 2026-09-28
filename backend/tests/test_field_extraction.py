import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from backend.app.amount_extractor import extract_subtotal, extract_tax
from backend.app.date_extractor import extract_date, parse_document_date
from backend.app.field_extraction import extract_fields
from backend.app.invoice_extractor import extract_invoice_number
from backend.app.ocr_models import OCRBlock
from backend.app.vendor_extractor import extract_vendor
from backend.scripts import run_ocr


def block(text, x=20, y=100, confidence=0.95):
    return OCRBlock(text, x, y, 80, 12, confidence)


@pytest.mark.parametrize("extractor,label", [
    (extract_subtotal, "Subtotal"), (extract_subtotal, "Sub Total:"),
    (extract_tax, "Tax"), (extract_tax, "Sales Tax"),
    (extract_tax, "VAT"), (extract_tax, "GST"), (extract_tax, "HST"),
    (extract_tax, "Total Tax"),
])
def test_amount_labels(extractor, label):
    result = extractor([block(label), block("$1,234.56", x=900)])
    assert result.value == "1234.56"
    assert result.ocr_confidence == 0.95
    assert not result.needs_review


@pytest.mark.parametrize("extractor,label", [(extract_subtotal, "Subtotal"), (extract_tax, "Tax")])
def test_amount_below_and_zero(extractor, label):
    result = extractor([block(label), block("0.00", y=140)])
    assert result.value == "0.00"
    assert result.needs_review


def test_tax_components_are_not_summed_and_conflict_is_flagged():
    result = extract_tax([
        block("GST"), block("5.00", x=300),
        block("HST", y=160), block("13.00", x=300, y=160),
    ])
    assert result.value == "5.00"
    assert result.needs_review


def test_amounts_do_not_borrow_other_fields_or_rates():
    assert extract_subtotal([block("Total"), block("20.00", x=300)]) is None
    assert extract_tax([block("Tax"), block("13%", x=300)]) is None
    assert extract_tax([block("Tax"), block("Total", y=140), block("20.00", x=300, y=140)]) is None


@pytest.mark.parametrize("raw,expected", [
    ("2026-09-28", "2026-09-28"), ("2026/09/28", "2026-09-28"),
    ("09/28/2026", "2026-09-28"), ("28/09/2026", "2026-09-28"),
    ("September 28, 2026", "2026-09-28"), ("28 Sep 2026", "2026-09-28"),
    ("2024-02-29", "2024-02-29"), ("05/05/2026", "2026-05-05"),
])
def test_supported_dates(raw, expected):
    assert parse_document_date(raw) == (expected, False)
    result = extract_date([block("Invoice Date:"), block(raw, x=300)])
    assert result.value == expected
    assert not result.needs_review


@pytest.mark.parametrize("raw", ["2026-02-29", "31/04/2026", "13/13/2026", "01/02/26", "2026-09/28", "Septober 28 2026"])
def test_invalid_or_unsupported_dates(raw):
    assert parse_document_date(raw) is None
    assert extract_date([block("Date: " + raw)]) is None


def test_ambiguous_date_preserves_text_for_review():
    result = extract_date([block("Date: 03/04/2026")])
    assert result.value == "03/04/2026"
    assert result.needs_review
    assert result.extraction_confidence == 0.5


def test_date_below_label_and_unlabeled_fallback():
    for blocks in ([block("Date:"), block("2026-09-28", y=140)], [block("2026-09-28")]):
        result = extract_date(blocks)
        assert result.value == "2026-09-28"
        assert result.needs_review


def test_due_date_is_not_document_date():
    assert extract_date([block("Due Date: 2026-10-28")]) is None
    result = extract_date([block("Due Date: 2026-10-28"), block("Date: 2026-09-28", y=160)])
    assert result.value == "2026-09-28"


def test_conflicting_document_dates_require_review():
    result = extract_date([block("Date: 2026-09-28"), block("Date: 2026-09-29", y=160)])
    assert result.needs_review


@pytest.mark.parametrize("label", ["Invoice #", "Invoice Number:", "Invoice No.", "Inv. No:", "Invoice ID:"])
def test_invoice_labels_and_identifier_preservation(label):
    result = extract_invoice_number([block(label), block("INV-0007/26", x=900)])
    assert result.value == "INV-0007/26"
    assert not result.needs_review


def test_invoice_whole_line_and_below():
    assert extract_invoice_number([block("Invoice #: 000007")]).value == "000007"
    result = extract_invoice_number([block("Invoice No:"), block("AB-007", y=140)])
    assert result.value == "AB-007"
    assert result.needs_review


@pytest.mark.parametrize("text", ["Order # 123", "Invoice Date: 2026-09-28", "Invoice #", "Invoice # pending", "Invoice # INV 123"])
def test_invoice_rejects_unrelated_or_unsupported_text(text):
    assert extract_invoice_number([block(text)]) is None


def test_invoice_conflict_and_unknown_ocr_confidence():
    result = extract_invoice_number([
        block("Invoice # 001", confidence=None), block("Invoice # 002", y=160),
    ])
    assert result.value == "002"
    assert result.needs_review


def test_vendor_explicit_label_over_header():
    result = extract_vendor([block("Other Store", y=20), block("Vendor: Maple Supplies")])
    assert result.value == "Maple Supplies"
    assert not result.needs_review


def test_vendor_below_label():
    result = extract_vendor([block("Sold By:"), block("Maple Supplies", y=140)])
    assert result.value == "Maple Supplies"
    assert result.needs_review


def test_vendor_header_skips_title_and_requires_review():
    result = extract_vendor([block("SYNTHETIC RECEIPT", y=20), block("Sample Store", y=60)])
    assert result.value == "Sample Store"
    assert result.needs_review


@pytest.mark.parametrize("text", ["123 Main Street", "support@example.com", "www.example.com", "TOTAL 22.55", "Invoice # 007"])
def test_vendor_rejects_contact_and_field_lines(text):
    assert extract_vendor([block(text)]) is None


def test_vendor_does_not_use_customer_name():
    assert extract_vendor([block("Bill To:"), block("Customer Company", y=140)]) is None


def test_vendor_conflicts_are_flagged():
    result = extract_vendor([block("Vendor: Maple Supplies"), block("Seller: Other Store", y=160)])
    assert result.needs_review


def test_missing_fields_remain_missing():
    assert extract_fields([]) == dict.fromkeys(
        ["vendor", "invoice_number", "date", "subtotal", "tax", "total"]
    )


def test_invoice_hash_can_touch_identifier():
    assert extract_invoice_number([block("Invoice #INV-007")]).value == "INV-007"


def test_equal_tax_components_still_require_review():
    result = extract_tax([
        block("GST"), block("5.00", x=300),
        block("PST", y=160), block("5.00", x=300, y=160),
    ])
    assert result.value == "5.00"
    assert result.needs_review


def test_explicit_total_tax_wins_over_components():
    result = extract_tax([
        block("GST"), block("5.00", x=300),
        block("PST", y=160), block("8.00", x=300, y=160),
        block("Total Tax", y=220), block("13.00", x=300, y=220),
    ])
    assert result.value == "13.00"
    assert not result.needs_review


@pytest.fixture
def invoice_blocks():
    path = Path(__file__).parent / "fixtures" / "invoice_blocks.json"
    return [OCRBlock(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def test_complete_invoice_in_shuffled_reading_order(invoice_blocks):
    fields = extract_fields(list(reversed(invoice_blocks)))
    assert {name: field.value for name, field in fields.items()} == {
        "vendor": "Maple Supplies", "invoice_number": "INV-0007",
        "date": "2026-09-28", "subtotal": "100.00", "tax": "13.00", "total": "113.00",
    }
    for name, field in fields.items():
        assert field.field_name == name
        assert field.source_text
        assert field.ocr_confidence == 0.95
        assert not field.needs_review


def test_script_exports_all_fields_with_preview(monkeypatch, invoice_blocks):
    monkeypatch.setattr(run_ocr, "run_ocr_pipeline", lambda *_a, **_k: (
        np.full((450, 1000), 255, dtype=np.uint8), invoice_blocks
    ))
    with TemporaryDirectory(dir=Path(__file__).parent) as directory:
        output = Path(directory) / "invoice.json"
        preview = Path(directory) / "preview.png"
        run_ocr.main(["invoice.png", "--json-output", str(output), "--annotate-output", str(preview)])
        result = json.loads(output.read_text(encoding="utf-8"))
        assert preview.is_file()
    assert {name: field["value"] for name, field in result["fields"].items()} == {
        "vendor": "Maple Supplies", "invoice_number": "INV-0007",
        "date": "2026-09-28", "subtotal": "100.00", "tax": "13.00", "total": "113.00",
    }
    assert len(result["blocks"]) == len(invoice_blocks)
