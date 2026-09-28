# Milestone 3: Rule-Based Field Extraction

Milestone 3 extracts six fields from normalized OCR blocks. It is deterministic
rule-based code, not a trained field-extraction model. OCR and extraction remain
separate. No business-rule validation, API, database, or frontend is added.

## Run

```bash
python -m backend.scripts.run_ocr sample_data/synthetic/synthetic_receipt.png --json-output processed/receipt_fields.json
python -m pytest backend/tests
```

The existing receipt contains a vendor, tax, and total, but no invoice number,
date, or stated subtotal. Missing values stay null; no arithmetic is used to
invent them. JSON exports always contain all six field keys and original blocks.
The CLI's other preprocessing and preview options remain available.

## Modules

- `field_extraction.py`: `extract_fields(blocks)` combines the six results.
- `amount_extractor.py`: shared label/layout matching for subtotal, tax, and total.
- `total_extractor.py`: preserves the original `extract_total(blocks)` interface.
- `field_candidates.py`: shared labeled text lookup, evidence, and candidate selection.
- `date_extractor.py`: date matching and conservative normalization.
- `invoice_extractor.py`: labeled invoice identifiers.
- `vendor_extractor.py`: labeled vendor names or a low-confidence header guess.

## Supported Rules

| Field | Labels or evidence | Output |
| --- | --- | --- |
| Total | Total, Grand Total, Amount Due, Balance Due, Total Due | Decimal string |
| Subtotal | Subtotal, Sub Total | Decimal string |
| Tax | Tax, Sales Tax, Total Tax, VAT, GST, HST, PST | Decimal string |
| Date | Date, Invoice Date, Receipt Date; standalone date fallback | ISO date or ambiguous original text |
| Invoice number | Invoice/Inv. with #, Number, No., or ID | Original identifier, including leading zeros |
| Vendor | Vendor, Seller, Sold By, From; header fallback | Original name |

Labels are case-insensitive. Label/value pairs can sit on the same line or the
next aligned line within 60 pixels. Values can be far to the right. Amounts use
the existing vertical-center and horizontal-center matching rules; text fields
use joined lines and left-edge alignment for below-label values.

Money must have two decimal places, an optional dollar sign, and optional valid
thousands separators. Each amount must occupy one OCR block, with a separate
dollar-sign block allowed. Labels must be separate from amount blocks. Negative
amounts, decimal commas, currency codes, and tax-rate-plus-amount rows are not
supported. Tax components are not summed: multiple tax rows require review even
if their amounts are equal. A stated Total Tax takes precedence over components.

Dates support YYYY-MM-DD, YYYY/MM/DD, unambiguous day/month or month/day formats
with four-digit years, and English month names (full or three-letter). Calendar
parsing rejects impossible dates. `03/04/2026` stays exactly that string and needs
review because no locale has been chosen. Due Date is not a document-date label.
Unlabeled standalone dates are weak candidates and always need review.

Invoice identifiers permit letters, digits, hyphens, and slashes and must contain
at least one digit. Alphabetic-only identifiers and identifiers split by spaces
are unsupported. An explicit invoice-number label is required.

Vendor inference considers the first five text lines, skips common titles,
contact details and field labels, and stops at customer/item sections. It can
still select an address or an unrelated heading, so header guesses always need
review. Explicit vendor labels are preferred. Names starting with digits or
containing two consecutive digits are currently rejected by this simple filter.

## Confidence and Conflicts

Every present field has `field_name`, `value`, `source_text`, `ocr_confidence`,
`extraction_confidence`, and `needs_review`. Source text retains original OCR
spelling. OCR confidence is the minimum across supporting blocks, or null when
any supporting confidence is unavailable. These scores are not accuracy metrics.

Same-line label matches score 0.9, below-label matches 0.75, standalone dates
0.65, and vendor header guesses 0.6. Ambiguous dates and conflicting candidates
score 0.5. Scores below 0.8 or low/unknown OCR confidence trigger review. Stronger
layout evidence ranks first, then OCR confidence; ties retain reading order.

No date-locale guessing, arithmetic validation, tax calculation, or missing-field
inference is performed. Fixed pixel tolerances and simple line grouping are not
reliable for every scan resolution or multi-column document.

## Verification

`backend/tests/fixtures/invoice_blocks.json` is a fictional invoice containing all
six fields. Tests cover shuffled OCR order, aliases, below-label values, missing
fields, invalid and ambiguous dates, competing candidates, tax components,
vendor inference, and JSON/preview integration. Unit tests do not need Tesseract.
Real-image smoke tests need the local Tesseract installation. Synthetic examples
demonstrate the workflow, not general accuracy on real invoices.

The next milestone is business-rule validation, including subtotal + tax versus total.
