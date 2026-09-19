# Milestone 2: OCR + Bounding Boxes

Milestone 2 takes a local document image that has been prepared by the preprocessing layer and runs OCR while preserving text location and confidence.

The OCR layer turns image text into normalized OCR blocks. The script now also supports Milestone 3 total extraction through an optional JSON export.

## Current Flow

```text
local JPG/PNG path
    -> validate and load with OpenCV
    -> preprocess image
    -> run Tesseract OCR
    -> normalize raw OCR rows into OCRBlock objects
    -> print detected text, box coordinates, and confidence
    -> optionally save an annotated bounding-box preview
    -> optionally extract the total and save OCR blocks plus fields as JSON
```

## Files

- `backend/app/ocr_models.py`: defines `OCRBlock`, the internal shape for detected OCR text.
- `backend/app/ocr.py`: runs pytesseract, normalizes Tesseract output, and handles confidence values.
- `backend/app/ocr_visualization.py`: draws OCR bounding boxes on a copy of an image.
- `backend/scripts/run_ocr.py`: command-line runner for local OCR smoke tests and optional preview images.
- `backend/tests/test_ocr_models.py`: tests the OCR block model.
- `backend/tests/test_ocr.py`: tests OCR normalization and wrapper behavior.
- `backend/tests/test_ocr_visualization.py`: tests bounding-box drawing.
- `backend/tests/test_run_ocr_script.py`: tests the command-line OCR workflow.

## Dependencies

Python dependencies are listed in `backend/requirements.txt`.

Install them inside the virtual environment:

```bash
source .venv/Scripts/activate
python -m pip install -r backend/requirements.txt
```

`pytesseract` is only the Python wrapper. The actual Tesseract OCR program must also be installed on the machine.

On Windows, verify Tesseract is available with:

```bash
tesseract --version
```

The OCR wrapper also checks common Windows install paths, including:

```text
C:/Program Files/Tesseract-OCR/tesseract.exe
C:/Program Files (x86)/Tesseract-OCR/tesseract.exe
```

That fallback helps when Tesseract is installed but the current terminal session has not refreshed its PATH yet.

## Commands

Run OCR on the synthetic receipt using the default preprocessing path:

```bash
python -m backend.scripts.run_ocr sample_data/synthetic/synthetic_receipt.png
```

Save an OCR bounding-box preview:

```bash
python -m backend.scripts.run_ocr sample_data/synthetic/synthetic_receipt.png --annotate-output processed/synthetic_receipt_ocr_boxes.png
```

Save a preview with text labels above the boxes:

```bash
python -m backend.scripts.run_ocr sample_data/synthetic/synthetic_receipt.png --annotate-output processed/synthetic_receipt_ocr_boxes_labeled.png --annotate-labels
```

Try OCR with thresholding enabled:

```bash
python -m backend.scripts.run_ocr sample_data/synthetic/synthetic_receipt.png --threshold --threshold-value 175
```

Generated outputs should go in `processed/`, which is ignored by Git.

Save OCR blocks and the extracted total as JSON:

```bash
python -m backend.scripts.run_ocr sample_data/synthetic/synthetic_receipt.png --json-output processed/synthetic_receipt_results.json
```

The JSON contains `blocks` (original OCR text, coordinates, and confidence) and
`fields.total` (value, source text, OCR confidence, extraction confidence, and
`needs_review`). A missing total is saved as `"total": null`; it is not an error.
Amounts remain strings. Extraction confidence is a rule score, not a measured
probability. Review flags are extraction hints, not business-rule validation.

The output must have a `.json` extension. Missing parent folders are created,
and an existing output file is overwritten. Write failures propagate as errors.
The JSON option can be combined with the preview and preprocessing options.
Without it, the script continues to print OCR blocks without saving JSON.

Run the tests:

```bash
python -m pytest backend/tests
```

## OCRBlock Format

Each detected text block is normalized into this shape:

```python
OCRBlock(
    text="TOTAL",
    x=60,
    y=428,
    width=73,
    height=17,
    confidence=0.96,
)
```

Coordinate meanings:

- `x`: left edge of the detected text box, measured in pixels from the left side of the image.
- `y`: top edge of the detected text box, measured in pixels from the top of the image.
- `width`: box width in pixels.
- `height`: box height in pixels.
- `confidence`: OCR confidence normalized from `0.0` to `1.0`, or `None` if confidence is unavailable.

Tesseract reports confidence as a percentage-like value. The project stores it as a decimal, so `96` becomes `0.96`.

## Bounding-Box Preview

The preview image draws one rectangle for each `OCRBlock`.

Boxes are drawn on a copy of the processed image, so the original image array is not modified.

OpenCV uses BGR color order. The default box color is green:

```python
(0, 255, 0)
```

Bounding boxes come from Tesseract, not from our drawing code. If a box looks too wide, too tall, or overlaps another box, that usually means Tesseract estimated the text region that way.

## Current Observations

On the synthetic receipt, plain grayscale preprocessing currently gives the best OCR result.

Thresholding can still be useful for messy real-world scans or phone photos, but on a clean synthetic image it may damage letter shapes or merge nearby text.

Denoising can help with speckled images, but it can also blur or alter small text. It should stay optional until measurements show it helps.

## Current Limitations

- Local JPG/PNG images only.
- No PDF input yet.
- Total extraction supports English labels and amounts with two decimal places;
  amounts must occupy one OCR block. Layout matching uses fixed pixel tolerances.
- No extraction of vendor, date, invoice number, subtotal, or tax yet.
- No business-rule validation yet.
- No API, database, frontend, ML training, or Docker yet.
- OCR quality depends on the local Tesseract installation and the input image quality.
- OCR bounding boxes are useful but not perfect; later extraction code should not assume every box is exact.

## Next Step

Continue Milestone 3 with subtotal and tax extraction after reviewing total extraction on sample documents.
