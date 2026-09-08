# DocuLens

DocuLens extracts structured information from receipts and invoices using
computer vision, OCR, and document analysis.

## Planned Features

- Document image preprocessing
- OCR with text localization
- Structured field extraction
- Data validation
- REST API
- Database persistence
- Document review interface

## Status

Currently under development.

Milestone 1 can load local JPG/PNG images, resize when needed, convert to grayscale, optionally denoise, optionally threshold, and save processed output locally.

Milestone 2 can run local Tesseract OCR, normalize detected text into OCR blocks, preserve bounding boxes and confidence, and save annotated OCR preview images.

Field extraction, validation, API, database, frontend, ML training, and Docker are not implemented yet.

## Docs

- [Preprocessing workflow](docs/preprocessing.md)
- [OCR workflow](docs/ocr.md)
