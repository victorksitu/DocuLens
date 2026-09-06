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

Milestone 1 focuses on local image preprocessing. The current preprocessing layer can load JPG/PNG images, resize when needed, convert to grayscale, optionally denoise, optionally threshold, and save processed output locally.

See [docs/preprocessing.md](docs/preprocessing.md) for the current preprocessing workflow and sample commands.
