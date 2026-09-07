from collections.abc import Mapping, Sequence
from pathlib import Path
from shutil import which

import cv2
import numpy as np

from backend.app.ocr_models import OCRBlock

REQUIRED_TESSERACT_KEYS = ("text", "left", "top", "width", "height", "conf")
WINDOWS_TESSERACT_PATHS = (
    Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
)


def extract_text_blocks(image: np.ndarray) -> list[OCRBlock]:
    """Run OCR on an OpenCV image and return normalized OCR blocks."""
    _validate_ocr_image(image)
    tesseract_image = _prepare_image_for_tesseract(image)
    data = _image_to_tesseract_data(tesseract_image)

    return normalize_tesseract_data(data)


def normalize_tesseract_data(data: Mapping[str, Sequence[object]]) -> list[OCRBlock]:
    """Convert Tesseract-style OCR rows into normalized OCR blocks."""
    _validate_tesseract_data(data)

    blocks: list[OCRBlock] = []
    row_count = len(data["text"])

    for index in range(row_count):
        text = str(data["text"][index]).strip()

        if not text:
            continue

        block = _normalize_tesseract_row(data, index, text)

        if block is not None:
            blocks.append(block)

    return blocks


def _validate_ocr_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    if image.ndim == 2:
        return

    if image.ndim == 3 and image.shape[2] == 3:
        return

    raise ValueError("image must be a grayscale or 3-channel color image")


def _prepare_image_for_tesseract(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _image_to_tesseract_data(image: np.ndarray) -> Mapping[str, Sequence[object]]:
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise RuntimeError("pytesseract is required to run OCR") from exc

    _configure_tesseract_command(pytesseract)

    try:
        return pytesseract.image_to_data(image, output_type=Output.DICT)
    except pytesseract.pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract OCR is not installed or is not on PATH") from exc


def _configure_tesseract_command(pytesseract_module) -> None:
    if which("tesseract") is not None:
        return

    for path in WINDOWS_TESSERACT_PATHS:
        if path.exists():
            pytesseract_module.pytesseract.tesseract_cmd = str(path)
            return


def _validate_tesseract_data(data: Mapping[str, Sequence[object]]) -> None:
    missing_keys = [key for key in REQUIRED_TESSERACT_KEYS if key not in data]

    if missing_keys:
        raise ValueError(f"Missing Tesseract OCR keys: {', '.join(missing_keys)}")

    row_count = len(data["text"])

    for key in REQUIRED_TESSERACT_KEYS:
        if len(data[key]) != row_count:
            raise ValueError("Tesseract OCR columns must have the same length")


def _normalize_tesseract_row(
    data: Mapping[str, Sequence[object]],
    index: int,
    text: str,
) -> OCRBlock | None:
    try:
        x = _parse_int(data["left"][index])
        y = _parse_int(data["top"][index])
        width = _parse_int(data["width"][index])
        height = _parse_int(data["height"][index])
    except (TypeError, ValueError):
        return None

    try:
        confidence = _normalize_tesseract_confidence(data["conf"][index])
        return OCRBlock(
            text=text,
            x=x,
            y=y,
            width=width,
            height=height,
            confidence=confidence,
        )
    except ValueError:
        return None


def _parse_int(value: object) -> int:
    return int(str(value).strip())


def _normalize_tesseract_confidence(raw_confidence: object) -> float | None:
    confidence_text = str(raw_confidence).strip()

    if not confidence_text:
        return None

    confidence = float(confidence_text)

    if confidence < 0:
        return None

    if confidence > 100:
        raise ValueError("Tesseract confidence must be between -1 and 100")

    return confidence / 100
