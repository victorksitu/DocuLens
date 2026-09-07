import numpy as np
import pytest

from backend.app import ocr
from backend.app.ocr import extract_text_blocks, normalize_tesseract_data
from backend.app.ocr_models import OCRBlock


def test_extract_text_blocks_runs_ocr_and_normalizes_result(monkeypatch) -> None:
    image = np.zeros((40, 80), dtype=np.uint8)
    data = {
        "text": ["TOTAL"],
        "left": [10],
        "top": [20],
        "width": [30],
        "height": [12],
        "conf": ["96"],
    }

    def fake_image_to_tesseract_data(received_image):
        assert np.array_equal(received_image, image)
        return data

    monkeypatch.setattr(
        "backend.app.ocr._image_to_tesseract_data",
        fake_image_to_tesseract_data,
    )

    blocks = extract_text_blocks(image)

    assert blocks == [
        OCRBlock(text="TOTAL", x=10, y=20, width=30, height=12, confidence=0.96)
    ]


def test_extract_text_blocks_converts_color_image_from_bgr_to_rgb(monkeypatch) -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    data = {
        "text": ["A"],
        "left": [0],
        "top": [0],
        "width": [1],
        "height": [1],
        "conf": ["90"],
    }
    captured = {}

    def fake_image_to_tesseract_data(received_image):
        captured["pixel"] = received_image[0, 0].tolist()
        return data

    monkeypatch.setattr(
        "backend.app.ocr._image_to_tesseract_data",
        fake_image_to_tesseract_data,
    )

    extract_text_blocks(image)

    assert captured["pixel"] == [30, 20, 10]


def test_extract_text_blocks_rejects_empty_images() -> None:
    image = np.array([], dtype=np.uint8)

    with pytest.raises(ValueError, match="must not be empty"):
        extract_text_blocks(image)


def test_extract_text_blocks_rejects_unsupported_image_shapes() -> None:
    image = np.zeros((10,), dtype=np.uint8)

    with pytest.raises(ValueError, match="grayscale or 3-channel color"):
        extract_text_blocks(image)


def test_configure_tesseract_command_uses_default_windows_path(monkeypatch) -> None:
    class FakePath:
        def exists(self) -> bool:
            return True

        def __str__(self) -> str:
            return "C:/Program Files/Tesseract-OCR/tesseract.exe"

    class FakePytesseract:
        class pytesseract:
            tesseract_cmd = "tesseract"

    tesseract_path = FakePath()

    monkeypatch.setattr(ocr, "which", lambda _command: None)
    monkeypatch.setattr(ocr, "WINDOWS_TESSERACT_PATHS", (tesseract_path,))

    ocr._configure_tesseract_command(FakePytesseract)

    assert (
        FakePytesseract.pytesseract.tesseract_cmd
        == "C:/Program Files/Tesseract-OCR/tesseract.exe"
    )


def test_normalize_tesseract_data_converts_rows_to_ocr_blocks() -> None:
    data = {
        "text": ["TOTAL", "22.55"],
        "left": [60, 150],
        "top": [340, 340],
        "width": [80, 70],
        "height": [25, 25],
        "conf": ["96", "91"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks == [
        OCRBlock(text="TOTAL", x=60, y=340, width=80, height=25, confidence=0.96),
        OCRBlock(text="22.55", x=150, y=340, width=70, height=25, confidence=0.91),
    ]


def test_normalize_tesseract_data_strips_text() -> None:
    data = {
        "text": ["  TOTAL  "],
        "left": [60],
        "top": [340],
        "width": [80],
        "height": [25],
        "conf": ["96"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks[0].text == "TOTAL"


def test_normalize_tesseract_data_skips_empty_text_rows() -> None:
    data = {
        "text": ["", "   ", "TOTAL"],
        "left": [0, 0, 60],
        "top": [0, 0, 340],
        "width": [0, 0, 80],
        "height": [0, 0, 25],
        "conf": ["-1", "-1", "96"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks == [
        OCRBlock(text="TOTAL", x=60, y=340, width=80, height=25, confidence=0.96)
    ]


def test_normalize_tesseract_data_skips_invalid_boxes() -> None:
    data = {
        "text": ["BAD", "TOTAL", "ALSO_BAD"],
        "left": [-1, 60, 90],
        "top": [0, 340, 100],
        "width": [10, 80, 0],
        "height": [10, 25, 10],
        "conf": ["90", "96", "90"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks == [
        OCRBlock(text="TOTAL", x=60, y=340, width=80, height=25, confidence=0.96)
    ]


def test_normalize_tesseract_data_converts_negative_confidence_to_none() -> None:
    data = {
        "text": ["TOTAL"],
        "left": [60],
        "top": [340],
        "width": [80],
        "height": [25],
        "conf": ["-1"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks == [
        OCRBlock(text="TOTAL", x=60, y=340, width=80, height=25, confidence=None)
    ]


def test_normalize_tesseract_data_converts_decimal_confidence() -> None:
    data = {
        "text": ["TOTAL"],
        "left": [60],
        "top": [340],
        "width": [80],
        "height": [25],
        "conf": ["95.5"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks[0].confidence == 0.955


def test_normalize_tesseract_data_rejects_missing_required_keys() -> None:
    data = {
        "text": ["TOTAL"],
        "left": [60],
        "top": [340],
        "width": [80],
        "height": [25],
    }

    with pytest.raises(ValueError, match="Missing Tesseract OCR keys"):
        normalize_tesseract_data(data)


def test_normalize_tesseract_data_rejects_mismatched_column_lengths() -> None:
    data = {
        "text": ["TOTAL", "22.55"],
        "left": [60],
        "top": [340, 340],
        "width": [80, 70],
        "height": [25, 25],
        "conf": ["96", "91"],
    }

    with pytest.raises(ValueError, match="same length"):
        normalize_tesseract_data(data)


