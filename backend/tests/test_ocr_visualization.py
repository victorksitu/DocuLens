import numpy as np
import pytest

from backend.app.ocr_models import OCRBlock
from backend.app.ocr_visualization import draw_ocr_blocks


def test_draw_ocr_blocks_draws_rectangle_on_grayscale_copy() -> None:
    image = np.full((20, 30), fill_value=255, dtype=np.uint8)
    block = OCRBlock(text="TOTAL", x=5, y=4, width=10, height=6, confidence=0.96)

    annotated = draw_ocr_blocks(image, [block], thickness=1)

    assert annotated.shape == (20, 30, 3)
    assert image[4, 5] == 255
    assert annotated[4, 5].tolist() == [0, 255, 0]


def test_draw_ocr_blocks_draws_rectangle_on_color_copy() -> None:
    image = np.full((20, 30, 3), fill_value=255, dtype=np.uint8)
    block = OCRBlock(text="TOTAL", x=5, y=4, width=10, height=6, confidence=0.96)

    annotated = draw_ocr_blocks(image, [block], box_color=(255, 0, 0), thickness=1)

    assert annotated.shape == image.shape
    assert image[4, 5].tolist() == [255, 255, 255]
    assert annotated[4, 5].tolist() == [255, 0, 0]


def test_draw_ocr_blocks_can_include_text_labels() -> None:
    image = np.full((30, 80), fill_value=255, dtype=np.uint8)
    block = OCRBlock(text="TOTAL", x=5, y=20, width=30, height=8, confidence=0.96)

    annotated = draw_ocr_blocks(image, [block], include_labels=True)

    assert annotated.shape == (30, 80, 3)
    assert not np.array_equal(annotated, np.full((30, 80, 3), fill_value=255, dtype=np.uint8))


def test_draw_ocr_blocks_rejects_empty_images() -> None:
    image = np.array([], dtype=np.uint8)

    with pytest.raises(ValueError, match="must not be empty"):
        draw_ocr_blocks(image, [])


def test_draw_ocr_blocks_rejects_unsupported_image_shapes() -> None:
    image = np.zeros((10,), dtype=np.uint8)

    with pytest.raises(ValueError, match="grayscale or 3-channel color"):
        draw_ocr_blocks(image, [])


def test_draw_ocr_blocks_rejects_invalid_thickness() -> None:
    image = np.zeros((10, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="thickness"):
        draw_ocr_blocks(image, [], thickness=0)
