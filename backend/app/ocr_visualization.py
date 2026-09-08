from collections.abc import Sequence

import cv2
import numpy as np

from backend.app.ocr_models import OCRBlock

Color = tuple[int, int, int]
DEFAULT_BOX_COLOR: Color = (0, 255, 0)
DEFAULT_TEXT_COLOR: Color = (0, 0, 255)


def draw_ocr_blocks(
    image: np.ndarray,
    blocks: Sequence[OCRBlock],
    box_color: Color = DEFAULT_BOX_COLOR,
    text_color: Color = DEFAULT_TEXT_COLOR,
    thickness: int = 2,
    include_labels: bool = False,
) -> np.ndarray:
    """Draw OCR bounding boxes on a copy of an OpenCV image."""
    _validate_visualization_image(image)
    _validate_thickness(thickness)

    annotated = _as_bgr_copy(image)

    for block in blocks:
        top_left = (block.x, block.y)
        bottom_right = (block.x + block.width, block.y + block.height)
        cv2.rectangle(annotated, top_left, bottom_right, box_color, thickness)

        if include_labels:
            _draw_label(annotated, block, text_color)

    return annotated


def _as_bgr_copy(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.copy()


def _draw_label(image: np.ndarray, block: OCRBlock, text_color: Color) -> None:
    label_y = max(block.y - 5, 12)
    cv2.putText(
        image,
        block.text,
        (block.x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        text_color,
        1,
        cv2.LINE_AA,
    )


def _validate_visualization_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    if image.ndim == 2:
        return

    if image.ndim == 3 and image.shape[2] == 3:
        return

    raise ValueError("image must be a grayscale or 3-channel color image")


def _validate_thickness(thickness: int) -> None:
    if thickness <= 0:
        raise ValueError("thickness must be greater than 0")
