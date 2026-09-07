from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

from backend.app.image_loader import load_image
from backend.app.ocr import extract_text_blocks
from backend.app.ocr_models import OCRBlock
from backend.app.preprocessing import preprocess_image


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Run OCR on a local document image.")
    parser.add_argument("input_path", type=Path, help="Path to the source JPG/PNG image.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Maximum image width before OCR.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1600,
        help="Maximum image height before OCR.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply a gentle median filter before OCR.",
    )
    parser.add_argument(
        "--denoise-kernel-size",
        type=int,
        default=3,
        help="Odd median filter size for denoising. Use 3 for a gentle default.",
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Convert the grayscale image to black-and-white before OCR.",
    )
    parser.add_argument(
        "--threshold-value",
        type=int,
        default=127,
        help="Pixel cutoff for thresholding. Pixels above this become white.",
    )
    return parser.parse_args(argv)


def run_ocr_on_local_image(
    input_path: str | Path,
    max_width: int = 1600,
    max_height: int = 1600,
    apply_denoise: bool = False,
    denoise_kernel_size: int = 3,
    apply_threshold: bool = False,
    threshold_value: int = 127,
) -> list[OCRBlock]:
    image = load_image(input_path)
    processed_image = preprocess_image(
        image,
        max_width=max_width,
        max_height=max_height,
        apply_denoise=apply_denoise,
        denoise_kernel_size=denoise_kernel_size,
        apply_threshold=apply_threshold,
        threshold_value=threshold_value,
    )
    return extract_text_blocks(processed_image)


def print_ocr_blocks(blocks: Sequence[OCRBlock]) -> None:
    if not blocks:
        print("No text detected.")
        return

    print(f"Detected {len(blocks)} text block(s):")

    for block in blocks:
        confidence = _format_confidence(block.confidence)
        print(
            f'- "{block.text}" '
            f"x={block.x} y={block.y} "
            f"width={block.width} height={block.height} "
            f"confidence={confidence}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    blocks = run_ocr_on_local_image(
        args.input_path,
        max_width=args.max_width,
        max_height=args.max_height,
        apply_denoise=args.denoise,
        denoise_kernel_size=args.denoise_kernel_size,
        apply_threshold=args.threshold,
        threshold_value=args.threshold_value,
    )
    print_ocr_blocks(blocks)


def _format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"

    return f"{confidence:.2f}"


if __name__ == "__main__":
    main()
