import json
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from backend.app.image_loader import load_image
from backend.app.ocr import extract_text_blocks
from backend.app.ocr_models import OCRBlock
from backend.app.ocr_visualization import draw_ocr_blocks
from backend.app.preprocessing import preprocess_image, save_processed_image
from backend.app.total_extractor import extract_total


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
    parser.add_argument(
        "--annotate-output",
        type=Path,
        help="Optional path where an OCR bounding-box preview should be saved.",
    )
    parser.add_argument(
        "--annotate-labels",
        action="store_true",
        help="Draw detected text labels above OCR boxes in the preview image.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON output path for OCR blocks and the extracted total.",
    )
    return parser.parse_args(argv)


def run_ocr_pipeline(
    input_path: str | Path,
    max_width: int = 1600,
    max_height: int = 1600,
    apply_denoise: bool = False,
    denoise_kernel_size: int = 3,
    apply_threshold: bool = False,
    threshold_value: int = 127,
) -> tuple[np.ndarray, list[OCRBlock]]:
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
    blocks = extract_text_blocks(processed_image)

    return processed_image, blocks


def run_ocr_on_local_image(
    input_path: str | Path,
    max_width: int = 1600,
    max_height: int = 1600,
    apply_denoise: bool = False,
    denoise_kernel_size: int = 3,
    apply_threshold: bool = False,
    threshold_value: int = 127,
) -> list[OCRBlock]:
    _processed_image, blocks = run_ocr_pipeline(
        input_path,
        max_width=max_width,
        max_height=max_height,
        apply_denoise=apply_denoise,
        denoise_kernel_size=denoise_kernel_size,
        apply_threshold=apply_threshold,
        threshold_value=threshold_value,
    )
    return blocks


def save_ocr_preview(
    image: np.ndarray,
    blocks: Sequence[OCRBlock],
    output_path: str | Path,
    include_labels: bool = False,
) -> Path:
    annotated = draw_ocr_blocks(image, blocks, include_labels=include_labels)
    return save_processed_image(annotated, output_path)


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


def save_ocr_results(blocks: Sequence[OCRBlock], output_path: str | Path) -> Path:
    """Save OCR evidence and total extraction, using null for a missing total."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".json":
        raise ValueError("OCR results output path must use the .json extension")

    total = extract_total(blocks)
    results = {
        "blocks": [asdict(block) for block in blocks],
        "fields": {"total": asdict(total) if total is not None else None},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    processed_image, blocks = run_ocr_pipeline(
        args.input_path,
        max_width=args.max_width,
        max_height=args.max_height,
        apply_denoise=args.denoise,
        denoise_kernel_size=args.denoise_kernel_size,
        apply_threshold=args.threshold,
        threshold_value=args.threshold_value,
    )
    print_ocr_blocks(blocks)

    if args.json_output is not None:
        saved_path = save_ocr_results(blocks, args.json_output)
        print(f"Saved OCR results to {saved_path}")

    if args.annotate_output is not None:
        saved_path = save_ocr_preview(
            processed_image,
            blocks,
            args.annotate_output,
            include_labels=args.annotate_labels,
        )
        print(f"Saved OCR preview to {saved_path}")


def _format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"

    return f"{confidence:.2f}"


if __name__ == "__main__":
    main()
