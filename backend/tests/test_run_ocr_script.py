from pathlib import Path

import numpy as np

from backend.app.ocr_models import OCRBlock
from backend.scripts import run_ocr


def test_run_ocr_on_local_image_loads_preprocesses_and_extracts_blocks(
    monkeypatch,
) -> None:
    input_path = Path("sample_data/synthetic/synthetic_receipt.png")
    loaded_image = np.full((20, 40, 3), fill_value=200, dtype=np.uint8)
    processed_image = np.full((20, 40), fill_value=200, dtype=np.uint8)
    expected_blocks = [
        OCRBlock(text="TOTAL", x=10, y=20, width=30, height=12, confidence=0.96)
    ]
    calls = {}

    def fake_load_image(received_path):
        calls["input_path"] = received_path
        return loaded_image

    def fake_preprocess_image(received_image, **kwargs):
        calls["preprocess_image"] = received_image
        calls["preprocess_options"] = kwargs
        return processed_image

    def fake_extract_text_blocks(received_image):
        calls["ocr_image"] = received_image
        return expected_blocks

    monkeypatch.setattr(run_ocr, "load_image", fake_load_image)
    monkeypatch.setattr(run_ocr, "preprocess_image", fake_preprocess_image)
    monkeypatch.setattr(run_ocr, "extract_text_blocks", fake_extract_text_blocks)

    blocks = run_ocr.run_ocr_on_local_image(
        input_path,
        max_width=100,
        max_height=200,
        apply_denoise=True,
        denoise_kernel_size=5,
        apply_threshold=True,
        threshold_value=150,
    )

    assert blocks == expected_blocks
    assert calls["input_path"] == input_path
    assert calls["preprocess_image"] is loaded_image
    assert calls["preprocess_options"] == {
        "max_width": 100,
        "max_height": 200,
        "apply_denoise": True,
        "denoise_kernel_size": 5,
        "apply_threshold": True,
        "threshold_value": 150,
    }
    assert calls["ocr_image"] is processed_image


def test_print_ocr_blocks_outputs_text_boxes_and_confidence(capsys) -> None:
    blocks = [
        OCRBlock(text="TOTAL", x=10, y=20, width=30, height=12, confidence=0.96),
        OCRBlock(text="TAX", x=10, y=40, width=20, height=12, confidence=None),
    ]

    run_ocr.print_ocr_blocks(blocks)

    output = capsys.readouterr().out

    assert "Detected 2 text block(s):" in output
    assert '"TOTAL" x=10 y=20 width=30 height=12 confidence=0.96' in output
    assert '"TAX" x=10 y=40 width=20 height=12 confidence=unknown' in output


def test_print_ocr_blocks_handles_empty_results(capsys) -> None:
    run_ocr.print_ocr_blocks([])

    output = capsys.readouterr().out

    assert output == "No text detected.\n"


def test_main_passes_cli_options_to_pipeline(monkeypatch, capsys) -> None:
    captured = {}

    def fake_run_ocr_on_local_image(received_path, **kwargs):
        captured["input_path"] = received_path
        captured["options"] = kwargs
        return [
            OCRBlock(text="TOTAL", x=10, y=20, width=30, height=12, confidence=0.96)
        ]

    monkeypatch.setattr(
        run_ocr,
        "run_ocr_on_local_image",
        fake_run_ocr_on_local_image,
    )

    run_ocr.main(
        [
            "sample_data/synthetic/synthetic_receipt.png",
            "--max-width",
            "800",
            "--max-height",
            "900",
            "--denoise",
            "--denoise-kernel-size",
            "5",
            "--threshold",
            "--threshold-value",
            "150",
        ]
    )

    output = capsys.readouterr().out

    assert captured["input_path"] == Path("sample_data/synthetic/synthetic_receipt.png")
    assert captured["options"] == {
        "max_width": 800,
        "max_height": 900,
        "apply_denoise": True,
        "denoise_kernel_size": 5,
        "apply_threshold": True,
        "threshold_value": 150,
    }
    assert '"TOTAL" x=10 y=20 width=30 height=12 confidence=0.96' in output
