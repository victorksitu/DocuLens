import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from backend.app.ocr_models import OCRBlock
from backend.scripts import run_ocr


def test_run_ocr_pipeline_loads_preprocesses_and_extracts_blocks(
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

    actual_processed_image, blocks = run_ocr.run_ocr_pipeline(
        input_path,
        max_width=100,
        max_height=200,
        apply_denoise=True,
        denoise_kernel_size=5,
        apply_threshold=True,
        threshold_value=150,
    )

    assert actual_processed_image is processed_image
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


def test_run_ocr_on_local_image_returns_only_blocks(monkeypatch) -> None:
    expected_blocks = [
        OCRBlock(text="TOTAL", x=10, y=20, width=30, height=12, confidence=0.96)
    ]

    def fake_run_ocr_pipeline(_input_path, **_kwargs):
        return np.zeros((20, 40), dtype=np.uint8), expected_blocks

    monkeypatch.setattr(run_ocr, "run_ocr_pipeline", fake_run_ocr_pipeline)

    blocks = run_ocr.run_ocr_on_local_image("document.png")

    assert blocks == expected_blocks


def test_save_ocr_preview_draws_and_saves_annotated_image(monkeypatch) -> None:
    image = np.full((20, 40), fill_value=200, dtype=np.uint8)
    annotated_image = np.full((20, 40, 3), fill_value=100, dtype=np.uint8)
    blocks = [
        OCRBlock(text="TOTAL", x=10, y=20, width=30, height=12, confidence=0.96)
    ]
    output_path = Path("processed/ocr_boxes.png")
    calls = {}

    def fake_draw_ocr_blocks(received_image, received_blocks, include_labels=False):
        calls["draw_image"] = received_image
        calls["draw_blocks"] = received_blocks
        calls["include_labels"] = include_labels
        return annotated_image

    def fake_save_processed_image(received_image, received_path):
        calls["save_image"] = received_image
        calls["save_path"] = received_path
        return Path(received_path)

    monkeypatch.setattr(run_ocr, "draw_ocr_blocks", fake_draw_ocr_blocks)
    monkeypatch.setattr(run_ocr, "save_processed_image", fake_save_processed_image)

    saved_path = run_ocr.save_ocr_preview(
        image,
        blocks,
        output_path,
        include_labels=True,
    )

    assert saved_path == output_path
    assert calls["draw_image"] is image
    assert calls["draw_blocks"] == blocks
    assert calls["include_labels"] is True
    assert calls["save_image"] is annotated_image
    assert calls["save_path"] == output_path


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


def test_main_passes_cli_options_to_pipeline_and_saves_preview(
    monkeypatch,
    capsys,
) -> None:
    processed_image = np.full((20, 40), fill_value=200, dtype=np.uint8)
    blocks = [
        OCRBlock(text="TOTAL", x=10, y=20, width=30, height=12, confidence=0.96)
    ]
    captured = {}

    def fake_run_ocr_pipeline(received_path, **kwargs):
        captured["input_path"] = received_path
        captured["options"] = kwargs
        return processed_image, blocks

    def fake_save_ocr_preview(received_image, received_blocks, received_path, **kwargs):
        captured["preview_image"] = received_image
        captured["preview_blocks"] = received_blocks
        captured["preview_path"] = received_path
        captured["preview_options"] = kwargs
        return Path(received_path)

    monkeypatch.setattr(run_ocr, "run_ocr_pipeline", fake_run_ocr_pipeline)
    monkeypatch.setattr(run_ocr, "save_ocr_preview", fake_save_ocr_preview)

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
            "--annotate-output",
            "processed/ocr_boxes.png",
            "--annotate-labels",
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
    assert captured["preview_image"] is processed_image
    assert captured["preview_blocks"] == blocks
    assert captured["preview_path"] == Path("processed/ocr_boxes.png")
    assert captured["preview_options"] == {"include_labels": True}
    assert '"TOTAL" x=10 y=20 width=30 height=12 confidence=0.96' in output
    assert "Saved OCR preview to " in output
    assert "ocr_boxes.png" in output


def test_main_saves_total_and_ocr_evidence_to_json(monkeypatch, capsys) -> None:
    blocks = [
        OCRBlock("TOTAL", 10, 20, 50, 12, 0.96),
        OCRBlock("$22.55", 100, 20, 60, 12, 0.92),
    ]
    monkeypatch.setattr(
        run_ocr, "run_ocr_pipeline",
        lambda *_args, **_kwargs: (np.zeros((50, 200), dtype=np.uint8), blocks),
    )
    with TemporaryDirectory(dir=Path(__file__).parent) as directory:
        output_path = Path(directory) / "nested" / "results.json"
        run_ocr.main(["receipt.png", "--json-output", str(output_path)])
        results = json.loads(output_path.read_text(encoding="utf-8"))

    assert results["blocks"] == [asdict(block) for block in blocks]
    assert results["fields"]["total"] == {
        "field_name": "total",
        "value": "22.55",
        "source_text": "TOTAL $22.55",
        "ocr_confidence": 0.92,
        "extraction_confidence": 0.9,
        "needs_review": False,
    }
    assert "Saved OCR results to" in capsys.readouterr().out


@pytest.mark.parametrize("blocks", [[], [OCRBlock("Store", 10, 20, 50, 12)]])
def test_main_saves_null_when_total_is_missing(monkeypatch, blocks) -> None:
    monkeypatch.setattr(
        run_ocr, "run_ocr_pipeline",
        lambda *_args, **_kwargs: (np.zeros((50, 200), dtype=np.uint8), blocks),
    )
    with TemporaryDirectory(dir=Path(__file__).parent) as directory:
        output_path = Path(directory) / "results.json"
        run_ocr.main(["receipt.png", "--json-output", str(output_path)])
        results = json.loads(output_path.read_text(encoding="utf-8"))

    assert results == {
        "blocks": [asdict(block) for block in blocks],
        "fields": {"total": None},
    }


def test_save_ocr_results_preserves_review_flag_and_unknown_confidence() -> None:
    blocks = [
        OCRBlock("TOTAL", 10, 20, 50, 12),
        OCRBlock("22.55", 100, 20, 60, 12, 0.4),
    ]
    with TemporaryDirectory(dir=Path(__file__).parent) as directory:
        output_path = Path(directory) / "results.JSON"
        assert run_ocr.save_ocr_results(blocks, output_path) == output_path
        results = json.loads(output_path.read_text(encoding="utf-8"))

    assert results["fields"]["total"]["needs_review"] is True
    assert results["fields"]["total"]["ocr_confidence"] is None


def test_save_ocr_results_rejects_non_json_path() -> None:
    with pytest.raises(ValueError, match=".json extension"):
        run_ocr.save_ocr_results([], "results.png")


def test_save_ocr_results_propagates_write_failure(monkeypatch) -> None:
    def fail_write(*_args, **_kwargs):
        raise PermissionError("Cannot write results")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with pytest.raises(PermissionError, match="Cannot write results"):
            run_ocr.save_ocr_results([], Path(directory) / "results.json")


def test_main_without_json_option_does_not_save_results(monkeypatch) -> None:
    monkeypatch.setattr(
        run_ocr, "run_ocr_pipeline",
        lambda *_args, **_kwargs: (np.zeros((20, 40), dtype=np.uint8), []),
    )

    def unexpected_save(*_args, **_kwargs):
        pytest.fail("JSON saving should be opt-in")

    monkeypatch.setattr(run_ocr, "save_ocr_results", unexpected_save)
    run_ocr.main(["receipt.png"])

