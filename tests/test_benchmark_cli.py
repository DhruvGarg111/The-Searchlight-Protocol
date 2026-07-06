from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_cli_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "benchmarks/run_pipeline_benchmark.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Benchmark Searchlight pipeline runs." in result.stdout
    assert "--fixture-dir" in result.stdout
    assert "--baseline-name" in result.stdout


def test_benchmark_compare_cli_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "benchmarks/compare_pipeline_benchmarks.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Compare two Searchlight benchmark JSON files." in result.stdout
    assert "--min-stage-improvement" in result.stdout
    assert "--max-total-regression" in result.stdout


def test_benchmark_compare_rejects_detection_quality_delta(tmp_path: Path) -> None:
    baseline = {
        "records": [
            {
                "image": "fixture.png",
                "run_index": 0,
                "pre_nms_crops": 1,
                "post_nms_crops": 1,
                "detection_count": 1,
                "output_keys": ["final_detections"],
                "crop_image_count": 0,
                "payload_bytes": 100,
                "timings_ms": {"layercam_generation": 100.0, "total_pipeline": 200.0},
                "crop_records": [
                    {"id": 1, "bbox": [0, 0, 20, 20], "score": 0.75, "has_image": False},
                ],
                "detection_records": [
                    {
                        "crop_id": 1,
                        "class": "target",
                        "confidence": 0.9,
                        "global_bbox": [1.0, 2.0, 12.0, 14.0],
                    },
                ],
                "output_hashes": {"final_detections": "same"},
            },
        ],
    }
    candidate = json.loads(json.dumps(baseline))
    candidate["records"][0]["payload_bytes"] = 90
    candidate["records"][0]["timings_ms"]["layercam_generation"] = 80.0
    candidate["records"][0]["timings_ms"]["total_pipeline"] = 180.0
    candidate["records"][0]["detection_records"][0]["global_bbox"] = [5.0, 2.0, 12.0, 14.0]

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/compare_pipeline_benchmarks.py",
            str(baseline_path),
            str(candidate_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "detection 1 bbox changed" in result.stderr
