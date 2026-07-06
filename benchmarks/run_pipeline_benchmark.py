from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Searchlight pipeline runs.")
    parser.add_argument(
        "--fixture-dir",
        default=os.getenv("SEARCHLIGHT_BENCHMARK_FIXTURE_DIR"),
        help="Directory of fixture images. Defaults to SEARCHLIGHT_BENCHMARK_FIXTURE_DIR.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Specific image path to benchmark. Can be passed multiple times.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Measured runs per image.")
    parser.add_argument("--warm-runs", type=int, default=1, help="Warmup runs per image.")
    parser.add_argument(
        "--profile",
        choices=("full", "display", "metadata"),
        default="full",
        help="Response profile to benchmark.",
    )
    parser.add_argument(
        "--enable-global-nms",
        action="store_true",
        help="Enable final global detection NMS for this benchmark.",
    )
    parser.add_argument(
        "--output",
        help="Explicit JSON output file. Defaults to a temp directory outside source.",
    )
    parser.add_argument(
        "--baseline-name",
        help="Write a curated summary to benchmarks/baselines/<name>.json.",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root to import for benchmarking. Defaults to this repository.",
    )
    return parser.parse_args()


def _discover_images(args: argparse.Namespace) -> list[Path]:
    image_paths = [Path(path).resolve() for path in args.image]
    if args.fixture_dir:
        fixture_dir = Path(args.fixture_dir).resolve()
        image_paths.extend(
            sorted(
                path
                for path in fixture_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
        )

    unique_paths = []
    seen = set()
    for image_path in image_paths:
        if image_path not in seen:
            seen.add(image_path)
            unique_paths.append(image_path)

    missing = [str(path) for path in unique_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Benchmark image(s) not found: {missing}")
    if not unique_paths:
        raise ValueError("Provide --image or --fixture-dir/SEARCHLIGHT_BENCHMARK_FIXTURE_DIR.")
    return unique_paths


def _output_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.baseline_name:
        baseline_dir = PROJECT_ROOT / "benchmarks" / "baselines"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        return baseline_dir / f"{args.baseline_name}.json"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    output_dir = Path(os.getenv("SEARCHLIGHT_BENCHMARK_OUTPUT_DIR", tempfile.gettempdir()))
    output_dir = output_dir / "searchlight-benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"benchmark-{timestamp}.json"


def _payload_size_bytes(result: dict[str, Any]) -> int:
    return len(json.dumps(result, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _load_runtime(project_root: str):
    root = Path(project_root).resolve()
    backend_dir = root / "webapp" / "backend"
    for import_path in (str(backend_dir), str(root)):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)

    try:
        import cv2
        import torch
        from webapp.backend.core.config import get_config
        from webapp.backend.models.pipeline import PipelineSettings
        from webapp.backend.services.pipeline_service import SearchlightPipelineService
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Benchmarking requires project ML dependencies. Install requirements.txt "
            "and webapp/backend/requirements.txt first.",
        ) from exc

    if not hasattr(cv2, "COLOR_RGB2RGB"):
        cv2.COLOR_RGB2RGB = cv2.COLOR_BGR2RGB

    return torch, get_config, PipelineSettings, SearchlightPipelineService


def _build_settings(PipelineSettings: Any, args: argparse.Namespace) -> Any:
    fields = getattr(PipelineSettings, "model_fields", {})
    kwargs: dict[str, Any] = {}
    if "response_profile" in fields:
        kwargs["response_profile"] = args.profile
    if "enable_global_nms" in fields:
        kwargs["enable_global_nms"] = args.enable_global_nms
    return PipelineSettings(**kwargs)


def _run_once(
    torch_module,
    service: Any,
    image_path: Path,
    settings: Any,
) -> dict[str, Any]:
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    started = time.perf_counter()
    result = service.run_from_path(str(image_path), settings)
    wall_ms = (time.perf_counter() - started) * 1000.0
    _, peak_traced_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    counts = result["counts"]
    research = result["research"]
    meta = result["meta"]
    output_keys = sorted(result["outputs"].keys())
    crop_image_count = sum(1 for crop in result["crops"] if crop.get("image"))
    crop_records = [
        {
            "id": int(crop["id"]),
            "bbox": [int(value) for value in crop["bbox"]],
            "score": round(float(crop["score"]), 6),
            "has_image": bool(crop.get("image")),
        }
        for crop in result["crops"]
    ]
    detection_records = [
        {
            "crop_id": int(detection["crop_id"]),
            "class": str(detection["class"]),
            "confidence": round(float(detection["confidence"]), 6),
            "global_bbox": [round(float(value), 4) for value in detection["global_bbox"]],
        }
        for detection in result["detections"]
    ]

    return {
        "image": image_path.name,
        "dimensions": list(_image_dimensions(image_path)),
        "device": meta["device"],
        "torch_version": meta["torch_version"],
        "cuda_available": meta["cuda_available"],
        "wall_ms": round(wall_ms, 2),
        "timings_ms": research["timings_ms"],
        "pre_nms_crops": counts["pre_nms_crops"],
        "post_nms_crops": counts["post_nms_crops"],
        "detection_count": counts["detections"],
        "payload_bytes": _payload_size_bytes(result),
        "output_keys": output_keys,
        "output_hashes": {
            key: _sha256_text(result["outputs"][key])
            for key in output_keys
        },
        "crop_records": crop_records,
        "crop_image_count": crop_image_count,
        "detection_records": detection_records,
        "tracemalloc_peak_bytes": peak_traced_bytes,
        "cuda_peak_memory_bytes": (
            int(torch_module.cuda.max_memory_allocated())
            if torch_module.cuda.is_available()
            else 0
        ),
    }


def main() -> int:
    args = _parse_args()
    image_paths = _discover_images(args)
    torch, get_config, PipelineSettings, SearchlightPipelineService = _load_runtime(args.project_root)
    settings = _build_settings(PipelineSettings, args)
    config = get_config()
    service = SearchlightPipelineService(config)

    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        for _ in range(max(0, args.warm_runs)):
            service.run_from_path(str(image_path), settings)

        for run_index in range(max(1, args.runs)):
            record = _run_once(torch, service, image_path, settings)
            record["run_index"] = run_index
            records.append(record)

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "global_nms_enabled": args.enable_global_nms,
        "config": {
            "resnet_input_max_dim": getattr(config, "resnet_input_max_dim", 1800),
            "response_display_max_dim": getattr(config, "response_display_max_dim", None),
            "response_display_crop_limit": getattr(config, "response_display_crop_limit", None),
            "response_display_format": getattr(config, "response_display_format", None),
            "yolo_model_version": config.yolo_model_version,
            "yolo_model_variant": config.yolo_model_variant,
            "yolo_model_path": str(config.yolo_model_path),
        },
        "records": records,
    }

    output_path = _output_path(args)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
