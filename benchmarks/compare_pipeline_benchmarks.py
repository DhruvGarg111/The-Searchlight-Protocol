from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Searchlight benchmark JSON files.")
    parser.add_argument("baseline", help="Baseline benchmark JSON file.")
    parser.add_argument("candidate", help="Candidate benchmark JSON file.")
    parser.add_argument(
        "--stage",
        default="layercam_generation",
        help="Target timing stage expected to improve by --min-stage-improvement.",
    )
    parser.add_argument(
        "--min-stage-improvement",
        type=float,
        default=0.10,
        help="Minimum fractional improvement required for --stage.",
    )
    parser.add_argument(
        "--max-total-regression",
        type=float,
        default=0.0,
        help="Maximum allowed fractional regression in total_pipeline.",
    )
    parser.add_argument(
        "--allow-payload-growth",
        action="store_true",
        help="Allow candidate serialized payload bytes to exceed baseline.",
    )
    parser.add_argument(
        "--bbox-tolerance",
        type=float,
        default=1e-3,
        help="Allowed absolute coordinate delta when comparing detection boxes.",
    )
    parser.add_argument(
        "--confidence-tolerance",
        type=float,
        default=1e-4,
        help="Allowed absolute confidence delta when comparing detections.",
    )
    parser.add_argument(
        "--crop-score-tolerance",
        type=float,
        default=1e-6,
        help="Allowed absolute score delta when comparing retained crops.",
    )
    parser.add_argument(
        "--skip-output-hash-check",
        action="store_true",
        help="Do not compare encoded visual output hashes.",
    )
    return parser.parse_args()


def _load_records(path: str) -> dict[tuple[str, int], dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} does not contain benchmark records")

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record["image"]), int(record.get("run_index", 0)))
        indexed[key] = record
    return indexed


def _fractional_delta(before: float, after: float) -> float:
    if before <= 0:
        return 0.0 if after <= before else 1.0
    return (after - before) / before


def _compare_crop_records(
    label: str,
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    score_tolerance: float,
) -> list[str]:
    errors: list[str] = []
    if len(baseline) != len(candidate):
        return [f"{label}: crop_records length changed from {len(baseline)} to {len(candidate)}"]

    for index, (base_crop, candidate_crop) in enumerate(zip(baseline, candidate), start=1):
        for field in ("id", "bbox", "has_image"):
            if base_crop.get(field) != candidate_crop.get(field):
                errors.append(
                    f"{label}: crop {index} {field} changed from {base_crop.get(field)} "
                    f"to {candidate_crop.get(field)}",
                )

        score_delta = abs(float(base_crop.get("score", 0.0)) - float(candidate_crop.get("score", 0.0)))
        if score_delta > score_tolerance:
            errors.append(
                f"{label}: crop {index} score changed by {score_delta:.6f}, "
                f"above {score_tolerance:.6f}",
            )

    return errors


def _compare_detection_records(
    label: str,
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    bbox_tolerance: float,
    confidence_tolerance: float,
) -> list[str]:
    errors: list[str] = []
    if len(baseline) != len(candidate):
        return [
            f"{label}: detection_records length changed from {len(baseline)} to {len(candidate)}",
        ]

    for index, (base_detection, candidate_detection) in enumerate(zip(baseline, candidate), start=1):
        for field in ("crop_id", "class"):
            if base_detection.get(field) != candidate_detection.get(field):
                errors.append(
                    f"{label}: detection {index} {field} changed from {base_detection.get(field)} "
                    f"to {candidate_detection.get(field)}",
                )

        confidence_delta = abs(
            float(base_detection.get("confidence", 0.0))
            - float(candidate_detection.get("confidence", 0.0)),
        )
        if confidence_delta > confidence_tolerance:
            errors.append(
                f"{label}: detection {index} confidence changed by {confidence_delta:.6f}, "
                f"above {confidence_tolerance:.6f}",
            )

        base_bbox = base_detection.get("global_bbox", [])
        candidate_bbox = candidate_detection.get("global_bbox", [])
        if len(base_bbox) != len(candidate_bbox):
            errors.append(
                f"{label}: detection {index} bbox length changed from {len(base_bbox)} "
                f"to {len(candidate_bbox)}",
            )
            continue

        bbox_delta = max(
            (
                abs(float(base_value) - float(candidate_value))
                for base_value, candidate_value in zip(base_bbox, candidate_bbox)
            ),
            default=0.0,
        )
        if bbox_delta > bbox_tolerance:
            errors.append(
                f"{label}: detection {index} bbox changed by {bbox_delta:.4f}, "
                f"above {bbox_tolerance:.4f}",
            )

    return errors


def _compare_record(
    key: tuple[str, int],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    errors: list[str] = []
    label = f"{key[0]} run {key[1]}"

    for field in ("pre_nms_crops", "post_nms_crops", "detection_count", "output_keys", "crop_image_count"):
        if baseline.get(field) != candidate.get(field):
            errors.append(f"{label}: {field} changed from {baseline.get(field)} to {candidate.get(field)}")

    if "crop_records" in baseline and "crop_records" in candidate:
        errors.extend(
            _compare_crop_records(
                label,
                baseline["crop_records"],
                candidate["crop_records"],
                args.crop_score_tolerance,
            ),
        )

    if "detection_records" in baseline and "detection_records" in candidate:
        errors.extend(
            _compare_detection_records(
                label,
                baseline["detection_records"],
                candidate["detection_records"],
                args.bbox_tolerance,
                args.confidence_tolerance,
            ),
        )

    if not args.skip_output_hash_check and "output_hashes" in baseline and "output_hashes" in candidate:
        if baseline["output_hashes"] != candidate["output_hashes"]:
            errors.append(f"{label}: encoded visual output hashes changed")

    if not args.allow_payload_growth and candidate["payload_bytes"] > baseline["payload_bytes"]:
        errors.append(
            f"{label}: payload grew from {baseline['payload_bytes']} to {candidate['payload_bytes']} bytes",
        )

    baseline_timings = baseline.get("timings_ms", {})
    candidate_timings = candidate.get("timings_ms", {})
    if args.stage in baseline_timings and args.stage in candidate_timings:
        stage_delta = _fractional_delta(
            float(baseline_timings[args.stage]),
            float(candidate_timings[args.stage]),
        )
        if stage_delta > -args.min_stage_improvement:
            errors.append(
                f"{label}: {args.stage} improved {-stage_delta:.1%}, below {args.min_stage_improvement:.1%}",
            )

    if "total_pipeline" in baseline_timings and "total_pipeline" in candidate_timings:
        total_delta = _fractional_delta(
            float(baseline_timings["total_pipeline"]),
            float(candidate_timings["total_pipeline"]),
        )
        if total_delta > args.max_total_regression:
            errors.append(
                f"{label}: total_pipeline regressed {total_delta:.1%}, above {args.max_total_regression:.1%}",
            )

    return errors


def main() -> int:
    args = _parse_args()
    baseline_records = _load_records(args.baseline)
    candidate_records = _load_records(args.candidate)

    missing = sorted(set(baseline_records) - set(candidate_records))
    extra = sorted(set(candidate_records) - set(baseline_records))
    errors = []
    if missing:
        errors.append(f"Candidate is missing records: {missing}")
    if extra:
        errors.append(f"Candidate has extra records: {extra}")

    for key in sorted(set(baseline_records) & set(candidate_records)):
        errors.extend(_compare_record(key, baseline_records[key], candidate_records[key], args))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("Benchmark comparison passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
