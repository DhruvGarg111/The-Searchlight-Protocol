from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from webapp.backend.models.pipeline import PipelineSettings
from webapp.backend.services.pipeline_service import SearchlightPipelineService


class FakeBox:
    def __init__(self, xyxy, cls_index: int, confidence: float) -> None:
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.cls = torch.tensor([cls_index], dtype=torch.float32)
        self.conf = torch.tensor([confidence], dtype=torch.float32)


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes
        self.names = {0: "target", 1: "decoy"}


class FakeDetector:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append({"source": source, "kwargs": kwargs})
        return [
            FakeResult([FakeBox([1, 2, 11, 12], 0, 0.91)]),
            FakeResult([FakeBox([3, 4, 13, 14], 1, 0.82)]),
        ]


class SequentialEquivalentDetector:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append({"source": source, "kwargs": kwargs})
        sources = source if isinstance(source, list) else [source]
        results = []
        for crop_image in sources:
            marker = int(np.asarray(crop_image)[0, 0, 0])
            results.append(
                FakeResult(
                    [FakeBox([marker + 1, 2, marker + 11, 12], marker % 2, 0.9 - marker * 0.1)],
                ),
            )
        return results


def _sequential_yolo_reference(crops, detector, settings):
    detections = []
    for crop in crops:
        bbox_x, bbox_y, _, _ = crop["bbox"]
        results = detector.predict(
            source=crop["image"],
            conf=settings.yolo_confidence,
            iou=settings.yolo_iou_threshold,
            augment=True,
            agnostic_nms=True,
            verbose=False,
        )
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_index = int(box.cls[0])
            detections.append(
                {
                    "crop_id": crop["id"],
                    "class": results[0].names[cls_index],
                    "class_index": cls_index,
                    "confidence": float(box.conf[0]),
                    "global_bbox": [x1 + bbox_x, y1 + bbox_y, x2 + bbox_x, y2 + bbox_y],
                },
            )
    return detections


def test_yolo_runs_crops_in_one_batch_and_remaps_boxes(make_app_config) -> None:
    service = SearchlightPipelineService(make_app_config())
    fake_detector = FakeDetector()
    service._detector = fake_detector

    crops = [
        {"id": 1, "bbox": (10, 20, 64, 64), "image": np.zeros((64, 64, 3), dtype=np.uint8)},
        {"id": 2, "bbox": (30, 40, 64, 64), "image": np.zeros((64, 64, 3), dtype=np.uint8)},
    ]

    detections, meta = service._run_yolo(crops, PipelineSettings())

    assert len(fake_detector.calls) == 1
    assert isinstance(fake_detector.calls[0]["source"], list)
    assert fake_detector.calls[0]["kwargs"]["augment"] is True
    assert fake_detector.calls[0]["kwargs"]["agnostic_nms"] is True
    assert meta["batched"] is True
    assert meta["raw_detections"] == 2
    assert detections[0]["global_bbox"] == [11.0, 22.0, 21.0, 32.0]
    assert detections[1]["global_bbox"] == [33.0, 44.0, 43.0, 54.0]


def test_batched_yolo_matches_sequential_reference(make_app_config) -> None:
    settings = PipelineSettings()
    crops = [
        {"id": 1, "bbox": (10, 20, 64, 64), "image": np.zeros((64, 64, 3), dtype=np.uint8)},
        {"id": 2, "bbox": (30, 40, 64, 64), "image": np.ones((64, 64, 3), dtype=np.uint8)},
    ]

    service = SearchlightPipelineService(make_app_config())
    batched_detector = SequentialEquivalentDetector()
    service._detector = batched_detector
    batched_detections, meta = service._run_yolo(crops, settings)

    sequential_detections = _sequential_yolo_reference(crops, SequentialEquivalentDetector(), settings)

    assert meta["batched"] is True
    assert len(batched_detector.calls) == 1
    assert batched_detections == sequential_detections


def test_global_detection_nms_is_class_aware() -> None:
    detections = [
        {
            "crop_id": 1,
            "class": "target",
            "class_index": 0,
            "confidence": 0.9,
            "global_bbox": [0, 0, 20, 20],
        },
        {
            "crop_id": 2,
            "class": "target",
            "class_index": 0,
            "confidence": 0.6,
            "global_bbox": [2, 2, 22, 22],
        },
        {
            "crop_id": 3,
            "class": "decoy",
            "class_index": 1,
            "confidence": 0.5,
            "global_bbox": [2, 2, 22, 22],
        },
    ]

    kept = SearchlightPipelineService._apply_global_detection_nms(detections, iou_threshold=0.5)

    assert len(kept) == 2
    assert {(det["class"], det["confidence"]) for det in kept} == {
        ("target", 0.9),
        ("decoy", 0.5),
    }


def test_response_profiles_control_output_and_crop_image_payloads(make_app_config) -> None:
    service = SearchlightPipelineService(
        make_app_config(response_display_max_dim=16, response_display_crop_limit=1),
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cams = [np.zeros((8, 8), dtype=np.float32) for _ in range(3)]
    crops = [
        {"id": 1, "score": 0.9, "bbox": (0, 0, 16, 16), "image": image},
        {"id": 2, "score": 0.8, "bbox": (8, 8, 16, 16), "image": image},
    ]

    full_outputs = service._build_output_payload(
        PipelineSettings(response_profile="full"),
        image,
        cams,
        np.zeros((8, 8), dtype=np.float32),
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32), dtype=np.float32),
        image,
        image,
    )
    display_outputs = service._build_output_payload(
        PipelineSettings(response_profile="display"),
        image,
        cams,
        np.zeros((8, 8), dtype=np.float32),
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32), dtype=np.float32),
        image,
        image,
    )
    metadata_outputs = service._build_output_payload(
        PipelineSettings(response_profile="metadata"),
        image,
        cams,
        np.zeros((8, 8), dtype=np.float32),
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32), dtype=np.float32),
        image,
        image,
    )

    assert set(full_outputs) == {
        "original_image",
        "layer2_cam",
        "layer3_cam",
        "layer4_cam",
        "weighted_fusion_cam",
        "slicer_mask",
        "heatmap_resized",
        "post_nms_boundaries",
        "final_detections",
    }
    assert set(display_outputs) == {
        "original_image",
        "weighted_fusion_cam",
        "post_nms_boundaries",
        "final_detections",
    }
    assert metadata_outputs == {}

    display_crops = service._build_crop_payload(PipelineSettings(response_profile="display"), crops)
    metadata_crops = service._build_crop_payload(PipelineSettings(response_profile="metadata"), crops)

    assert display_crops[0]["image"].startswith("data:image/png;base64,")
    assert display_crops[1]["image"] == ""
    assert all(crop["image"] == "" for crop in metadata_crops)
