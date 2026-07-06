from __future__ import annotations

import asyncio
from io import BytesIO

from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, UploadFile

from webapp.backend.core.dependencies import get_app_config, get_pipeline_service
from webapp.backend.routers.pipeline import router as pipeline_router
from webapp.backend.routers.pipeline import run_pipeline


def _image_upload() -> UploadFile:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), color=(8, 16, 24)).save(buffer, format="PNG")
    buffer.seek(0)
    return UploadFile(
        file=buffer,
        filename="fixture.png",
        headers=Headers({"content-type": "image/png"}),
    )


class StubPipelineService:
    def __init__(self) -> None:
        self.calls = []

    def run_from_bytes(self, image_bytes, suffix, settings):
        self.calls.append((image_bytes, suffix, settings))
        return {
            "meta": {
                "device": "cpu",
                "original_size": [32, 24],
                "scale_factor": 1.0,
                "torch_version": "test",
                "cuda_available": False,
            },
            "settings": settings.model_dump(),
            "counts": {
                "pre_nms_crops": 1,
                "post_nms_crops": 1,
                "detections": 1,
            },
            "research": {
                "run_id": "abc123",
                "started_at_utc": "2026-07-05T00:00:00+00:00",
                "experiment": "test",
                "objective": "contract test",
                "model_stack": {
                    "guide_backbone": "ResNet18 (ImageNet1K_V1)",
                    "guide_target_layers": ["layer2[-1]", "layer3[-1]", "layer4[-1]"],
                    "cam_fusion_weights": {"layer2": 0.7, "layer3": 0.9, "layer4": 1.0},
                    "detector": "YOLOv8-n",
                },
                "detector_inference": {
                    "conf": settings.yolo_confidence,
                    "iou": settings.yolo_iou_threshold,
                    "augment": True,
                    "agnostic_nms": True,
                },
                "crop_selection": {
                    "padding_factor": settings.padding_factor,
                    "heatmap_threshold": settings.heatmap_threshold,
                    "min_crop_size": settings.min_crop_size,
                    "nms_iou_threshold": settings.nms_iou_threshold,
                    "pre_nms_count": 1,
                    "post_nms_count": 1,
                },
                "timings_ms": {"total_pipeline": 1.0},
                "detection_meta": {"batched": True},
            },
            "outputs": {
                "original_image": "data:image/png;base64,",
                "layer2_cam": "data:image/png;base64,",
                "layer3_cam": "data:image/png;base64,",
                "layer4_cam": "data:image/png;base64,",
                "weighted_fusion_cam": "data:image/png;base64,",
                "slicer_mask": "data:image/png;base64,",
                "heatmap_resized": "data:image/png;base64,",
                "post_nms_boundaries": "data:image/png;base64,",
                "final_detections": "data:image/png;base64,",
            },
            "crops": [
                {
                    "id": 1,
                    "score": 0.75,
                    "bbox": [0, 0, 16, 16],
                    "image": "data:image/png;base64,",
                },
            ],
            "detections": [
                {
                    "crop_id": 1,
                    "class": "target",
                    "confidence": 0.91,
                    "global_bbox": [1.0, 2.0, 12.0, 14.0],
                },
            ],
        }


def test_run_pipeline_contract_with_stubbed_service(make_app_config) -> None:
    service = StubPipelineService()

    response = asyncio.run(
        run_pipeline(
            image=_image_upload(),
            padding_factor=0.4,
            heatmap_threshold=0.4,
            yolo_confidence=0.3,
            min_crop_size=120,
            nms_iou_threshold=0.2,
            yolo_iou_threshold=0.6,
            response_profile="display",
            enable_global_nms=True,
            global_nms_iou_threshold=0.45,
            pipeline_service=service,
            config=make_app_config(),
        ),
    )

    payload = response.model_dump(by_alias=True)

    assert set(payload) == {
        "meta",
        "settings",
        "counts",
        "research",
        "outputs",
        "crops",
        "detections",
        "input_filename",
    }
    assert payload["input_filename"] == "fixture.png"
    assert payload["settings"]["response_profile"] == "display"
    assert payload["settings"]["enable_global_nms"] is True
    assert payload["settings"]["global_nms_iou_threshold"] == 0.45
    assert payload["detections"][0]["class"] == "target"
    assert service.calls[0][1] == ".png"


def test_run_pipeline_defaults_to_full_response_contract(make_app_config) -> None:
    service = StubPipelineService()
    app = FastAPI()
    app.include_router(pipeline_router)
    app.dependency_overrides[get_pipeline_service] = lambda: service
    app.dependency_overrides[get_app_config] = lambda: make_app_config(enable_global_nms_default=True)

    buffer = BytesIO()
    Image.new("RGB", (32, 24), color=(8, 16, 24)).save(buffer, format="PNG")
    response = TestClient(app).post(
        "/run-pipeline",
        files={"image": ("fixture.png", buffer.getvalue(), "image/png")},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["settings"]["response_profile"] == "full"
    assert payload["settings"]["enable_global_nms"] is True
    assert set(payload["outputs"]) == {
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
    assert payload["crops"][0]["image"].startswith("data:image/png;base64,")
