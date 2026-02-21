from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    from .pipeline import PipelineSettings
except ImportError:
    from models.pipeline import PipelineSettings


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
    errors: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str
    device: str
    models_ready: bool


class MetaPayload(BaseModel):
    device: str
    original_size: list[int]
    scale_factor: float
    torch_version: str
    cuda_available: bool


class CountsPayload(BaseModel):
    pre_nms_crops: int
    post_nms_crops: int
    detections: int


class ModelStackPayload(BaseModel):
    guide_backbone: str
    guide_target_layers: list[str]
    cam_fusion_weights: dict[str, float]
    detector: str


class DetectorInferencePayload(BaseModel):
    conf: float
    iou: float
    augment: bool
    agnostic_nms: bool


class CropSelectionPayload(BaseModel):
    padding_factor: float
    heatmap_threshold: float
    min_crop_size: int
    nms_iou_threshold: float
    pre_nms_count: int
    post_nms_count: int


class ResearchPayload(BaseModel):
    run_id: str
    started_at_utc: str
    experiment: str
    objective: str
    model_stack: ModelStackPayload
    detector_inference: DetectorInferencePayload
    crop_selection: CropSelectionPayload
    timings_ms: dict[str, float]
    detection_meta: dict[str, Any]


class CropPayload(BaseModel):
    id: int
    score: float
    bbox: list[int]
    image: str


class DetectionPayload(BaseModel):
    crop_id: int
    class_name: str = Field(alias="class")
    confidence: float
    global_bbox: list[float]

    model_config = ConfigDict(populate_by_name=True)


class RunPipelineResponse(BaseModel):
    meta: MetaPayload
    settings: PipelineSettings
    counts: CountsPayload
    research: ResearchPayload
    outputs: dict[str, str]
    crops: list[CropPayload]
    detections: list[DetectionPayload]
    input_filename: str
