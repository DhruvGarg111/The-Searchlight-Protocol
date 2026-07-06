from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PipelineSettings(BaseModel):
    """Runtime knobs for a single Searchlight pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    padding_factor: float = Field(default=0.4, ge=0.0, le=1.0)
    heatmap_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    yolo_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    min_crop_size: int = Field(default=120, ge=32, le=4096)
    nms_iou_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    yolo_iou_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    response_profile: Literal["full", "display", "metadata"] = "full"
    enable_global_nms: bool = False
    global_nms_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
