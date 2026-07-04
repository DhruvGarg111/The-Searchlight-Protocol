from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


class YOLODetector:
    """Thin wrapper around the Ultralytics YOLO model for standardizing runtime operations.

    Handles model loading, automatic device placement, configuration-driven thresholding,
    and runs predictions with gradient computation disabled.
    """

    def __init__(
        self,
        model_version: str = "v8",
        model_variant: str = "n",
        model_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
    ) -> None:
        """Initializes the YOLODetector.

        Args:
            model_version: YOLO model version string (e.g., "v8", "v9"). Defaults to "v8".
            model_variant: YOLO variant string (e.g., "n", "s", "m", "l", "x"). Defaults to "n".
            model_path: Optional override for the weights file path. If None, resolves to a default filename
                like "yolov8n.pt" in the current directory or downloads it.
            conf: Default confidence threshold for filtering detections. Defaults to 0.25.
            iou: Default Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS). Defaults to 0.45.
            device: Target execution device (e.g., "cuda", "cpu"). If None, auto-selects CUDA if available, else CPU.
        """
        self.model_path = model_path or f"yolo{model_version}{model_variant}.pt"
        self.conf = conf
        self.iou = iou
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        resolved_path = Path(self.model_path)
        load_target = str(resolved_path) if resolved_path.exists() else self.model_path

        self.model = YOLO(load_target)
        try:
            self.model.to(self.device)
        except Exception:
            LOGGER.debug("Could not eagerly move YOLO model to %s", self.device)

    def predict(
        self,
        source: Any,
        conf: float | None = None,
        iou: float | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Runs YOLO object detection inference on the specified image source.

        Args:
            source: Input image source. Can be a file path, NumPy array, PIL Image, or list of sources.
            conf: Confidence threshold override. If None, uses the default configured threshold.
            iou: IoU threshold override for NMS. If None, uses the default configured threshold.
            **kwargs: Additional runtime arguments to forward to the underlying YOLO model predict method.

        Returns:
            list[Any]: A list of ultralytics.engine.results.Results objects containing detection details.
        """
        with torch.no_grad():
            return self.model.predict(
                source=source,
                conf=self.conf if conf is None else conf,
                iou=self.iou if iou is None else iou,
                device=self.device,
                **kwargs,
            )
