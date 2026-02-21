from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


class YOLODetector:
    """Thin wrapper around Ultralytics YOLO for controlled runtime options."""

    def __init__(
        self,
        model_version: str = "v8",
        model_variant: str = "n",
        model_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
    ) -> None:
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
        with torch.no_grad():
            return self.model.predict(
                source=source,
                conf=self.conf if conf is None else conf,
                iou=self.iou if iou is None else iou,
                device=self.device,
                **kwargs,
            )
