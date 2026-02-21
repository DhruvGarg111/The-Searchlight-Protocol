from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
import torchvision.models as models

try:
    from ..core.config import AppConfig
    from ..models.pipeline import PipelineSettings
except ImportError:
    from core.config import AppConfig
    from models.pipeline import PipelineSettings

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Detector import YOLODetector
from ImageLoader import DroneImageLoader
from LayerCam import MultiLayerCAM
from Slicer import IntelligentSlicer

CAM_FUSION_WEIGHTS = {"layer2": 0.4, "layer3": 1.0, "layer4": 1.0}


class SearchlightPipelineService:
    """Orchestrates end-to-end Searchlight inference with cached model instances."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._guide_model: torch.nn.Module | None = None
        self._detector: YOLODetector | None = None

        self._model_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def models_ready(self) -> bool:
        return self._guide_model is not None and self._detector is not None

    def warmup(self) -> None:
        """Eager-load heavy models during API startup."""
        start = time.perf_counter()
        self._ensure_guide_model()
        self._ensure_detector()
        LOGGER.info(
            "Pipeline models warmed up on %s in %.2f ms",
            self.device,
            (time.perf_counter() - start) * 1000.0,
        )

    def close(self) -> None:
        self._guide_model = None
        self._detector = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_from_bytes(
        self,
        image_bytes: bytes,
        suffix: str,
        settings: PipelineSettings,
    ) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(image_bytes)
            temp_path = tmp_file.name

        try:
            return self.run_from_path(temp_path, settings)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                LOGGER.warning("Failed to remove temp file %s", temp_path)

    def run_from_path(
        self,
        image_path: str,
        settings: PipelineSettings,
    ) -> dict[str, Any]:
        lock_ctx = self._run_lock if self.config.serial_execution else nullcontext()

        with lock_ctx:
            run_id = str(uuid.uuid4())
            run_started_utc = datetime.now(timezone.utc).isoformat()
            stage_timings_ms: dict[str, float] = {}
            pipeline_start = time.perf_counter()

            self._maybe_clear_cuda_cache()

            t0 = time.perf_counter()
            loader = DroneImageLoader(max_dim=settings.max_layercam_dim, device=self.device)
            original_np, input_tensor, original_size, scale_factor = loader.load(image_path)
            stage_timings_ms["image_load"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            guide_model = self._ensure_guide_model()
            target_layers = [guide_model.layer2[-1], guide_model.layer3[-1], guide_model.layer4[-1]]

            cam_engine = MultiLayerCAM(guide_model, target_layers)
            try:
                _ = cam_engine.generate_combined_cam(input_tensor)
                individual_cams = cam_engine.individual_cams.copy()
            finally:
                cam_engine.remove()

            stage_timings_ms["layercam_generation"] = (time.perf_counter() - t0) * 1000.0

            layer2_w, layer3_w, layer4_w = (
                CAM_FUSION_WEIGHTS["layer2"],
                CAM_FUSION_WEIGHTS["layer3"],
                CAM_FUSION_WEIGHTS["layer4"],
            )
            heatmap = (
                (individual_cams[0] * layer2_w)
                + (individual_cams[1] * layer3_w)
                + (individual_cams[2] * layer4_w)
            ) / (layer2_w + layer3_w + layer4_w)

            t0 = time.perf_counter()
            slicer = IntelligentSlicer(
                padding_factor=settings.padding_factor,
                info_threshold=settings.heatmap_threshold,
                min_crop_size=settings.min_crop_size,
            )
            crops, mask, heatmap_resized = slicer.slice(original_np, heatmap)
            pre_nms_count = len(crops)
            stage_timings_ms["intelligent_slicing"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            crops = self._apply_nms(crops, settings.nms_iou_threshold)
            for index, crop in enumerate(crops, start=1):
                crop["id"] = index
            stage_timings_ms["crop_nms"] = (time.perf_counter() - t0) * 1000.0

            post_nms_overlay = self._draw_crop_boxes(original_np, crops)

            t0 = time.perf_counter()
            detections, detection_meta = self._run_yolo(crops, settings)
            stage_timings_ms["yolo_detection"] = (time.perf_counter() - t0) * 1000.0

            final_overlay = self._draw_final_detections(original_np, detections)
            stage_timings_ms["total_pipeline"] = (time.perf_counter() - pipeline_start) * 1000.0

            response: dict[str, Any] = {
                "meta": {
                    "device": str(self.device),
                    "original_size": [int(original_size[0]), int(original_size[1])],
                    "scale_factor": float(scale_factor),
                    "torch_version": str(torch.__version__),
                    "cuda_available": bool(torch.cuda.is_available()),
                },
                "settings": settings.model_dump(),
                "counts": {
                    "pre_nms_crops": pre_nms_count,
                    "post_nms_crops": len(crops),
                    "detections": len(detections),
                },
                "research": {
                    "run_id": run_id,
                    "started_at_utc": run_started_utc,
                    "experiment": "searchlight-protocol-research",
                    "objective": "LayerCAM-guided high-resolution small object detection in aerial imagery.",
                    "model_stack": {
                        "guide_backbone": "ResNet50 (ImageNet1K_V1)",
                        "guide_target_layers": ["layer2[-1]", "layer3[-1]", "layer4[-1]"],
                        "cam_fusion_weights": CAM_FUSION_WEIGHTS,
                        "detector": f"YOLO{self.config.yolo_model_version}-{self.config.yolo_model_variant}",
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
                        "pre_nms_count": pre_nms_count,
                        "post_nms_count": len(crops),
                    },
                    "timings_ms": {stage: round(value, 2) for stage, value in stage_timings_ms.items()},
                    "detection_meta": detection_meta,
                },
                "outputs": {
                    "original_image": self._as_data_url(original_np),
                    "layer2_cam": self._as_data_url(self._colorize_heatmap(individual_cams[0])),
                    "layer3_cam": self._as_data_url(self._colorize_heatmap(individual_cams[1])),
                    "layer4_cam": self._as_data_url(self._colorize_heatmap(individual_cams[2])),
                    "weighted_fusion_cam": self._as_data_url(self._colorize_heatmap(heatmap)),
                    "slicer_mask": self._as_data_url(mask),
                    "heatmap_resized": self._as_data_url(self._colorize_heatmap(heatmap_resized)),
                    "post_nms_boundaries": self._as_data_url(post_nms_overlay),
                    "final_detections": self._as_data_url(final_overlay),
                },
                "crops": [
                    {
                        "id": int(crop["id"]),
                        "score": float(crop["score"]),
                        "bbox": [int(v) for v in crop["bbox"]],
                        "image": self._as_data_url(crop["image"]),
                    }
                    for crop in crops
                ],
                "detections": [
                    {
                        "crop_id": int(det["crop_id"]),
                        "class": det["class"],
                        "confidence": float(det["confidence"]),
                        "global_bbox": [float(v) for v in det["global_bbox"]],
                    }
                    for det in detections
                ],
            }

            self._maybe_clear_cuda_cache()
            return response

    def _ensure_guide_model(self) -> torch.nn.Module:
        with self._model_lock:
            if self._guide_model is None:
                LOGGER.info("Loading ResNet50 guide model on %s", self.device)
                try:
                    weights = models.ResNet50_Weights.IMAGENET1K_V1
                except AttributeError:
                    weights = "IMAGENET1K_V1"

                self._guide_model = models.resnet50(weights=weights).to(self.device)
                self._guide_model.eval()
        return self._guide_model

    def _ensure_detector(self) -> YOLODetector:
        with self._model_lock:
            if self._detector is None:
                model_path = self.config.yolo_model_path
                if not model_path.exists():
                    raise RuntimeError(f"YOLO model file not found: {model_path}")

                LOGGER.info("Loading YOLO detector from %s", model_path)
                self._detector = YOLODetector(
                    model_version=self.config.yolo_model_version,
                    model_variant=self.config.yolo_model_variant,
                    model_path=str(model_path),
                    device=str(self.device),
                )
        return self._detector

    def _run_yolo(
        self,
        crops: list[dict[str, Any]],
        settings: PipelineSettings,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not crops:
            return [], {"processed_crops": 0, "raw_detections": 0}

        detector = self._ensure_detector()
        detections: list[dict[str, Any]] = []
        raw_detections = 0

        for crop in crops:
            crop_image = crop["image"]
            bbox_x, bbox_y, _, _ = crop["bbox"]
            crop_id = crop["id"]

            results = detector.predict(
                source=crop_image,
                conf=settings.yolo_confidence,
                iou=settings.yolo_iou_threshold,
                augment=True,
                agnostic_nms=True,
                verbose=False,
            )

            if not results:
                continue

            raw_detections += len(results[0].boxes)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_index = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = results[0].names[cls_index]

                detections.append(
                    {
                        "crop_id": crop_id,
                        "class": class_name,
                        "confidence": confidence,
                        "global_bbox": [x1 + bbox_x, y1 + bbox_y, x2 + bbox_x, y2 + bbox_y],
                    },
                )

        return detections, {"processed_crops": len(crops), "raw_detections": raw_detections}

    def _maybe_clear_cuda_cache(self) -> None:
        if self.config.clear_cuda_cache_per_request and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _apply_nms(crops: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
        if not crops:
            return []

        boxes = []
        scores = []

        for crop in crops:
            x, y, width, height = crop["bbox"]
            boxes.append([float(x), float(y), float(x + width), float(y + height)])
            scores.append(float(crop["score"]))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)

        return [crops[i] for i in keep_indices.tolist()]

    @staticmethod
    def _draw_crop_boxes(image_rgb: np.ndarray, crops: list[dict[str, Any]]) -> np.ndarray:
        canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        for crop in crops:
            x, y, width, height = [int(v) for v in crop["bbox"]]
            crop_id = int(crop["id"])

            cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(
                canvas,
                f"crop {crop_id}",
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _draw_final_detections(image_rgb: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        for detection in detections:
            gx1, gy1, gx2, gy2 = detection["global_bbox"]
            x1, y1, x2, y2 = int(gx1), int(gy1), int(gx2), int(gy2)
            label = f"{detection['class']} {detection['confidence']:.2f}"

            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                canvas,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
        normalized = heatmap.astype(np.float32)
        normalized -= normalized.min()
        normalized /= normalized.max() + 1e-9

        heatmap_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _as_data_url(image: np.ndarray) -> str:
        image_u8 = SearchlightPipelineService._normalize_image_uint8(image)
        image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)

        encoded_ok, encoded = cv2.imencode(".png", image_bgr)
        if not encoded_ok:
            raise RuntimeError("Failed to encode output image.")

        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/png;base64,{image_b64}"

    @staticmethod
    def _normalize_image_uint8(image: np.ndarray) -> np.ndarray:
        image_np = np.asarray(image)

        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.float32)
            if image_np.max() <= 1.0:
                image_np *= 255.0
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.ndim == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError("Expected an RGB image array.")

        return image_np
