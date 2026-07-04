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

CAM_FUSION_WEIGHTS = {"layer2": 0.7, "layer3": 0.9, "layer4": 1.0}
RESNET_INPUT_MAX_DIM = 1800


class SearchlightPipelineService:
    """Orchestrates end-to-end Searchlight inference with cached model instances.

    Fuses deep learning models (ResNet18 Layer-CAM for semantic guidance and YOLOv8 for detection)
    along with morphological region extraction and coordinate transformation routines to run
    optimized inference on high-resolution drone imagery.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initializes the SearchlightPipelineService.

        Args:
            config: Application configuration parameters.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._guide_model: torch.nn.Module | None = None
        self._detector: YOLODetector | None = None

        self._model_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def models_ready(self) -> bool:
        """Checks if both the Guide and Detector models are loaded and ready."""
        return self._guide_model is not None and self._detector is not None

    def warmup(self) -> None:
        """Eagerly loads Guide and Detector models into GPU/CPU memory during API startup."""
        start = time.perf_counter()
        self._ensure_guide_model()
        self._ensure_detector()
        LOGGER.info(
            "Pipeline models warmed up on %s in %.2f ms",
            self.device,
            (time.perf_counter() - start) * 1000.0,
        )

    def close(self) -> None:
        """Releases cached model instances to free up GPU and system memory."""
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
        """Runs the complete detection pipeline from raw image bytes.

        Args:
            image_bytes: The raw image file bytes.
            suffix: File extension/suffix (e.g., '.jpg', '.png') to use for the temp file.
            settings: Pipeline settings.

        Returns:
            dict[str, Any]: The complete pipeline execution results dictionary.
        """
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
        """Runs the complete 3-stage detection pipeline on a file path.

        Optionally locks execution to serialize requests and optimize resource utilization.

        Args:
            image_path: System file path to the target high-resolution image.
            settings: Hyperparameter settings containing padding, thresholds, confidence levels.

        Returns:
            dict[str, Any]: Output dictionary containing metadata, metrics, base64-encoded visual maps,
                and remapped global detections.
        """
        lock_ctx = self._run_lock if self.config.serial_execution else nullcontext()

        with lock_ctx:
            run_id = str(uuid.uuid4())
            run_started_utc = datetime.now(timezone.utc).isoformat()
            stage_timings_ms: dict[str, float] = {}
            pipeline_start = time.perf_counter()

            self._maybe_clear_cuda_cache()

            t0 = time.perf_counter()
            loader = DroneImageLoader(max_dim=RESNET_INPUT_MAX_DIM, device=self.device)
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
                        "guide_backbone": "ResNet18 (ImageNet1K_V1)",
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
        """Loads and returns the ResNet18 guide model, caching it for subsequent calls.

        Ensures thread-safe initialization using an internal lock.
        """
        with self._model_lock:
            if self._guide_model is None:
                LOGGER.info("Loading ResNet18 guide model on %s", self.device)
                try:
                    weights = models.ResNet18_Weights.IMAGENET1K_V1
                except AttributeError:
                    weights = "IMAGENET1K_V1"

                self._guide_model = models.resnet18(weights=weights).to(self.device)
                self._guide_model.eval()
        return self._guide_model

    def _ensure_detector(self) -> YOLODetector:
        """Loads and returns the YOLODetector model, caching it for subsequent calls.

        Resolves path overrides and model options dynamically.
        """
        with self._model_lock:
            if self._detector is None:
                model_path = self.config.yolo_model_path
                if model_path.exists():
                    detector_model_path = str(model_path)
                    LOGGER.info("Loading YOLO detector from %s", model_path)
                else:
                    detector_model_path = (
                        f"yolo{self.config.yolo_model_version}{self.config.yolo_model_variant}.pt"
                    )
                    LOGGER.warning(
                        "YOLO model file not found at %s; falling back to %s",
                        model_path,
                        detector_model_path,
                    )

                self._detector = YOLODetector(
                    model_version=self.config.yolo_model_version,
                    model_variant=self.config.yolo_model_variant,
                    model_path=detector_model_path,
                    device=str(self.device),
                )
        return self._detector

    def _run_yolo(
        self,
        crops: list[dict[str, Any]],
        settings: PipelineSettings,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Sequentially runs YOLO detection on each crop and translates bounding boxes back to global coordinates.

        Args:
            crops: List of crop dictionaries containing 'image', 'bbox', and 'id'.
            settings: Pipeline settings.

        Returns:
            A tuple of (detections, detection_meta).
        """
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
        """Clears CUDA memory cache if configured to do so."""
        if self.config.clear_cuda_cache_per_request and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _apply_nms(crops: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
        """Applies Non-Maximum Suppression to overlapping image crops to filter duplicate candidate regions.

        Args:
            crops: List of extracted candidates.
            iou_threshold: Overlap ratio threshold for NMS.

        Returns:
            List of filtered crops.
        """
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
        """Renders bounding boxes and labels for selected crops on the original image."""
        canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        line_thickness, font_scale, text_thickness = SearchlightPipelineService._annotation_style(
            image_rgb,
        )

        for crop in crops:
            x, y, width, height = [int(v) for v in crop["bbox"]]
            crop_id = int(crop["id"])

            box_color = (0, 0, 255)
            cv2.rectangle(
                canvas,
                (x, y),
                (x + width, y + height),
                box_color,
                line_thickness,
            )
            SearchlightPipelineService._draw_box_label(
                canvas=canvas,
                text=f"crop {crop_id}",
                anchor=(x, y),
                color=box_color,
                font_scale=font_scale,
                text_thickness=text_thickness,
                line_thickness=line_thickness,
            )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _draw_final_detections(image_rgb: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        """Renders final remapped detection bounding boxes and class labels."""
        canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        line_thickness, font_scale, text_thickness = SearchlightPipelineService._annotation_style(
            image_rgb,
        )

        for detection in detections:
            gx1, gy1, gx2, gy2 = detection["global_bbox"]
            x1, y1, x2, y2 = int(gx1), int(gy1), int(gx2), int(gy2)
            label = f"{detection['class']} {detection['confidence']:.2f}"

            box_color = (0, 0, 255)
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                box_color,
                line_thickness,
            )
            SearchlightPipelineService._draw_box_label(
                canvas=canvas,
                text=label,
                anchor=(x1, y1),
                color=box_color,
                font_scale=font_scale,
                text_thickness=text_thickness,
                line_thickness=line_thickness,
            )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _annotation_style(image_rgb: np.ndarray) -> tuple[int, float, int]:
        """Calculates dynamic line and font scale settings based on the image dimensions."""
        height, width = image_rgb.shape[:2]
        max_dim = max(height, width)

        line_thickness = int(np.clip(round(max_dim / 700.0), 1, 20))
        font_scale = float(np.clip(max_dim / 1800.0, 0.55, 6.0))
        text_thickness = int(np.clip(round(line_thickness * 0.7), 1, 12))
        return line_thickness, font_scale, text_thickness

    @staticmethod
    def _draw_box_label(
        canvas: np.ndarray,
        text: str,
        anchor: tuple[int, int],
        color: tuple[int, int, int],
        font_scale: float,
        text_thickness: int,
        line_thickness: int,
    ) -> None:
        """Helper to draw a styled label box containing text next to a detection box."""
        x, y = anchor
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            text_thickness,
        )
        padding = max(2, line_thickness)
        margin = max(2, line_thickness)

        label_x1 = max(0, x)
        label_y2 = y - margin
        label_y1 = label_y2 - text_height - baseline - (2 * padding)

        if label_y1 < 0:
            label_y1 = min(canvas.shape[0] - 1, y + margin)
            label_y2 = min(
                canvas.shape[0] - 1,
                label_y1 + text_height + baseline + (2 * padding),
            )

        label_x2 = min(canvas.shape[1] - 1, label_x1 + text_width + (2 * padding))
        if label_x2 <= label_x1 or label_y2 <= label_y1:
            return

        cv2.rectangle(
            canvas,
            (label_x1, int(label_y1)),
            (int(label_x2), int(label_y2)),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            canvas,
            (label_x1, int(label_y1)),
            (int(label_x2), int(label_y2)),
            color,
            max(1, line_thickness // 2),
        )

        text_x = label_x1 + padding
        text_y = int(label_y2 - baseline - padding)
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
        """Converts a normalized single-channel heatmap to a 3-channel RGB Jet-colormap image."""
        normalized = heatmap.astype(np.float32)
        normalized -= normalized.min()
        normalized /= normalized.max() + 1e-9

        heatmap_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_RGB2RGB)

    @staticmethod
    def _as_data_url(image: np.ndarray) -> str:
        """Encodes an RGB image array to a base64 Data URL (PNG)."""
        image_u8 = SearchlightPipelineService._normalize_image_uint8(image)
        image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)

        encoded_ok, encoded = cv2.imencode(".png", image_bgr)
        if not encoded_ok:
            raise RuntimeError("Failed to encode output image.")

        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/png;base64,{image_b64}"

    @staticmethod
    def _normalize_image_uint8(image: np.ndarray) -> np.ndarray:
        """Converts any input image shape/format into a standard RGB uint8 NumPy image array."""
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
