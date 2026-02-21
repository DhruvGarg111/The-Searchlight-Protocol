from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class IntelligentSlicer:
    """Contours high-activation heatmap regions and emits padded crops."""

    def __init__(self, padding_factor: float, info_threshold: float, min_crop_size: int) -> None:
        self.padding_factor = padding_factor
        self.info_threshold = info_threshold
        self.min_crop_size = min_crop_size

    def slice(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
    ) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
        height, width = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (width, height))

        mask = (heatmap_resized > self.info_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crops: list[dict[str, object]] = []
        for index, contour in enumerate(contours):
            x, y, bbox_w, bbox_h = cv2.boundingRect(contour)

            pad_x = int(bbox_w * self.padding_factor)
            pad_y = int(bbox_h * self.padding_factor)

            crop_w = bbox_w + 2 * pad_x
            crop_h = bbox_h + 2 * pad_y

            if crop_w < self.min_crop_size:
                pad_x += (self.min_crop_size - crop_w) // 2 + 1

            if crop_h < self.min_crop_size:
                pad_y += (self.min_crop_size - crop_h) // 2 + 1

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + bbox_w + pad_x)
            y2 = min(height, y + bbox_h + pad_y)

            final_w = x2 - x1
            final_h = y2 - y1

            if final_w < self.min_crop_size:
                shortfall = self.min_crop_size - final_w
                if x1 > 0:
                    x1 = max(0, x1 - shortfall)
                else:
                    x2 = min(width, x2 + shortfall)

            if final_h < self.min_crop_size:
                shortfall = self.min_crop_size - final_h
                if y1 > 0:
                    y1 = max(0, y1 - shortfall)
                else:
                    y2 = min(height, y2 + shortfall)

            crop_img = original_image[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            crops.append(
                {
                    "id": index,
                    "image": crop_img,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "score": float(np.mean(heatmap_resized[y : y + bbox_h, x : x + bbox_w])),
                },
            )

        LOGGER.debug("Generated %d crops (min size: %d px)", len(crops), self.min_crop_size)
        return crops, mask, heatmap_resized
