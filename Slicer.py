from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class IntelligentSlicer:
    """Slices a large high-resolution image into smaller regions of interest based on a semantic heatmap.

    Avoids brute-force grid-slicing by identifying contiguous regions of high activation in a heatmap,
    finding their bounding contours, applying padding, enforcing size constraints, and cropping them
    for targeted downstream detection.
    """

    def __init__(self, padding_factor: float, info_threshold: float, min_crop_size: int) -> None:
        """Initializes the IntelligentSlicer.

        Args:
            padding_factor: The ratio of padding to apply around detected contours
                (e.g., 0.4 applies 40% of bounding box width/height as padding on each side).
            info_threshold: The cutoff threshold (between 0.0 and 1.0) above which
                heatmap activations are considered regions of interest.
            min_crop_size: The minimum pixel dimension (height or width) for any crop.
                Crops smaller than this will be expanded symmetrically or clamped.
        """
        self.padding_factor = padding_factor
        self.info_threshold = info_threshold
        self.min_crop_size = min_crop_size

    def slice(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
    ) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
        """Extracts regions of interest from the original image based on high activation regions.

        Args:
            original_image: The full-resolution original image as a NumPy array (HWC, RGB).
            heatmap: The normalized 2D semantic guidance heatmap (0.0 to 1.0) of shape (H_cam, W_cam).

        Returns:
            A tuple containing:
                - crops (list[dict[str, object]]): List of dictionary representations of crops.
                    Each dict contains:
                        - "id" (int): Index of the crop.
                        - "image" (np.ndarray): Padded and cropped sub-image.
                        - "bbox" (tuple[int, int, int, int]): Crop boundaries in original image coordinates
                          as (x, y, width, height).
                        - "score" (float): Mean heatmap activation value within the bounding box.
                - mask (np.ndarray): The binary threshold mask used to identify regions of interest.
                - heatmap_resized (np.ndarray): The original heatmap resized to match original_image dimensions.
        """
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
