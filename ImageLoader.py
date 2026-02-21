from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

DEFAULT_CONTRAST_FACTOR = 1.8


class DroneImageLoader:
    """Loads and normalizes aerial images for CAM generation."""

    def __init__(
        self,
        max_dim: int,
        contrast_factor: float = DEFAULT_CONTRAST_FACTOR,
        device: torch.device | str | None = None,
    ) -> None:
        self.max_dim = max_dim
        self.contrast_factor = contrast_factor
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )

    def load(self, image_path: str | Path) -> tuple[np.ndarray, torch.Tensor, tuple[int, int], float]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        if self.contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.contrast_factor)
            LOGGER.debug("Applied contrast enhancement: x%.2f", self.contrast_factor)

        original_np = np.array(image)

        width, height = original_size
        scale = min(self.max_dim / width, self.max_dim / height, 1.0)

        if scale < 1.0:
            resized_size = (int(width * scale), int(height * scale))
            image_resized = image.resize(resized_size, Image.LANCZOS)
            LOGGER.debug(
                "Downsampled input for CAM from %s to %s (scale=%.3f)",
                original_size,
                resized_size,
                scale,
            )
        else:
            image_resized = image
            scale = 1.0
            LOGGER.debug("Loaded input at original size %s", original_size)

        tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
        return original_np, tensor, original_size, scale
