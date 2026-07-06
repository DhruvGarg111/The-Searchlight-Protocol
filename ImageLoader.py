from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

DEFAULT_CONTRAST_FACTOR = 1.8


class DroneImageLoader:
    """Loads, enhances, and normalizes high-resolution aerial images for CAM generation.

    This loader applies contrast enhancement to emphasize features in low-contrast aerial imagery
    and downsamples the image while preserving the aspect ratio to fit within a specified maximum
    dimension. Finally, it converts the image to a normalized PyTorch tensor ready for model inference.
    """

    def __init__(
        self,
        max_dim: int,
        contrast_factor: float = DEFAULT_CONTRAST_FACTOR,
        device: torch.device | str | None = None,
    ) -> None:
        """Initializes the DroneImageLoader.

        Args:
            max_dim: The maximum height or width dimension allowed for the model-input tensor.
                Images larger than this will be downsampled to fit this dimension.
            contrast_factor: Factor to multiply the image contrast by. Defaults to 1.8.
                Use 1.0 to disable contrast enhancement.
            device: The PyTorch device (CPU, CUDA, etc.) to place the final tensor on.
                If None, defaults to GPU if CUDA is available, else CPU.
        """
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
        """Loads and preprocesses an image from the given path.

        Args:
            image_path: Absolute or relative path to the target image file.

        Returns:
            A tuple containing:
                - original_np (np.ndarray): The contrast-enhanced, full-resolution image as a NumPy array (HWC, RGB).
                - tensor (torch.Tensor): Preprocessed, downscaled, and normalized image tensor of shape (1, 3, H, W).
                - original_size (tuple[int, int]): The (width, height) of the original image before any resizing.
                - scale (float): The scaling factor applied to the image (e.g., 0.5 means resized to 50% width/height).

        Raises:
            FileNotFoundError: If the image_path does not exist on disk.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as image:
            return self.load_image(image)

    def load_bytes(self, image_bytes: bytes) -> tuple[np.ndarray, torch.Tensor, tuple[int, int], float]:
        """Loads and preprocesses an image directly from encoded image bytes.

        Args:
            image_bytes: Encoded image bytes from an upload or in-memory fixture.

        Returns:
            The same tuple returned by :meth:`load`.
        """
        with Image.open(BytesIO(image_bytes)) as image:
            return self.load_image(image)

    def load_image(self, image: Image.Image) -> tuple[np.ndarray, torch.Tensor, tuple[int, int], float]:
        """Preprocesses an already opened PIL image.

        Args:
            image: PIL image object. The image is converted to RGB before processing.

        Returns:
            A tuple containing the contrast-enhanced original image array, model tensor,
            original size, and resize scale.
        """
        image = image.convert("RGB")
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
