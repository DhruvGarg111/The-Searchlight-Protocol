from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}



def safe_suffix(filename: str | None) -> str:
    if not filename:
        return ".jpg"

    suffix = Path(filename).suffix.lower()
    return suffix if suffix in ALLOWED_SUFFIXES else ".jpg"



def validate_content_type(content_type: str | None) -> None:
    if content_type and not content_type.startswith("image/"):
        raise ValueError("Uploaded file must be an image.")



def validate_upload_size(image_bytes: bytes, max_upload_bytes: int) -> None:
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    if len(image_bytes) > max_upload_bytes:
        raise ValueError(
            f"Uploaded image exceeds max size of {max_upload_bytes // (1024 * 1024)} MB.",
        )



def validate_image_payload(
    image_bytes: bytes,
    max_dimension: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Return (width, height) after validating uploaded bytes are a sane image."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file could not be decoded as an image.") from exc

    if width <= 0 or height <= 0:
        raise ValueError("Uploaded image has invalid dimensions.")

    if width > max_dimension or height > max_dimension:
        raise ValueError(
            f"Uploaded image dimensions exceed max allowed size of {max_dimension}px.",
        )

    if width * height > max_pixels:
        raise ValueError("Uploaded image exceeds max allowed pixel count.")

    return width, height
