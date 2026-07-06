from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from webapp.backend.utils.validation import validate_image_payload, validate_upload_size


def _png_bytes(width: int = 80, height: int = 40) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(12, 34, 56)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_validation_rejects_invalid_image_content() -> None:
    with pytest.raises(ValueError, match="could not be decoded"):
        validate_image_payload(b"not an image", max_dimension=12000, max_pixels=90_000_000)


def test_validation_rejects_oversized_upload() -> None:
    with pytest.raises(ValueError, match="exceeds max size"):
        validate_upload_size(b"x" * 11, max_upload_bytes=10)


def test_loader_byte_path_matches_file_path(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ImageLoader import DroneImageLoader

    image_bytes = _png_bytes()
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(image_bytes)

    loader = DroneImageLoader(max_dim=32, contrast_factor=1.0, device="cpu")
    path_original, path_tensor, path_size, path_scale = loader.load(image_path)
    byte_original, byte_tensor, byte_size, byte_scale = loader.load_bytes(image_bytes)

    np.testing.assert_array_equal(byte_original, path_original)
    assert byte_size == path_size
    assert byte_scale == path_scale
    assert torch.allclose(byte_tensor.cpu(), path_tensor.cpu())
