from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.backend.core.config import AppConfig


@pytest.fixture
def make_app_config(tmp_path):
    def _factory(**overrides):
        values = {
            "api_title": "Searchlight Pipeline API",
            "api_version": "1.0.0",
            "allow_origins": ("http://localhost:5173",),
            "log_level": "INFO",
            "max_upload_bytes": 25 * 1024 * 1024,
            "max_image_dimension": 12000,
            "max_image_pixels": 90_000_000,
            "request_timeout_seconds": 240.0,
            "resnet_input_max_dim": 1800,
            "preload_models": False,
            "clear_cuda_cache_per_request": False,
            "serial_execution": True,
            "enable_global_nms_default": False,
            "response_display_max_dim": 1600,
            "response_display_crop_limit": 6,
            "response_display_format": "png",
            "yolo_model_version": "v8",
            "yolo_model_variant": "n",
            "yolo_model_path": tmp_path / "yolov8n.pt",
        }
        values.update(overrides)
        return AppConfig(**values)

    return _factory
