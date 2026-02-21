from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]



def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}



def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default



def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    api_title: str
    api_version: str
    allow_origins: tuple[str, ...]
    log_level: str
    max_upload_bytes: int
    max_image_dimension: int
    max_image_pixels: int
    request_timeout_seconds: float
    preload_models: bool
    clear_cuda_cache_per_request: bool
    serial_execution: bool
    yolo_model_version: str
    yolo_model_variant: str
    yolo_model_path: Path


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    origins_env = os.getenv("SEARCHLIGHT_ALLOW_ORIGINS")
    if origins_env:
        allow_origins = tuple(origin.strip() for origin in origins_env.split(",") if origin.strip())
    else:
        allow_origins = (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        )

    model_path_env = os.getenv("SEARCHLIGHT_YOLO_MODEL_PATH")
    if model_path_env:
        model_path = Path(model_path_env)
        if not model_path.is_absolute():
            model_path = (BACKEND_DIR / model_path).resolve()
    else:
        model_path = BACKEND_DIR / "yolov9m.pt"

    return AppConfig(
        api_title=os.getenv("SEARCHLIGHT_API_TITLE", "Searchlight Pipeline API"),
        api_version=os.getenv("SEARCHLIGHT_API_VERSION", "1.0.0"),
        allow_origins=allow_origins,
        log_level=os.getenv("SEARCHLIGHT_LOG_LEVEL", "INFO").upper(),
        max_upload_bytes=_env_int("SEARCHLIGHT_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
        max_image_dimension=_env_int("SEARCHLIGHT_MAX_IMAGE_DIMENSION", 12000),
        max_image_pixels=_env_int("SEARCHLIGHT_MAX_IMAGE_PIXELS", 90_000_000),
        request_timeout_seconds=_env_float("SEARCHLIGHT_REQUEST_TIMEOUT_SECONDS", 240.0),
        preload_models=_env_bool("SEARCHLIGHT_PRELOAD_MODELS", True),
        clear_cuda_cache_per_request=_env_bool("SEARCHLIGHT_CLEAR_CUDA_CACHE_PER_REQUEST", False),
        serial_execution=_env_bool("SEARCHLIGHT_SERIAL_EXECUTION", True),
        yolo_model_version=os.getenv("SEARCHLIGHT_YOLO_MODEL_VERSION", "v9"),
        yolo_model_variant=os.getenv("SEARCHLIGHT_YOLO_MODEL_VARIANT", "m"),
        yolo_model_path=model_path,
    )
