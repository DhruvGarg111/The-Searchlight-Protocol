from __future__ import annotations

from typing import Any

from fastapi import Request

try:
    from ..core.config import AppConfig
except ImportError:
    from core.config import AppConfig


def get_app_config(request: Request) -> AppConfig:
    return request.app.state.config


def get_pipeline_service(request: Request) -> Any:
    return request.app.state.pipeline_service
