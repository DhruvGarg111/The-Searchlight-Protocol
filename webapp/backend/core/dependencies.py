from __future__ import annotations

from fastapi import Request

try:
    from ..core.config import AppConfig
    from ..services.pipeline_service import SearchlightPipelineService
except ImportError:
    from core.config import AppConfig
    from services.pipeline_service import SearchlightPipelineService



def get_app_config(request: Request) -> AppConfig:
    return request.app.state.config



def get_pipeline_service(request: Request) -> SearchlightPipelineService:
    return request.app.state.pipeline_service
