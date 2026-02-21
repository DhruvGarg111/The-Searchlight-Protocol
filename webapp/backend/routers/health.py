from __future__ import annotations

from fastapi import APIRouter, Depends

try:
    from ..core.dependencies import get_pipeline_service
    from ..models.api import HealthResponse
    from ..services.pipeline_service import SearchlightPipelineService
except ImportError:
    from core.dependencies import get_pipeline_service
    from models.api import HealthResponse
    from services.pipeline_service import SearchlightPipelineService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    pipeline_service: SearchlightPipelineService = Depends(get_pipeline_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(pipeline_service.device),
        models_ready=pipeline_service.models_ready,
    )
