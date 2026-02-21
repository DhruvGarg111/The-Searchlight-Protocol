from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

try:
    from ..core.dependencies import get_app_config, get_pipeline_service
    from ..core.config import AppConfig
    from ..models.api import ErrorResponse, RunPipelineResponse
    from ..models.pipeline import PipelineSettings
    from ..services.pipeline_service import SearchlightPipelineService
    from ..utils.validation import (
        safe_suffix,
        validate_content_type,
        validate_image_payload,
        validate_upload_size,
    )
except ImportError:
    from core.dependencies import get_app_config, get_pipeline_service
    from core.config import AppConfig
    from models.api import ErrorResponse, RunPipelineResponse
    from models.pipeline import PipelineSettings
    from services.pipeline_service import SearchlightPipelineService
    from utils.validation import (
        safe_suffix,
        validate_content_type,
        validate_image_payload,
        validate_upload_size,
    )

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.post(
    "/run-pipeline",
    response_model=RunPipelineResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def run_pipeline(
    image: UploadFile = File(...),
    padding_factor: float = Form(0.4, ge=0.0, le=1.0),
    heatmap_threshold: float = Form(0.4, ge=0.0, le=1.0),
    yolo_confidence: float = Form(0.3, ge=0.0, le=1.0),
    min_crop_size: int = Form(120, ge=32, le=4096),
    nms_iou_threshold: float = Form(0.2, ge=0.0, le=1.0),
    pipeline_service: SearchlightPipelineService = Depends(get_pipeline_service),
    config: AppConfig = Depends(get_app_config),
) -> RunPipelineResponse:
    try:
        validate_content_type(image.content_type)

        image_bytes = await image.read()
        await image.close()

        validate_upload_size(image_bytes, max_upload_bytes=config.max_upload_bytes)
        validate_image_payload(
            image_bytes,
            max_dimension=config.max_image_dimension,
            max_pixels=config.max_image_pixels,
        )

        settings = PipelineSettings(
            padding_factor=padding_factor,
            heatmap_threshold=heatmap_threshold,
            yolo_confidence=yolo_confidence,
            min_crop_size=min_crop_size,
            nms_iou_threshold=nms_iou_threshold,
        )

        result = await asyncio.wait_for(
            run_in_threadpool(
                pipeline_service.run_from_bytes,
                image_bytes,
                safe_suffix(image.filename),
                settings,
            ),
            timeout=config.request_timeout_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Pipeline execution timed out.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail="Pipeline execution failed.") from exc

    result["input_filename"] = image.filename or "uploaded-image"
    return RunPipelineResponse.model_validate(result)
