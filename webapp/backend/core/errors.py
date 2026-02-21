from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger(__name__)



def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(  # type: ignore[unused-ignore]
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        LOGGER.warning("Validation error on %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Invalid request payload.",
                "error_code": "VALIDATION_ERROR",
                "errors": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(  # type: ignore[unused-ignore]
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        LOGGER.exception("Unhandled server exception on %s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error.",
                "error_code": "INTERNAL_ERROR",
            },
        )
