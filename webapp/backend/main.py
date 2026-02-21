from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from .core.config import get_config
    from .core.errors import register_exception_handlers
    from .core.logging import configure_logging
    from .routers.health import router as health_router
    from .routers.pipeline import router as pipeline_router
    from .services.pipeline_service import SearchlightPipelineService
except ImportError:
    from core.config import get_config
    from core.errors import register_exception_handlers
    from core.logging import configure_logging
    from routers.health import router as health_router
    from routers.pipeline import router as pipeline_router
    from services.pipeline_service import SearchlightPipelineService



def create_app() -> FastAPI:
    config = get_config()
    configure_logging(config.log_level)

    app = FastAPI(title=config.api_title, version=config.api_version)
    app.state.config = config
    app.state.pipeline_service = SearchlightPipelineService(config)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(pipeline_router, prefix="/api", tags=["pipeline"])
    register_exception_handlers(app)

    @app.on_event("startup")
    def on_startup() -> None:
        if config.preload_models:
            app.state.pipeline_service.warmup()

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        app.state.pipeline_service.close()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
