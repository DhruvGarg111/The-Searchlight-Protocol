from __future__ import annotations

"""Backward-compatibility shim for legacy imports.

Older code imports `NotebookPipelineRunner` and `PipelineSettings` from this module.
The new architecture keeps these names while delegating to the service/model packages.
"""

try:
    from .core.config import get_config
    from .models.pipeline import PipelineSettings
    from .services.pipeline_service import SearchlightPipelineService
except ImportError:
    from core.config import get_config
    from models.pipeline import PipelineSettings
    from services.pipeline_service import SearchlightPipelineService


class NotebookPipelineRunner(SearchlightPipelineService):
    def __init__(self) -> None:
        super().__init__(get_config())
