from __future__ import annotations

import numpy as np
import pytest

from Slicer import IntelligentSlicer


def test_slicer_keeps_crop_bounds_inside_image_and_expands_to_minimum_size() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    heatmap = np.zeros((10, 12), dtype=np.float32)
    heatmap[1, 1] = 1.0

    slicer = IntelligentSlicer(padding_factor=0.0, info_threshold=0.5, min_crop_size=32)
    crops, _, _ = slicer.slice(image, heatmap)

    assert len(crops) == 1
    x, y, width, height = crops[0]["bbox"]
    assert x >= 0
    assert y >= 0
    assert x + width <= image.shape[1]
    assert y + height <= image.shape[0]
    assert width >= 32
    assert height >= 32


def test_crop_nms_keeps_highest_scored_overlapping_crop_deterministically() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from webapp.backend.services.pipeline_service import SearchlightPipelineService

    crops = [
        {"id": 1, "bbox": (0, 0, 50, 50), "score": 0.5},
        {"id": 2, "bbox": (4, 4, 50, 50), "score": 0.9},
        {"id": 3, "bbox": (90, 90, 20, 20), "score": 0.4},
    ]

    kept = SearchlightPipelineService._apply_nms(crops, iou_threshold=0.5)

    assert [crop["id"] for crop in kept] == [2, 3]
