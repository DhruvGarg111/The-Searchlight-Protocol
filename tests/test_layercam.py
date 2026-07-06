from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from LayerCam import MultiLayerCAM


class FakeCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.layer2 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, padding=1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        tensor = self.layer2(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer4(tensor)
        tensor = self.pool(tensor).flatten(1)
        return self.classifier(tensor)


def test_multilayer_cam_uses_one_model_pass_and_normalizes_outputs() -> None:
    torch.manual_seed(7)
    model = FakeCNN()
    cam_engine = MultiLayerCAM(
        model,
        [model.layer2[0], model.layer3[0], model.layer4[0]],
        use_amp=False,
    )

    try:
        combined = cam_engine.generate_combined_cam(
            torch.rand(1, 3, 16, 16),
            weights=[0.7, 0.9, 1.0],
        )
    finally:
        cam_engine.remove()

    assert model.forward_calls == 1
    assert combined.shape == (16, 16)
    assert len(cam_engine.individual_cams) == 3
    assert all(cam.shape == (16, 16) for cam in cam_engine.individual_cams)
    assert np.isfinite(combined).all()
    assert 0.0 <= float(combined.min()) <= float(combined.max()) <= 1.0
