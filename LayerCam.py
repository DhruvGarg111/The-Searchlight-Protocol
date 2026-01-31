import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np

class LayerCAM:
    def __init__(self, model, target_layer, use_amp=True):
        self.model = model
        self.target_layer = target_layer
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()

        with autocast(enabled=self.use_amp):
            output = self.model(input_tensor)

        max_score = output.max()
        max_score.backward()

        with torch.no_grad():
            weights = F.relu(self.gradients)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

            cam = cam.cpu().numpy()[0, 0]
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-9)

        self.gradients = None
        self.activations = None

        return cam

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.gradients = None
        self.activations = None


class MultiLayerCAM:
    def __init__(self, model, target_layers):
        self.cams = [LayerCAM(model, layer) for layer in target_layers]
        self.model = model
        self.individual_cams = []

    def generate_combined_cam(self, input_tensor, weights=None):
        if weights is None:
            weights = [1.0 / len(self.cams)] * len(self.cams)

        self.individual_cams = []
        combined = None
        for cam_engine, w in zip(self.cams, weights):
            heatmap = cam_engine.generate(input_tensor)
            self.individual_cams.append(heatmap.copy())
            if combined is None:
                combined = w * heatmap
            else:
                combined += w * heatmap

        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-9)
        return combined

    def remove(self):
        for cam in self.cams:
            cam.remove()
