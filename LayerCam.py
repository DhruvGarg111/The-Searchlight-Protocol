import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np

class LayerCAM:
    """Computes Layer-CAM (Class Activation Map) for a specified target layer in a PyTorch model.

    Layer-CAM is a CAM variant that is suitable for producing high-quality class activation
    heatmaps from any convolutional layer, not just the final one. It computes weights using the
    positive gradients of the class score with respect to the layer's activations.
    """

    def __init__(self, model, target_layer, use_amp=True):
        """Initializes LayerCAM by setting up hooks on the target layer.

        Args:
            model (torch.nn.Module): The PyTorch neural network model.
            target_layer (torch.nn.Module): The specific layer (e.g., resnet.layer4[-1]) to capture CAM for.
            use_amp (bool): Whether to use Automatic Mixed Precision (AMP) during forward pass. Defaults to True.
        """
        self.model = model
        self.target_layer = target_layer
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks to capture activations and gradients of the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor):
        """Generates the Layer-CAM heatmap for the given input tensor.

        Args:
            input_tensor (torch.Tensor): Preprocessed input image tensor of shape (1, 3, H, W).

        Returns:
            np.ndarray: A 2D normalized heatmap of shape (H, W) with values in range [0, 1].
        """
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
        """Unregisters the forward and backward hooks and cleans up saved tensors."""
        for h in self.hooks:
            h.remove()
        self.gradients = None
        self.activations = None


class MultiLayerCAM:
    """Aggregates Layer-CAM heatmaps from multiple network depths to capture multi-scale context.

    Fuses coarse, high-level semantic activation maps (from deep layers) with fine-grained,
    low-level structural details (from shallower layers) using weighted summation.
    """

    def __init__(self, model, target_layers):
        """Initializes MultiLayerCAM by instantiating LayerCAM for each target layer.

        Args:
            model (torch.nn.Module): The PyTorch neural network model.
            target_layers (list[torch.nn.Module]): List of layer modules to compute CAMs for.
        """
        self.cams = [LayerCAM(model, layer) for layer in target_layers]
        self.model = model
        self.individual_cams = []

    def generate_combined_cam(self, input_tensor, weights=None):
        """Computes individual CAMs and fuses them into a single aggregated activation heatmap.

        Args:
            input_tensor (torch.Tensor): Preprocessed input image tensor of shape (1, 3, H, W).
            weights (list[float], optional): Relative weights for each layer's CAM.
                If None, equal weights (1 / num_layers) are assigned to all layers.

        Returns:
            np.ndarray: Fused and normalized 2D heatmap of shape (H, W) in range [0, 1].
        """
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
        """Cleans up and removes hooks from all individual LayerCAM instances."""
        for cam in self.cams:
            cam.remove()
