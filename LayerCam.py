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

    def __init__(self, model, target_layers, use_amp=True):
        """Initializes MultiLayerCAM by registering hooks for all target layers.

        Args:
            model (torch.nn.Module): The PyTorch neural network model.
            target_layers (list[torch.nn.Module]): List of layer modules to compute CAMs for.
            use_amp (bool): Whether to use Automatic Mixed Precision (AMP) during forward pass.
        """
        self.model = model
        self.target_layers = list(target_layers)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.individual_cams = []
        self.activations = [None] * len(self.target_layers)
        self.gradients = [None] * len(self.target_layers)
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers hooks for all target layers so one backward pass captures every CAM input."""
        for index, layer in enumerate(self.target_layers):
            def forward_hook(module, input, output, layer_index=index):
                self.activations[layer_index] = output.detach()

            def backward_hook(module, grad_input, grad_output, layer_index=index):
                self.gradients[layer_index] = grad_output[0].detach()

            self.hooks.append(layer.register_forward_hook(forward_hook))
            self.hooks.append(layer.register_full_backward_hook(backward_hook))

    def generate_combined_cam(self, input_tensor, weights=None):
        """Computes individual CAMs in one model pass and fuses them into one activation heatmap.

        Args:
            input_tensor (torch.Tensor): Preprocessed input image tensor of shape (1, 3, H, W).
            weights (list[float], optional): Relative weights for each layer's CAM.
                If None, equal weights (1 / num_layers) are assigned to all layers.

        Returns:
            np.ndarray: Fused and normalized 2D heatmap of shape (H, W) in range [0, 1].
        """
        if weights is None:
            weights = [1.0 / len(self.target_layers)] * len(self.target_layers)

        if len(weights) != len(self.target_layers):
            raise ValueError("weights length must match target layer count")

        self.model.eval()
        self.model.zero_grad()

        with autocast(enabled=self.use_amp):
            output = self.model(input_tensor)

        output.max().backward()

        self.individual_cams = []
        combined = None
        for layer_index, w in enumerate(weights):
            heatmap = self._compute_layer_cam(layer_index, input_tensor.shape[2:])
            self.individual_cams.append(heatmap.copy())
            if combined is None:
                combined = w * heatmap
            else:
                combined += w * heatmap

        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-9)
        self._clear_saved_tensors()
        return combined

    def _compute_layer_cam(self, layer_index, output_size):
        """Computes a normalized CAM for one captured layer."""
        gradients = self.gradients[layer_index]
        activations = self.activations[layer_index]

        if gradients is None or activations is None:
            raise RuntimeError(f"Missing CAM tensors for target layer {layer_index}")

        with torch.no_grad():
            weights = F.relu(gradients)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=output_size, mode='bilinear', align_corners=False)

            cam = cam.cpu().numpy()[0, 0]
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-9)

        return cam

    def _clear_saved_tensors(self):
        """Drops captured activations and gradients after CAM computation."""
        self.gradients = [None] * len(self.target_layers)
        self.activations = [None] * len(self.target_layers)

    def remove(self):
        """Cleans up and removes hooks from all target layers."""
        for hook in self.hooks:
            hook.remove()
        self._clear_saved_tensors()
