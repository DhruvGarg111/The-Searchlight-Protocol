# Module: LayerCam

The `LayerCam` module implements class activation mapping algorithms that isolate high-information spatial zones in image classifiers.

---

## 🛠️ Class: LayerCAM

### `__init__`
```python
def __init__(
    self,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    use_amp: bool = True
) -> None
```

Sets up hooks to capture activations and gradients at a specific target layer.

#### Parameters:
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `torch.nn.Module` | *Required* | Classification backbone model (e.g. ResNet). |
| `target_layer` | `torch.nn.Module` | *Required* | Sub-module block hook target. |
| `use_amp` | `bool` | `True` | Uses Mixed Precision if CUDA is available. |

---

### `generate`
```python
def generate(
    self,
    input_tensor: torch.Tensor
) -> np.ndarray
```

Calculates the Class Activation Map.

#### Parameters:
| Name | Type | Description |
| :--- | :--- | :--- |
| `input_tensor` | `torch.Tensor` | Normalized input tensor of shape (1, 3, H, W). |

#### Returns:
*   `np.ndarray`: 2D normalized array of values in $[0, 1]$ matching the input height and width.

---

### `remove`
```python
def remove(self) -> None
```

Removes hooks from target layer. Must be called to avoid leaks.

---

## 🛠️ Class: MultiLayerCAM

### `__init__`
```python
def __init__(
    self,
    model: torch.nn.Module,
    target_layers: list[torch.nn.Module]
) -> None
```

Wraps multiple LayerCAM runs across different network layers.

#### Parameters:
| Name | Type | Description |
| :--- | :--- | :--- |
| `model` | `torch.nn.Module` | Backbone module. |
| `target_layers` | `list[torch.nn.Module]` | List of sub-modules to capture. |

---

### `generate_combined_cam`
```python
def generate_combined_cam(
    self,
    input_tensor: torch.Tensor,
    weights: list[float] | None = None
) -> np.ndarray
```

Runs inference across each registered hook target and aggregates maps.

#### Parameters:
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_tensor` | `torch.Tensor` | *Required* | Normalized input tensor. |
| `weights` | `list[float] \| None` | `None` | Linear aggregation weights. Defaults to equal weights. |

#### Returns:
*   `np.ndarray`: Normalized aggregated activation heatmap.

#### Example Usage:
```python
from LayerCam import MultiLayerCAM
import torchvision.models as models

resnet = models.resnet18(pretrained=True)
target_blocks = [resnet.layer2[-1], resnet.layer3[-1], resnet.layer4[-1]]

cam_engine = MultiLayerCAM(resnet, target_blocks)
try:
    heatmap = cam_engine.generate_combined_cam(tensor, weights=[0.7, 0.9, 1.0])
finally:
    cam_engine.remove()
```
