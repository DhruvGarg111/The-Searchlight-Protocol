# Module: ImageLoader

The `ImageLoader` module is responsible for loading high-resolution aerial imagery, applying contrast enhancement to isolate features, downscaling to model resolution guidelines while maintaining aspect ratios, and normalizing tensors.

---

## 🛠️ Class: DroneImageLoader

### `__init__`
```python
def __init__(
    self,
    max_dim: int,
    contrast_factor: float = 1.8,
    device: torch.device | str | None = None
) -> None
```

Initializes the loader engine.

#### Parameters:
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_dim` | `int` | *Required* | Maximum size (height or width) for the returned PyTorch tensor. |
| `contrast_factor` | `float` | `1.8` | Enhancement multiplier. 1.0 disables contrast tuning. |
| `device` | `torch.device \| str \| None` | `None` | PyTorch device target. Auto-selects CUDA if available. |

---

### `load`
```python
def load(
    self,
    image_path: str | Path
) -> tuple[np.ndarray, torch.Tensor, tuple[int, int], float]
```

Reads an image from disk and converts it to model-ready structures.

#### Parameters:
| Name | Type | Description |
| :--- | :--- | :--- |
| `image_path` | `str \| Path` | Path to the target image file. Supports typical Pillow-friendly formats (JPEG, PNG, TIFF). |

#### Returns:
A `tuple` containing:
1.  **`original_np`** (`np.ndarray`): Contrast-enhanced full-resolution original image (RGB format, shape HWC).
2.  **`tensor`** (`torch.Tensor`): Normalised image float32 tensor scaled to fit within `max_dim` (shape 1x3xHxW).
3.  **`original_size`** (`tuple[int, int]`): Tuple showing the absolute `(width, height)` of the source image on disk.
4.  **`scale`** (`float`): Rescaling multiplier applied.

#### Raises:
*   `FileNotFoundError`: If the image does not exist.

#### Example Usage:
```python
from ImageLoader import DroneImageLoader

loader = DroneImageLoader(max_dim=1800, contrast_factor=1.5)
original_np, tensor, size, scale = loader.load("tactical_frame.png")

print(f"Original shape: {original_np.shape}, Model tensor shape: {tensor.shape}")
print(f"Scale applied: {scale} (original size: {size})")
```
