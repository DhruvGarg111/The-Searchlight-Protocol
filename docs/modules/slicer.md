# Module: Slicer

The `Slicer` module extracts regional sub-images from full-resolution inputs based on binary activation thresholds from a guidance heatmap.

---

## 🛠️ Class: IntelligentSlicer

### `__init__`
```python
def __init__(
    self,
    padding_factor: float,
    info_threshold: float,
    min_crop_size: int
) -> None
```

Initializes the slicing engine parameters.

#### Parameters:
| Name | Type | Description |
| :--- | :--- | :--- |
| `padding_factor` | `float` | Context padding ratio around bounding rectangle contours. |
| `info_threshold` | `float` | Cutoff above which heatmap pixels are considered region bounds. |
| `min_crop_size` | `int` | Floor limit for height/width dimensions of sliced crops. |

---

### `slice`
```python
def slice(
    self,
    original_image: np.ndarray,
    heatmap: np.ndarray
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]
```

Extracts regions of interest from the original image based on target regions.

#### Parameters:
| Name | Type | Description |
| :--- | :--- | :--- |
| `original_image` | `np.ndarray` | Raw RGB image array of shape HWC. |
| `heatmap` | `np.ndarray` | Normalized 2D guidance heatmap array. |

#### Returns:
A `tuple` containing:
1.  **`crops`** (`list[dict[str, object]]`): List of dictionaries, each holding crop metadata:
    *   `"id"` (`int`): Sequential index.
    *   `"image"` (`np.ndarray`): The actual cropped image array.
    *   `"bbox"` (`tuple[int, int, int, int]`): Global coordinates as `(x_offset, y_offset, crop_w, crop_h)`.
    *   `"score"` (`float`): Mean activation density inside the crop bounds.
2.  **`mask`** (`np.ndarray`): Thresholded binary mask (uint8, values 0 or 255).
3.  **`heatmap_resized`** (`np.ndarray`): Heatmap resized to match the dimensions of the original input.

#### Example Usage:
```python
from Slicer import IntelligentSlicer

slicer = IntelligentSlicer(padding_factor=0.3, info_threshold=0.4, min_crop_size=120)
crops, mask, resized_heatmap = slicer.slice(image_np, heatmap_np)

print(f"Extracted {len(crops)} crop candidates.")
first_crop = crops[0]
print(f"Crop ID: {first_crop['id']}, Offset: {first_crop['bbox']}")
```
