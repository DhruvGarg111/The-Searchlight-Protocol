# Module: Detector

The `Detector` module provides standard interfaces for object detection inference operations using cached Ultralytics YOLO engines.

---

## 🛠️ Class: YOLODetector

### `__init__`
```python
def __init__(
    self,
    model_version: str = "v8",
    model_variant: str = "n",
    model_path: str | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None
) -> None
```

Configures and instantiates the YOLO model instance.

#### Parameters:
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_version` | `str` | `"v8"` | Model generation code (e.g. `"v8"`, `"v9"`). |
| `model_variant` | `str` | `"n"` | Model scale option (e.g. `"n"`, `"s"`, `"m"`, `"l"`, `"x"`). |
| `model_path` | `str \| None` | `None` | Custom local path override to model weights file. |
| `conf` | `float` | `0.25` | Default detection confidence threshold. |
| `iou` | `float` | `0.45` | Default intersection-over-union threshold for NMS. |
| `device` | `str \| None` | `None` | Execution target. Auto-selects CUDA if available. |

---

### `predict`
```python
def predict(
    self,
    source: Any,
    conf: float | None = None,
    iou: float | None = None,
    **kwargs: Any
) -> list[Any]
```

Runs object detection inference on the provided source inside a `torch.no_grad()` context.

#### Parameters:
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `Any` | *Required* | Image source (file path, raw NumPy RGB array, PIL image, or a list of inputs). |
| `conf` | `float \| None` | `None` | Overrides default confidence setting. |
| `iou` | `float \| None` | `None` | Overrides default NMS IoU threshold. |
| `**kwargs` | `Any` | — | Forwarded directly to the underlying Ultralytics YOLO predict interface. |

#### Returns:
*   `list[Any]`: A list of `ultralytics.engine.results.Results` instances containing predictions.

#### Example Usage:
```python
from Detector import YOLODetector

detector = YOLODetector(model_version="v8", model_variant="n", conf=0.35)
results = detector.predict("crop_sample.png", augment=True)

for box in results[0].boxes:
    print(f"Class index: {box.cls[0].item()}, Confidence: {box.conf[0].item()}")
```
