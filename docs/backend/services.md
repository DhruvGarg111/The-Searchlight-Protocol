# Service Layer internals

The `SearchlightPipelineService` acts as the primary orchestrator that binds together raw processing classes into a single transactional request logic.

---

## 🛠️ Service Design

```
+-----------------------------------------------------------+
|               SearchlightPipelineService                 |
+-----------------------------------------------------------+
| - _guide_model: torch.nn.Module                           |
| - _detector: YOLODetector                                 |
| - _model_lock: threading.Lock                             |
| - _run_lock: threading.Lock                               |
+-----------------------------------------------------------+
| + warmup()                                                |
| + run_from_bytes(bytes, suffix, settings)                 |
| + run_from_path(path, settings)                           |
+-----------------------------------------------------------+
```

---

## 💡 Thread Safety and Locking

PyTorch models and underlying GPU CUDA contexts are generally not thread-safe when concurrent request threads try to mutate gradients or write activation hooks simultaneously.

To address this, the service implements two levels of thread safety:

### 1. Model Initialization Lock (`_model_lock`)
Used during lazy-loading inside `_ensure_guide_model()` and `_ensure_detector()`:
```python
with self._model_lock:
    if self._guide_model is None:
        self._guide_model = models.resnet18(...)
```
This guarantees that only one request thread can instantiate the singleton model properties, avoiding memory duplication or write collisions.

### 2. Request Serialization Lock (`_run_lock`)
If `SEARCHLIGHT_SERIAL_EXECUTION` is set to `True` (recommended for CPU environments or GPU setups with limited VRAM), the service forces all incoming pipeline requests to wait in a queue, executing them sequentially:

```python
lock_ctx = self._run_lock if self.config.serial_execution else nullcontext()
with lock_ctx:
    # Run pipeline stage 1, 2, 3
```

This prevents multiple execution threads from registering conflicting LayerCAM forward/backward hooks, which would corrupt gradients and crash the application.

---

## 💾 Image Encoding and Payload Size Risk

The service encodes visual debugging frames back to the client as Base64 Data URLs inside the JSON response:
*   `outputs.original_image`
*   `outputs.weighted_fusion_cam`
*   `outputs.post_nms_boundaries`
*   `outputs.final_detections`
*   Individual crops

This allows the React frontend to easily render output canvas elements without a separate file server storage setup.

> [!WARNING]
> While convenient, Base64 strings increase image payload size by approximately 33%. For a single 4K image, returning nine individual image copies can bloat response sizes to over 100 MB. In resource-constrained deployment targets (e.g., Hugging Face Spaces free tier), this can lead to Out-Of-Memory (OOM) process termination.
