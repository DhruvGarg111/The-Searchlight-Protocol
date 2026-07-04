# Known Issues & Backlog

This document acts as an active tracker for known performance bottlenecks, technical debt, and architectural risks identified in the codebase (reproduced from `PROJECT_ISSUES.md`).

---

## 📌 Severity Table

| ID | Title | Priority | Status | Impact Area |
| :--- | :--- | :--- | :--- | :--- |
| **SL-01** | Redundant Forward/Backward Passes | **High** | Open | CAM Performance |
| **SL-02** | Missing Global NMS for Remapped Boxes | **High** | Open | Target Accuracy |
| **SL-03** | Sequential YOLO Inference for Crops | **Medium** | Open | Detection Latency |
| **SL-04** | Massive Base64 Response Payload | **High** | Open | Memory & Network |
| **SL-05** | Hook Collision Risk in Concurrency | **High** | Open | Thread Safety |
| **SL-06** | Temporary File Leakage | **Medium** | Open | File System space |
| **SL-07** | Sub-optimal CAM Target Selection | **Medium** | Open | Guidance Quality |
| **SL-08** | Inefficient Triple Image Decoding | **Low** | Open | CPU overhead |
| **SL-09** | ResNet18/50 README discrepancy | **Resolved**| Closed | Documentation |

---

## 🔍 Detailed Issue Backlog

### SL-01: Redundant Forward and Backward Passes in MultiLayerCAM
*   **Description**: `MultiLayerCAM.generate_combined_cam` in `LayerCam.py` iterates over target layers and calls `cam_engine.generate()` individually. This runs independent forward/backward loops, doubling or tripling runtime latency.
*   **Suggested Fix**: Refactor `LayerCAM` to attach hooks simultaneously, executing a single forward/backward sweep to collect all activation gradients.

### SL-02: Missing Global NMS for Remapped Detections
*   **Description**: In `pipeline_service.py`, `_run_yolo` collects bounding boxes remapped to original coordinates from overlapping crops but does not apply global Non-Maximum Suppression. This leads to duplicate boxes at crop boundaries.
*   **Suggested Fix**: Convert remapped coordinate lists to tensors and pass them through `torchvision.ops.nms` before formatting the output payload.

### SL-03: Sequential YOLO Inference for Crops
*   **Description**: `_run_yolo` loops through crop images one by one. This prevents YOLO from using batched GPU execution features.
*   **Suggested Fix**: Package all crop image arrays into a single list and pass them to YOLO: `results = detector.predict(source=[c["image"] for c in crops])`.

### SL-04: Massive Base64 Response Payload Causing Memory Risks
*   **Description**: The API returns high-resolution visual debugging frames embedded as Base64 strings.
*   **Suggested Fix**: Downscale visualization frames before encoding, or save them to local storage disk caches and return lightweight URL endpoints instead.

### SL-05: Hook Collision Risk in Concurrent Pipeline Execution
*   **Description**: Concurrent requests registering/unregistering forward and backward hooks on the cached singleton ResNet model cause race conditions when `SEARCHLIGHT_SERIAL_EXECUTION` is disabled.
*   **Suggested Fix**: Register hooks permanently during startup initialization and route captured gradients to active requests using thread identifiers.

### SL-06: Temporary File Leakage in `run_from_bytes`
*   **Description**: If the backend process crashes mid-inference, temporary files stored in `/tmp` are not cleaned up.
*   **Suggested Fix**: Utilize Python context managers or configure automated scheduler cleanups.

### SL-07: Sub-optimal CAM Target Selection (`output.max()`)
*   **Description**: Backpropagating from `output.max()` limits guidance activation to a single, arbitrary ImageNet class, which might ignore other relevant object categories.
*   **Suggested Fix**: Base CAM generation on a union of all relevant target category logits.
