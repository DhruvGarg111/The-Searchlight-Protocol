# Known Issues & Backlog

This document acts as an active tracker for known performance bottlenecks, technical debt, and architectural risks identified in the codebase (reproduced from `PROJECT_ISSUES.md`).

---

## 📌 Severity Table

| ID | Title | Priority | Status | Impact Area |
| :--- | :--- | :--- | :--- | :--- |
| **SL-01** | Redundant Forward/Backward Passes | **High** | Resolved | CAM Performance |
| **SL-02** | Missing Global NMS for Remapped Boxes | **High** | Mitigated | Target Accuracy |
| **SL-03** | Sequential YOLO Inference for Crops | **Medium** | Resolved | Detection Latency |
| **SL-04** | Massive Base64 Response Payload | **High** | Mitigated | Memory & Network |
| **SL-05** | Hook Collision Risk in Concurrency | **High** | Mitigated | Thread Safety |
| **SL-06** | Temporary File Leakage | **Medium** | Resolved | File System space |
| **SL-07** | Sub-optimal CAM Target Selection | **Medium** | Open | Guidance Quality |
| **SL-08** | Inefficient Triple Image Decoding | **Low** | Mitigated | CPU overhead |
| **SL-09** | ResNet18/50 README discrepancy | **Resolved**| Closed | Documentation |

---

## 🔍 Detailed Issue Backlog

### SL-01: Redundant Forward and Backward Passes in MultiLayerCAM
*   **Description**: `MultiLayerCAM.generate_combined_cam` in `LayerCam.py` iterates over target layers and calls `cam_engine.generate()` individually. This runs independent forward/backward loops, doubling or tripling runtime latency.
*   **Status**: Resolved. `MultiLayerCAM` now attaches all hooks together and executes a single forward/backward sweep to collect activation gradients.

### SL-02: Missing Global NMS for Remapped Detections
*   **Description**: In `pipeline_service.py`, `_run_yolo` collects bounding boxes remapped to original coordinates from overlapping crops but does not apply global Non-Maximum Suppression. This leads to duplicate boxes at crop boundaries.
*   **Status**: Mitigated. Class-aware global NMS is available behind `enable_global_nms` / `SEARCHLIGHT_ENABLE_GLOBAL_NMS` and remains off by default pending fixture validation.

### SL-03: Sequential YOLO Inference for Crops
*   **Description**: `_run_yolo` loops through crop images one by one. This prevents YOLO from using batched GPU execution features.
*   **Status**: Resolved. Retained crops are passed to YOLO as a single source list.

### SL-04: Massive Base64 Response Payload Causing Memory Risks
*   **Description**: The API returns high-resolution visual debugging frames embedded as Base64 strings.
*   **Status**: Mitigated. `full` remains the default contract, while `display` and `metadata` profiles reduce image payloads for frontend and automation clients.

### SL-05: Hook Collision Risk in Concurrent Pipeline Execution
*   **Description**: Concurrent requests registering/unregistering forward and backward hooks on the cached singleton ResNet model cause race conditions when `SEARCHLIGHT_SERIAL_EXECUTION` is disabled.
*   **Status**: Mitigated. A CAM-stage lock guards hook registration and backward execution even when full request serialization is disabled.

### SL-06: Temporary File Leakage in `run_from_bytes`
*   **Description**: If the backend process crashes mid-inference, temporary files stored in `/tmp` are not cleaned up.
*   **Status**: Resolved. Uploads now use the in-memory byte loading path; `run_from_path` remains for local scripts.

### SL-07: Sub-optimal CAM Target Selection (`output.max()`)
*   **Description**: Backpropagating from `output.max()` limits guidance activation to a single, arbitrary ImageNet class, which might ignore other relevant object categories.
*   **Suggested Fix**: Base CAM generation on a union of all relevant target category logits.

### SL-08: Inefficient Triple Image Decoding
*   **Description**: Upload validation decoded the image, then `run_from_bytes` wrote a temp file and loaded it from disk for preprocessing.
*   **Status**: Mitigated. Upload execution now uses `DroneImageLoader.load_bytes()` to avoid temp-file round trips after validation.
