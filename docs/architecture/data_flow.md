# Data Flow & Lifecycle

This document traces the request-response flow of an image upload through the Searchlight Protocol system.

## 🔄 End-to-End Sequence Diagram

The diagram below details the chronological method calls, network boundaries, and file system interactions that occur when a user executes a run in the dashboard:

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher (Client)
    participant Console as React Console
    participant API as FastAPI Router
    participant Service as Pipeline Service
    participant Loader as Drone Image Loader
    participant CAM as Multi-Layer CAM Engine
    participant Slicer as Intelligent Slicer
    participant Detector as YOLO Detector

    User->>Console: Drag-and-Drop Image & Adjust Parameters
    User->>Console: Click "EXECUTE INFERENCE"
    Console->>API: POST /api/run-pipeline (Multipart Form Data)
    Note over API: validate_content_type()<br/>validate_upload_size()<br/>validate_image_payload()
    API->>Service: run_from_bytes(image_bytes, suffix, settings)
    Service->>Loader: load_bytes(image_bytes)
    Loader->>Loader: Apply Contrast Enhancement
    Loader->>Loader: Rescale to CAM dimensions (Aspect Preserved)
    Loader-->>Service: original_np, input_tensor, original_size, scale_factor
    Service->>Service: _ensure_guide_model()
    Service->>CAM: generate_combined_cam(input_tensor)
    CAM->>CAM: Single forward/backward pass captures all target layers
    CAM-->>Service: individual_cams[]
    Service->>Service: Remove Hooks & Free CUDA Cache
    Service->>Service: Fuse Layer CAM Heatmaps
    Service->>Slicer: slice(original_np, fused_heatmap)
    Slicer->>Slicer: Resize Heatmap & Apply Binary Mask Threshold
    Slicer->>Slicer: cv2.findContours() & Apply Padding Factor
    Slicer-->>Service: crops[], mask, heatmap_resized
    Service->>Service: _apply_nms() to Deduplicate Overlapping Crops
    Service->>Service: _ensure_detector() (YOLOv8)
    Service->>Detector: predict([crop_image, ...])
    Detector-->>Service: batched local detection boxes
    Service->>Service: Remap local boxes to original coordinate offsets
    opt enable_global_nms=true
        Service->>Service: Apply final class-aware global NMS
    end
    Service->>Service: Draw annotations and apply response profile
    Service-->>API: run_from_bytes dict response
    API-->>Console: 200 OK (RunPipelineResponse JSON)
    Console->>User: Display Telemetry Counters, Render Output Canvas
```

## 📦 Payload Schemas

### 1. Request Parameters (Multipart Form)
*   `image`: Binary file payload (JPEG, PNG, TIFF).
*   `padding_factor` (Float, Default: `0.4`): Relative boundary padding.
*   `heatmap_threshold` (Float, Default: `0.4`): Cutoff for activation region selection.
*   `yolo_confidence` (Float, Default: `0.3`): YOLO confidence score.
*   `min_crop_size` (Int, Default: `120`): Minimum resolution floor for crop slicing.
*   `nms_iou_threshold` (Float, Default: `0.2`): Overlap index for crop-level suppression.
*   `yolo_iou_threshold` (Float, Default: `0.6`): YOLO inference NMS threshold.
*   `response_profile` (`full`, `display`, `metadata`; Default: `full`): Controls image payload size.
*   `enable_global_nms` (Bool): Optional final NMS over remapped detections.
*   `global_nms_iou_threshold` (Float, Default: `0.5`): IoU threshold for optional global NMS.

### 2. Response Payload (JSON)
The response is structured into distinct, self-contained sections:
*   `meta`: Execution hardware and version specs.
*   `settings`: Echoes the hyperparameters used for the run.
*   `counts`: Performance metrics (total pre-nms crops, post-nms crops, detections).
*   `research`: Timing profiles (in milliseconds) and detection confidence scores.
*   `outputs`: Base64 data URLs representing output canvases when enabled by the response profile.
*   `crops`: Array of crop metadata and optional crop image payloads.
*   `detections`: Array of remapped global coordinate bounding boxes with category names.
