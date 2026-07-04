# API Reference

The backend exposes a lightweight REST API surface. All endpoints are prefixed with `/api`.

---

## 🟢 GET `/api/health`

Retrieves the current execution status and check telemetry.

### Response `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2026-07-04T15:30:00.000000Z",
  "models_ready": true,
  "device": "cuda"
}
```

---

## 🔵 POST `/api/run-pipeline`

Runs the three-stage detection pipeline on the provided image bytes.

### Request Format
*   **Content-Type**: `multipart/form-data`

### Form Fields
| Field | Type | Default | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `image` | `File` | *Required* | — | The aerial scene image file (JPEG, PNG, TIFF). |
| `padding_factor` | `Float` | `0.4` | $[0.0, 1.0]$ | Relative padding added to extracted crop bounding boxes. |
| `heatmap_threshold` | `Float` | `0.4` | $[0.0, 1.0]$ | Activation cutoff score above which regions are sliced. |
| `yolo_confidence` | `Float` | `0.3` | $[0.0, 1.0]$ | Min confidence threshold for YOLO detections. |
| `min_crop_size` | `Int` | `120` | $[32, 4096]$ | Resolution floor in pixels for any sliced crop. |
| `nms_iou_threshold` | `Float` | `0.2` | $[0.0, 1.0]$ | IoU threshold for candidate crop deduplication. |

---

### Response `200 OK`
*   **Model**: `RunPipelineResponse`

#### Field Definitions:
*   `input_filename`: The name of the uploaded image file.
*   `meta`: GPU status and software versions.
*   `settings`: Configuration settings applied.
*   `counts`: Total count of crops (pre-NMS & post-NMS) and final objects detected.
*   `outputs`: Base64-encoded visual maps (original, layer CAM maps, fused heatmap, slicer mask, bounding crops, final annotated detections).
*   `crops`: Array of crop specifications, containing `id`, `score`, global `bbox` coordinates, and base64 crop image.
*   `detections`: Array of objects found, containing `crop_id`, `class` name, `confidence`, and `global_bbox` coordinates `[x1, y1, x2, y2]`.

---

### Error Responses

#### `400 Bad Request`
Raised if validation fails (e.g. invalid file format, exceeds file size limit, or exceeds maximum allowed pixels).
```json
{
  "detail": "File size exceeds limit of 26214400 bytes."
}
```

#### `504 Gateway Timeout`
Raised if execution takes longer than the allowed timeout threshold.
```json
{
  "detail": "Pipeline execution timed out."
}
```

#### `500 Internal Server Error`
Raised if an unexpected model error occurs.
```json
{
  "detail": "Pipeline execution failed."
}
```
