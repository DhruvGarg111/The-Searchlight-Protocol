# Environment Configuration Reference

The FastAPI backend is configured using standard environment variables. This reference outlines the purpose, default values, and valid choices for each configuration parameter.

---

## ⚙️ Configuration Variables

| Variable Name | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `SEARCHLIGHT_API_TITLE` | `String` | `"Searchlight Pipeline API"` | Title shown in swagger docs. |
| `SEARCHLIGHT_API_VERSION` | `String` | `"1.0.0"` | Version code of the API. |
| `SEARCHLIGHT_ALLOW_ORIGINS` | `Comma List` | `http://localhost:5173,http://127.0.0.1:5173` | CORS allowed origins. |
| `SEARCHLIGHT_LOG_LEVEL` | `String` | `"INFO"` | Logging severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `SEARCHLIGHT_MAX_UPLOAD_BYTES` | `Int` | `26214400` (25 MB) | Absolute maximum allowed size for uploaded files. |
| `SEARCHLIGHT_MAX_IMAGE_DIMENSION` | `Int` | `12000` | Maximum pixel width or height allowed for validation. |
| `SEARCHLIGHT_MAX_IMAGE_PIXELS` | `Int` | `90000000` (90 MP) | Maximum allowed total pixels ($W \times H$) for validation. |
| `SEARCHLIGHT_REQUEST_TIMEOUT_SECONDS` | `Float` | `240.0` | Timeout threshold for async request offloading. |
| `SEARCHLIGHT_PRELOAD_MODELS` | `Bool` | `True` | Eagerly loads models during startup to reduce initial latency. |
| `SEARCHLIGHT_CLEAR_CUDA_CACHE_PER_REQUEST`| `Bool` | `False` | Forces cuda cache clear after every pipeline execution. |
| `SEARCHLIGHT_SERIAL_EXECUTION` | `Bool` | `True` | Locks execution threads to serialize incoming requests. |
| `SEARCHLIGHT_YOLO_MODEL_VERSION` | `String` | `"v8"` | Model version directory suffix (e.g. `"v8"`, `"v9"`). |
| `SEARCHLIGHT_YOLO_MODEL_VARIANT` | `String` | `"n"` | Model weight category (e.g. `"n"`, `"s"`, `"m"`, `"l"`). |
| `SEARCHLIGHT_YOLO_MODEL_PATH` | `String` | `yolov8n.pt` | Direct weights path override on disk. |

---

## 📝 Example Configuration Files

### 1. Local Development (`.env`)
```bash
SEARCHLIGHT_LOG_LEVEL=DEBUG
SEARCHLIGHT_PRELOAD_MODELS=false
SEARCHLIGHT_ALLOW_ORIGINS=http://localhost:5173
```

### 2. Production Server / Hugging Face Spaces
```bash
SEARCHLIGHT_PRELOAD_MODELS=true
SEARCHLIGHT_SERIAL_EXECUTION=true
SEARCHLIGHT_CLEAR_CUDA_CACHE_PER_REQUEST=true
SEARCHLIGHT_LOG_LEVEL=INFO
```
