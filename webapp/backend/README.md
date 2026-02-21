# Searchlight Backend

FastAPI backend for the Searchlight Protocol inference pipeline.

## Scope

This service provides:
- model lifecycle management (ResNet guide + YOLO detector)
- request validation and runtime guardrails
- pipeline execution and structured response payloads

## Package Layout

- `main.py`: app factory, CORS, startup/shutdown hooks
- `core/config.py`: environment-based settings
- `core/dependencies.py`: dependency injection helpers
- `core/errors.py`: standardized error handlers
- `routers/health.py`: health endpoint
- `routers/pipeline.py`: inference endpoint
- `services/pipeline_service.py`: pipeline orchestration and model caching
- `models/pipeline.py`: runtime parameter schema
- `models/api.py`: response schema
- `utils/validation.py`: upload and image validation
- `pipeline.py`: backward-compatibility shim

## Run Locally

```bash
# from repository root
pip install -r requirements.txt
pip install -r webapp/backend/requirements.txt
```

```bash
cd webapp/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

For GPU inference, use a conservative worker count to avoid model duplication.

## Deploy on Hugging Face Spaces (Docker)

This backend is coupled to root modules (`LayerCam.py`, `Slicer.py`, `Detector.py`, `ImageLoader.py`), so deployment must use the repository root `Dockerfile`, not `webapp/backend` in isolation.

Deployment steps:
1. Create a Hugging Face Space using `Docker` SDK.
2. Push the full repository to the Space.
3. Set Space variables:
   - `SEARCHLIGHT_PRELOAD_MODELS=true`
   - `SEARCHLIGHT_SERIAL_EXECUTION=true`
   - `SEARCHLIGHT_LOG_LEVEL=INFO`
4. After frontend deployment, set:
   - `SEARCHLIGHT_ALLOW_ORIGINS=https://<your-vercel-project>.vercel.app`

The container runs:

```bash
uvicorn webapp.backend.main:app --host 0.0.0.0 --port 7860
```

## API

### `GET /api/health`
Returns:
- `status`
- `device`
- `models_ready`

### `POST /api/run-pipeline`
`multipart/form-data` fields:
- `image` (required)
- `padding_factor` (default `0.4`)
- `heatmap_threshold` (default `0.4`)
- `yolo_confidence` (default `0.3`)
- `min_crop_size` (default `120`)
- `nms_iou_threshold` (default `0.2`)

ResNet guide preprocessing uses a fixed max image dimension of `1024`.

Response includes:
- `meta`, `settings`, `counts`, `research`
- `outputs` (base64 visualizations)
- `crops`, `detections`, `input_filename`

## Runtime Notes

- Models are cached in `SearchlightPipelineService`.
- Inference executes in a threadpool and is wrapped with timeout control.
- Upload validation checks media type, byte size, dimensions, and pixel count.

## Environment Variables

Primary variables:
- `SEARCHLIGHT_ALLOW_ORIGINS`
- `SEARCHLIGHT_LOG_LEVEL`
- `SEARCHLIGHT_MAX_UPLOAD_BYTES`
- `SEARCHLIGHT_MAX_IMAGE_DIMENSION`
- `SEARCHLIGHT_MAX_IMAGE_PIXELS`
- `SEARCHLIGHT_REQUEST_TIMEOUT_SECONDS`
- `SEARCHLIGHT_PRELOAD_MODELS`
- `SEARCHLIGHT_CLEAR_CUDA_CACHE_PER_REQUEST`
- `SEARCHLIGHT_SERIAL_EXECUTION`
- `SEARCHLIGHT_YOLO_MODEL_VERSION`
- `SEARCHLIGHT_YOLO_MODEL_VARIANT`
- `SEARCHLIGHT_YOLO_MODEL_PATH`

Defaults are defined in `core/config.py`.

## Production Smoke Checks

1. `GET /api/health` returns `200` and `status: "ok"`.
2. `POST /api/run-pipeline` accepts a valid image and returns inference output.
3. Invalid content type returns `400`.
4. Oversized uploads return `400`.
5. Requests from non-whitelisted origins are blocked by CORS.
