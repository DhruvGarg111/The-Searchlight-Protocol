# FastAPI Backend Overview

The backend of **The Searchlight Protocol** is a high-performance REST API built using **FastAPI** and served via **Uvicorn**. It handles client requests, performs size/dimension validations, offloads CPU/GPU-intensive inference execution to a worker thread pool, and manages the lifecycle of cached deep learning models.

---

## 📂 Project Directory Structure

```
webapp/backend/
├── main.py                  # API Factory and entry point
├── core/
│   ├── config.py            # Environment-variable parsing & dataclass configuration
│   ├── dependencies.py      # Dependency injection providers (Service, Config)
│   ├── errors.py            # Custom global exception handlers
│   └── logging.py           # Logging format config
├── routers/
│   ├── health.py            # Health Check API endpoints
│   └── pipeline.py          # Main execution router (runs model stack)
├── services/
│   └── pipeline_service.py  # Orchestrates Stage 1/2/3 and caches models
├── models/
│   ├── api.py               # Pydantic schemas for API endpoints
│   └── pipeline.py          # Pydantic schema for parameters
└── utils/
    └── validation.py        # Boundary check utils (file types, dimension limits)
```

---

## ⚡ Key Lifecycle Behaviors

### 1. Asynchronous Offloading
Because image loading, CAM computation, and YOLO inference are heavily CPU/GPU-bound synchronous tasks, calling them inside standard async endpoints would block FastAPI's single-threaded event loop. To avoid this, the backend uses Starlette's `run_in_threadpool` pattern inside `routers/pipeline.py`:

```python
result = await asyncio.wait_for(
    run_in_threadpool(
        pipeline_service.run_from_bytes,
        image_bytes,
        safe_suffix(image.filename),
        settings,
    ),
    timeout=config.request_timeout_seconds,
)
```

This offloads the execution to an internal thread pool, allowing the event loop to concurrently process other lightweight HTTP requests.

### 2. Startup Warmup
Large PyTorch models introduce high latency during their first forward pass due to memory allocation, kernel compilation, and disk read operations (often referred to as "cold-starts").

If `SEARCHLIGHT_PRELOAD_MODELS` is enabled, the backend triggers `pipeline_service.warmup()` during the FastAPI `startup` event, preloading the ResNet18 backbone and YOLO detector weights before opening the server ports.

### 3. Graceful Shutdown
During the `shutdown` event, `pipeline_service.close()` is executed. This releases model handles, clears caches, and calls `torch.cuda.empty_cache()` to ensure clean teardown.
