# Architecture Overview

This document describes the design, system boundary, and technical stack of **The Searchlight Protocol** codebase.

## 🏛️ Codebase Organization

The repository contains two isolated code ecosystems that share the project directory but do not import from one another:

```
The-Searchlight-Protocol/
├── Detector.py                  # Core YOLO wrapper
├── ImageLoader.py               # Preprocessing & Normalization
├── LayerCam.py                  # Class Activation Map engines
├── Slicer.py                    # Region extraction morphological engine
├── PROJECT_ISSUES.md            # Technical debt & backlog
├── README.md                    # Public quickstart
└── webapp/
    ├── backend/                 # FastAPI REST API & orchestration service
    └── frontend/                # React Vite dashboard console
```

## 🔌 Technical Stack

The Aerial Forensics Pipeline utilizes a modern Python + JavaScript stack built for low-latency deep learning inference and real-time visualization:

*   **Deep Learning & Image Processing (Core)**
    *   **PyTorch**: Tensor manipulations, ResNet feature maps, backpropagation gradients.
    *   **Torchvision**: Pretrained weights, torchvision transforms, and GPU-optimized Non-Maximum Suppression (NMS).
    *   **Ultralytics YOLO**: Localized bounding box object detection.
    *   **OpenCV (cv2)**: Contouring, padding, resize operations, data URL drawing, color mapping.
    *   **Pillow**: Image loading, contrast enhancement.
*   **Web Services & API Layer**
    *   **FastAPI**: Uvicorn-based async backend API. Supports request schema validation, CORSMiddleware, and offloads heavy synchronous inference workloads to background threadpools.
    *   **Pydantic**: Typing-safe API request/response modeling.
*   **Client Dashboard Interface**
    *   **React (Vite)**: Clean single-page research console.
    *   **Framer Motion**: Smooth telemetry counters and pipeline graph animations.
    *   **Tailwind CSS**: Dark, futuristic aesthetic custom component tokens.

## 🛰️ Structural Topology

The following diagram details how the frontend console, FastAPI routers, pipeline services, and raw modules interact:

```mermaid
graph TD
    classDef client fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef backend fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    classDef modules fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff;

    FE[React Frontend Console]::client -->|HTTP POST /api/run-pipeline| Router[routers/pipeline.py]::backend
    Router -->|Settings & Bytes| Service[services/pipeline_service.py]::backend
    Service -->|Lazy Load / Warmup| ImgLoader[ImageLoader.py]::modules
    Service -->|Lazy Load / Warmup| CAM[LayerCam.py]::modules
    Service -->|Lazy Load / Warmup| Slicer[Slicer.py]::modules
    Service -->|Lazy Load / Warmup| YOLO[Detector.py]::modules
```
