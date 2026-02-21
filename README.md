<div align="center">

<svg width="100%" viewBox="0 0 1400 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The Searchlight Protocol hero banner">
  <rect width="1400" height="300" fill="#0b1220"/>

  <rect x="0" y="22" width="1400" height="2" fill="#7c3aed" opacity="0.75"/>
  <rect x="0" y="278" width="1400" height="2" fill="#06b6d4" opacity="0.75"/>

  <circle cx="170" cy="84" r="64" fill="#7c3aed" opacity="0.10"/>
  <circle cx="1220" cy="210" r="86" fill="#06b6d4" opacity="0.10"/>
  <circle cx="1040" cy="72" r="42" fill="#f59e0b" opacity="0.12"/>

  <path d="M120 238 L380 52" stroke="#7c3aed" stroke-width="1.5" opacity="0.35"/>
  <path d="M960 246 L1240 62" stroke="#06b6d4" stroke-width="1.5" opacity="0.35"/>
  <path d="M520 248 L740 54" stroke="#f59e0b" stroke-width="1.5" opacity="0.28"/>

  <text x="700" y="134" text-anchor="middle" fill="#e2e8f0" font-family="Segoe UI, Arial, sans-serif" font-size="50" font-weight="700" letter-spacing="2">
    THE SEARCHLIGHT PROTOCOL
  </text>
  <text x="700" y="178" text-anchor="middle" fill="#94a3b8" font-family="Segoe UI, Arial, sans-serif" font-size="20" letter-spacing="0.8">
    Coarse-to-fine aerial intelligence for semantic guidance, selective slicing, and targeted detection
  </text>

  <rect x="486" y="206" width="128" height="4" fill="#7c3aed" opacity="0.85" rx="2"/>
  <rect x="636" y="206" width="128" height="4" fill="#06b6d4" opacity="0.85" rx="2"/>
  <rect x="786" y="206" width="128" height="4" fill="#f59e0b" opacity="0.85" rx="2"/>
</svg>

</div>

<div align="center">

<table>
  <tr>
    <td><img src="https://img.shields.io/badge/Python-3.10%2B-334155?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></td>
    <td><img src="https://img.shields.io/badge/PyTorch-2.x-475569?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></td>
    <td><img src="https://img.shields.io/badge/YOLOv8-Fusion-0f172a?style=for-the-badge" alt="YOLOv8"/></td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/FastAPI-Backend-0f766e?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/></td>
    <td><img src="https://img.shields.io/badge/React-Frontend-1e3a8a?style=for-the-badge&logo=react&logoColor=white" alt="React"/></td>
    <td><img src="https://img.shields.io/badge/Status-Research%20Prototype-1f2937?style=for-the-badge" alt="Research Prototype"/></td>
  </tr>
</table>

</div>

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

<table>
<tr>
<td width="100%" style="padding:16px; border:1px solid #1e293b; border-radius:8px;">
<b>PROJECT SUMMARY</b><br/>
The Searchlight Protocol is a coarse-to-fine aerial detection system that prioritizes semantic attention before localized inference. It reduces unnecessary detector compute by isolating high-information regions before running object detection.
</td>
</tr>
</table>

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Visual Pipeline

<div align="center">
<svg width="100%" viewBox="0 0 1180 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Guide to Slicer to Detector pipeline">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L0,10 L10,5 z" fill="#64748b"/>
    </marker>
  </defs>

  <rect x="70" y="52" width="300" height="116" rx="14" fill="#1e132f" stroke="#7c3aed" stroke-width="2"/>
  <rect x="70" y="52" width="300" height="116" rx="14" fill="#7c3aed" opacity="0.08"/>
  <text x="220" y="98" text-anchor="middle" fill="#e9d5ff" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="600">GUIDE</text>
  <text x="220" y="125" text-anchor="middle" fill="#c4b5fd" font-family="Segoe UI, Arial, sans-serif" font-size="14">ResNet18 + LayerCAM</text>

  <line x1="388" y1="110" x2="458" y2="110" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="470" y="52" width="300" height="116" rx="14" fill="#0f2530" stroke="#06b6d4" stroke-width="2"/>
  <rect x="470" y="52" width="300" height="116" rx="14" fill="#06b6d4" opacity="0.08"/>
  <text x="620" y="98" text-anchor="middle" fill="#cffafe" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="600">SLICER</text>
  <text x="620" y="125" text-anchor="middle" fill="#67e8f9" font-family="Segoe UI, Arial, sans-serif" font-size="14">Heatmap-based intelligent slicing</text>

  <line x1="788" y1="110" x2="858" y2="110" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="870" y="52" width="240" height="116" rx="14" fill="#2b210f" stroke="#f59e0b" stroke-width="2"/>
  <rect x="870" y="52" width="240" height="116" rx="14" fill="#f59e0b" opacity="0.08"/>
  <text x="990" y="98" text-anchor="middle" fill="#fef3c7" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="600">DETECTOR</text>
  <text x="990" y="125" text-anchor="middle" fill="#fcd34d" font-family="Segoe UI, Arial, sans-serif" font-size="14">YOLOv8 + global fusion</text>
</svg>
</div>

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Problem Motivation

High-resolution aerial frames make small-object detection difficult when detector input dimensions are constrained.

- Full-frame downscaling can erase small targets.
- Exhaustive slicing spends compute on low-information background.
- A practical pipeline needs semantic prioritization before local detection.

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Methodology

<table>
<tr>
<td width="100%" style="padding:16px; border:1px solid #1e293b; border-radius:8px;">
<b>STAGE 1 - GUIDE</b><br/>
ResNet18 + LayerCAM<br/>
Generates spatial relevance maps from multiple backbone depths (`layer2[-1]`, `layer3[-1]`, `layer4[-1]`) and fuses them with weighted aggregation (`0.4, 1.0, 1.0`).
</td>
</tr>
</table>

<table>
<tr>
<td width="100%" style="padding:16px; border:1px solid #1e293b; border-radius:8px;">
<b>STAGE 2 - SLICER</b><br/>
Activation-based intelligent slicing<br/>
Thresholds the fused heatmap, extracts contours, pads candidate regions, enforces minimum crop size, and applies NMS to reduce overlap.
</td>
</tr>
</table>

<table>
<tr>
<td width="100%" style="padding:16px; border:1px solid #1e293b; border-radius:8px;">
<b>STAGE 3 - DETECTOR</b><br/>
YOLOv8 detection with global remapping<br/>
Runs inference only on retained crops and maps local detections to image-level coordinates for final fused output.
</td>
</tr>
</table>

Coordinate remapping:

`[x1, y1, x2, y2]_global = [x1 + x_c, y1 + y_c, x2 + x_c, y2 + y_c]`

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Architectural Flow

1. Load and validate aerial frame.
2. Build multi-layer semantic attention map.
3. Slice high-activation regions into focused crops.
4. Run YOLO on selected crops only.
5. Fuse and remap detections to global coordinates.
6. Return overlays, metadata, and per-stage timings.

Reference orchestrator: `webapp/backend/services/pipeline_service.py`.

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Performance Characteristics

This repository does not claim fixed benchmark numbers. Runtime is scene-dependent and hardware-dependent.

Per-stage profiling fields are returned in `research.timings_ms`:

- `image_load`
- `layercam_generation`
- `intelligent_slicing`
- `crop_nms`
- `yolo_detection`
- `total_pipeline`

Current packaged detector checkpoint: `webapp/backend/yolov8n.pt`.

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Results

<div align="center">

<table>
  <tr>
    <td align="center" width="33%">
      <img src="Images/Output_1.jpg" width="300" alt="Output 1"/><br/>
      <sub>Output 1: attention, crop selection, and final detections.</sub>
    </td>
    <td align="center" width="33%">
      <img src="Images/Output_2.jpg" width="300" alt="Output 2"/><br/>
      <sub>Output 2: high-density scene under semantic slicing.</sub>
    </td>
    <td align="center" width="33%">
      <img src="Images/Output_3.jpg" width="300" alt="Output 3"/><br/>
      <sub>Output 3: end-to-end qualitative output.</sub>
    </td>
  </tr>
</table>

<p>
  <img src="Images/Architectural%20Diagram.png" width="86%" alt="Searchlight Protocol architecture"/>
</p>

</div>

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Web Application

<table>
<tr>
<td width="50%" style="padding:16px; border:1px solid #1e293b; border-radius:8px;">
<b>Frontend (React)</b><br/>
Interactive interface for input upload, parameter control, and visualization of attention maps, crops, and detections.
</td>
<td width="50%" style="padding:16px; border:1px solid #1e293b; border-radius:8px;">
<b>Backend (FastAPI)</b><br/>
Inference API handling validation, model lifecycle, orchestration, and structured response generation.
</td>
</tr>
</table>

Inference flow:

`Upload -> LayerCAM guidance -> Intelligent slicing -> YOLO inference -> Global fusion -> JSON response`

API endpoints:

- `GET /api/health`
- `POST /api/run-pipeline`

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Repository Structure

```text
project/
|-- LayerCam.py
|-- Slicer.py
|-- Detector.py
|-- ImageLoader.py
|-- The_searchlight_Protocol_main.ipynb
|-- Images/
`-- webapp/
    |-- backend/
    |   |-- main.py
    |   |-- routers/
    |   |-- services/pipeline_service.py
    |   `-- yolov8n.pt
    `-- frontend/
        |-- src/
        `-- package.json
```

Module notes:

- `LayerCam.py`: LayerCAM generation and multi-layer map fusion.
- `Slicer.py`: Heatmap thresholding, contour extraction, crop padding, and crop scoring.
- `Detector.py`: YOLO wrapper for controlled crop-level inference.
- `webapp/backend/services/pipeline_service.py`: End-to-end pipeline execution.
- `webapp/frontend/src/`: Research console UI and visualization components.

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Setup and Execution

Requirements:

- Python 3.10+
- Node.js 18+
- Optional CUDA GPU

Install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r webapp/backend/requirements.txt
```

Run backend:

```bash
cd webapp/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Run frontend:

```bash
cd webapp/frontend
npm install
npm run dev
```

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Free Deployment (Hugging Face + Vercel)

This repository includes a root-level `Dockerfile` and `.dockerignore` so the backend can import root modules (`LayerCam.py`, `Slicer.py`, `Detector.py`, `ImageLoader.py`) while serving FastAPI from `webapp/backend`.

### 1. Deploy Backend to Hugging Face Spaces (Docker)

1. Create a new Hugging Face Space with:
   - SDK: `Docker`
   - Hardware: `CPU Basic (free)`
2. Push this full repository to the Space (do not push `webapp/backend` alone).
3. Configure Space variables:
   - `SEARCHLIGHT_PRELOAD_MODELS=true`
   - `SEARCHLIGHT_SERIAL_EXECUTION=true`
   - `SEARCHLIGHT_LOG_LEVEL=INFO`
4. After build is `Running`, copy the Space URL:
   - `https://<your-username>-<space-name>.hf.space`

### 2. Deploy Frontend to Vercel

1. Import the same GitHub repository in Vercel.
2. Set `Root Directory` to `webapp/frontend`.
3. Add environment variable:
   - `VITE_API_BASE_URL=https://<your-username>-<space-name>.hf.space`
   - No trailing slash.
4. Deploy and copy the production URL:
   - `https://<your-project>.vercel.app`

### 3. Finalize CORS

Add this Hugging Face Space variable and let the Space restart:

- `SEARCHLIGHT_ALLOW_ORIGINS=https://<your-project>.vercel.app`

This backend currently assumes a production-only CORS policy.

### 4. Validation Checklist

- Backend health:
  - `GET https://<space>.hf.space/api/health` returns `200` with `status: "ok"`.
- Frontend integration:
  - Uploading an image from Vercel succeeds with no browser CORS errors.
- Negative cases:
  - Non-image upload returns `400`.
  - Oversized upload returns `400`.
  - Requests from non-whitelisted origins are blocked by CORS.

### 5. Rollback

If a deployment fails after changes, revert the last commit in the Hugging Face Space repo and redeploy the previous known-good image.

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

## Roadmap

- [x] Core 3-stage pipeline
- [x] Intelligent slicing
- [x] Web inference console
- [ ] Temporal consistency
- [ ] Edge deployment
- [ ] Tracking integration

<svg width="100%" height="6">
  <rect width="100%" height="6" fill="#0ea5e9" opacity="0.15"/>
</svg>

<div align="center">

Experimental Aerial Intelligence System  
Designed for high-resolution semantic localization and targeted inference.

</div>
