
<div align="center">

<!-- Dynamic Typing Banner -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=42&pause=1000&color=009688&center=true&vCenter=true&width=800&height=100&lines=The+Searchlight+Protocol;Aerial+Intelligence+System;Coarse-to-Fine+Detection;Semantic+Guidance+%2B+Slicing" alt="Typing SVG" />

**Intelligent layer-based semantic guidance, selective slicing, and targeted detection for high-resolution aerial imagery.**

<br>

<!-- Tech Stack Badges -->
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Fusion-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
<a href="https://deepwiki.com/DhruvGarg111/The-Searchlight-Protocol">
  <img src="https://deepwiki.com/badge.svg" height="28">
</a>
<br>

<!-- Quick Links -->
[**🌐 Launch Web App**](https://the-searchlight-protocol.vercel.app) &nbsp; • &nbsp;
[**🐛 Report Bug**](https://github.com/DhruvGarg111/The-Searchlight-Protocol/issues) &nbsp; • &nbsp;
[**💡 Request Feature**](https://github.com/DhruvGarg111/The-Searchlight-Protocol/issues)

</div>

---

<div align="center">
  <i>High-resolution aerial frames make small-object detection incredibly difficult when detector compute is constrained. <br>The Searchlight Protocol offers a <b>coarse-to-fine pipeline</b> that uses semantic prioritization to analyze only areas of interest.</i>
</div>

---

## 💎 Core Capabilities

<table align="center" width="100%">
  <tr>
    <td width="33%" align="center">
      <h3>🎯 Precision Slicing</h3>
      <p>Eliminates exhaustive grid-slicing by generating semantic heatmaps to isolate high-information background regions.</p>
    </td>
    <td width="33%" align="center">
      <h3>🧠 Multi-Layer Attention</h3>
      <p>Fuses spatial relevance maps from multiple backbone depths (Layer 2, 3, 4) using customized weighted aggregation.</p>
    </td>
    <td width="33%" align="center">
      <h3>⚡ Optimized Compute</h3>
      <p>By running high-fidelity YOLO inference <b>only</b> on retained candidate crops, detector workload is slashed dramatically.</p>
    </td>
  </tr>
</table>

## ⚙️ The Pipeline Architecture

The protocol operates in three distinct, highly optimized stages. Local detection bounding boxes are mapped seamlessly back to the global frame.

<div align="center">
  <img src="https://github.com/DhruvGarg111/The-Searchlight-Protocol/raw/main/Images/Architectural%20Diagram.png" alt="Architecture" width="90%" style="border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.1);">
</div>
<br>

> **1️⃣ STAGE 1: GUIDE (ResNet50 + LayerCAM)**  
> Evaluates the full-scale image to generate semantic heatmaps. Highlights target-rich zones while suppressing empty background noise.

> **2️⃣ STAGE 2: SLICER (Intelligent Activation)**  
> Thresholds the fused heatmap, extracts structural contours, pads candidates, and applies NMS to extract distinct, optimized bounding crops.

> **3️⃣ STAGE 3: DETECTOR (YOLO + Global Fusion)**  
> Ingests the localized crops into YOLO. Output bounding boxes are fused and remapped back to the absolute global coordinate space:  
> `[x1, y1, x2, y2]_global = [x1 + x_c, y1 + y_c, x2 + x_c, y2 + y_c]`

---

## 👁️ Visual Results

Witness the pipeline step-by-step: from semantic attention mapping to smart slicing and the final end-to-end detection output.

| <div align="center">**1. Semantic Attention**</div> | <div align="center">**2. Smart Crop Selection**</div> | <div align="center">**3. Remapped Detections**</div> |
| :---: | :---: | :---: |
| <img src="https://github.com/DhruvGarg111/The-Searchlight-Protocol/raw/main/Images/Output_1.jpg" width="300" style="border-radius: 8px;"> | <img src="https://github.com/DhruvGarg111/The-Searchlight-Protocol/raw/main/Images/Output_2.jpg" width="300" style="border-radius: 8px;"> | <img src="https://github.com/DhruvGarg111/The-Searchlight-Protocol/raw/main/Images/Output_3.jpg" width="300" style="border-radius: 8px;"> |
| *Identifies potential targets* | *Isolates areas of interest* | *High-fidelity detection* |

---

## 💻 Technical Stack

<details>
<summary><b>View Stack Details</b> (Click to expand)</summary>
<br>

- **Frontend Interface:** ⚛️ **React** - Provides an interactive console for input upload, parameter control, and real-time visualization of intermediate outputs.
- **Backend API:** ⚡ **FastAPI** - Handles validation, model lifecycle orchestration, memory-safe inference routing, and JSON response payload generation.
- **Computer Vision:** 👁️ **PyTorch & OpenCV** - Core engines for LayerCAM generation, morphological operations, NMS, and matrix manipulation.
- **Inference Pipeline Flow:**  
  `Upload` ➡️ `LayerCAM guidance` ➡️ `Slicing` ➡️ `YOLO execution` ➡️ `Global fusion` ➡️ `JSON payload`

</details>

---

## 🚀 Getting Started

### Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/DhruvGarg111/The-Searchlight-Protocol.git
cd The-Searchlight-Protocol
```

**2. Boot the API (Backend)**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r webapp/backend/requirements.txt

cd webapp/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Launch the Console (Frontend)**
```bash
# Open a new terminal tab
cd webapp/frontend
npm install
npm run dev
```

---

## ☁️ Zero-Cost Cloud Deployment

<details>
<summary><b>Deploying to Hugging Face + Vercel</b> (Click to expand)</summary>
<br>

**1. Backend (Hugging Face Spaces)**
- Create a new HF Space (`Docker`, `CPU Basic`).
- Push the **full** repository to the Space.
- Set **Space Variables**:
  - `SEARCHLIGHT_PRELOAD_MODELS=true`
  - `SEARCHLIGHT_SERIAL_EXECUTION=true`
  - `SEARCHLIGHT_LOG_LEVEL=INFO`
- Copy your running Space URL (e.g., `https://username-space.hf.space`).

**2. Frontend (Vercel)**
- Import the repo into Vercel and set the **Root Directory** to `webapp/frontend`.
- Add Environment Variable: `VITE_API_BASE_URL` = `<your_hf_space_url>`.
- Deploy and copy your Vercel project URL.

**3. Enable CORS**
- Go back to your Hugging Face Space variables.
- Add `SEARCHLIGHT_ALLOW_ORIGINS` = `<your_vercel_url>`.
- Restart the HF space.

</details>

---

## 🗺️ Roadmap & Evolution

- [x] **v1.0** — Core 3-stage coarse-to-fine pipeline implementation
- [x] **v1.1** — Heatmap-based intelligent slicing with NMS
- [x] **v1.2** — Full-stack web inference console (React + FastAPI)
- [ ] **v2.0** — Temporal consistency for live video streams
- [ ] **v2.1** — Edge device deployment optimizations (TensorRT/ONNX)
- [ ] **v3.0** — Multi-object tracking temporal integration

---

<div align="center">
  <p>Engineered & Designed by <a href="https://github.com/DhruvGarg111"><b>DhruvGarg111</b></a></p>
  <p>
    <a href="https://github.com/DhruvGarg111/The-Searchlight-Protocol/stargazers">
      <img src="https://img.shields.io/github/stars/DhruvGarg111/The-Searchlight-Protocol?style=social" alt="Stars"/>
    </a>
  </p>
  <i>If this repository accelerates your research or projects, please consider dropping a ⭐!</i>
</div>
