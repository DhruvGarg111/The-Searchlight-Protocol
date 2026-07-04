# The Searchlight Protocol Documentation

Welcome to the official developer and researcher documentation for **The Searchlight Protocol**. This documentation covers the system architecture, mathematical formulations, component APIs, deployment guides, and contributing workflows.

## 🗺️ Navigation Map

| Section | Description | Target Audience |
| :--- | :--- | :--- |
| [**Architecture Overview**](architecture/overview.md) | High-level overview of the dual-system codebase. | Everyone |
| [**The Detection Pipeline**](architecture/pipeline.md) | Deep dive into the 3-stage coarse-to-fine detection flow. | Researchers & CV Engineers |
| [**Data Flow & Sequence**](architecture/data_flow.md) | Visual representation of web request-response lifecycles. | Backend & Integration Engineers |
| [**Python Module APIs**](modules/) | Reference for `ImageLoader`, `LayerCam`, `Slicer`, and `Detector`. | Backend Developers |
| [**FastAPI Backend Guide**](backend/overview.md) | REST API structure, routers, settings, and services. | Backend Developers |
| [**React Frontend Console**](frontend/overview.md) | User interface component definitions and state flow. | Frontend Developers |
| [**Local Deployment**](deployment/local.md) | Getting the pipeline and console running on your own machine. | Everyone |
| [**Cloud Deployment**](deployment/huggingface.md) | Guides for Hugging Face Spaces and Vercel setup. | DevOps Engineers |
| [**Contributing Guide**](contributing/CONTRIBUTING.md) | Code styling, linting, and development workflows. | Contributors |
| [**Known Issues**](contributing/known_issues.md) | Technical debt, performance bottlenecks, and active issues. | Core Developers |

---

## 🛠️ System Overview

The Searchlight Protocol repository hosts two independent subsystems under a single repository structure:

1. **Aerial Forensics Pipeline**: An optimized, coarse-to-fine computer vision inference pipeline that uses semantic localization maps to restrict target detection (using YOLO) to relevant candidate slices. This allows for efficient execution on extremely high-resolution aerial frames without exhaustive grid-slicing.
2. **DeSci NFT Indexing Protocol**: A conceptual blockchain-based scientific data indexing protocol documented in separate notebook instances.

This documentation suite focuses primarily on the **Aerial Forensics Pipeline** and its accompanying full-stack web application.
