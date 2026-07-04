# Changelog

All notable changes to **The Searchlight Protocol** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] — 2026-07-04

### Added
- Complete system documentation inside the `docs/` folder structure, including architecture descriptions, API references, sequence diagrams, deployment guides, and contributor checklists.
- Clear Google-style docstrings added to core modules (`ImageLoader.py`, `LayerCam.py`, `Slicer.py`, `Detector.py`).
- Complete Python docstring coverage for backend routers, service orchestrations, configuration schemas, and app factory modules.
- JSDoc parameter documentation added for React frontend console component modules.

### Changed
- Updated the primary project `README.md` to correctly document the Stage 1 guidance model backbone as **ResNet18** (instead of ResNet50) to match the actual codebase implementation.

---

## [1.1.0] — 2026-02-28

### Added
- Heatmap-based intelligent slicing and padding constraints algorithm.
- Crop-level Non-Maximum Suppression (NMS) to reduce duplicate candidate crops.
- Multi-layer CAM fusion aggregation weighting features.

---

## [1.0.0] — 2026-01-15

### Added
- Core 3-stage coarse-to-fine detection pipeline structure.
- YOLOv8 detector integration interface.
- Global coordinate remapping function.
