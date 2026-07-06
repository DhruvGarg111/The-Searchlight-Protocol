# Searchlight Optimization Plan

This plan is scoped to optimizing the current project without silently removing product capabilities, API fields, visual outputs, model options, or deployment paths. Implementation should proceed only with baseline measurements and regression gates in place.

## Current Baseline

- Project shape: root Python modules implement the CV pipeline; `webapp/backend` exposes it through FastAPI; `webapp/frontend` is a Vite/React console.
- Working tree was clean when inspected before this plan was added.
- Available verification gates:
  - `python -m compileall -q Detector.py ImageLoader.py LayerCam.py Slicer.py webapp\backend` passed.
  - `npm.cmd run build` passed from `webapp/frontend`.
- Current frontend production build:
  - `dist/index.html`: 0.41 kB, gzip 0.29 kB.
  - `dist/assets/index-BX5mykDB.css`: 26.06 kB, gzip 6.00 kB.
  - `dist/assets/index-BUT6KuBq.js`: 287.15 kB, gzip 92.91 kB.
- No committed test suite was found.
- Largest relevant non-`node_modules` files:
  - `webapp/backend/yolov9m.pt`: 39.09 MB.
  - `webapp/backend/yolov8n.pt`: 6.25 MB.
- Existing docs already identify the main backend bottlenecks: redundant CAM passes, sequential crop inference, massive base64 payloads, missing global NMS, hook collision risk when request serialization is disabled, and temp-file leakage.

## Non-Negotiable Constraints

- Preserve the existing `/api/run-pipeline` response contract by default: `meta`, `settings`, `counts`, `research`, `outputs`, `crops`, `detections`, and `input_filename` must remain available unless a caller explicitly requests a lighter response mode.
- Preserve frontend behavior: upload, sliders, execution, result tabs, crop previews, detection overlays, and metrics must still work.
- Preserve model configurability through the current environment variables, including YOLO version, variant, and model path selection.
- Do not remove `yolov9m.pt` or other model assets without an explicit product decision, because the backend configuration supports alternate model choices.
- Do not lower CAM input resolution, disable YOLO augmentation, drop visual outputs, or change thresholds as a hidden optimization.
- Keep CPU and CUDA support.

## Success Criteria

- Warm end-to-end pipeline latency improves for representative images, or a targeted stage improves by at least 10 percent with no total latency regression.
- Peak memory and response payload size are measured before and after any payload or encoding change.
- Detection quality is not degraded on fixed fixtures: detection classes, confidence ranges, global boxes, crop counts, and visual outputs must match baseline within an explicit tolerance.
- Frontend build size does not regress without justification.
- Any behavior-changing optimization is feature flagged or request-scoped until benchmark and quality gates prove it safe as a default.

## Phase 0: Measurement And Safety Harness

1. Add a lightweight benchmark runner, for example `benchmarks/run_pipeline_benchmark.py`, that calls `SearchlightPipelineService` directly and records:
   - input image name, dimensions, device, torch version, CUDA availability;
   - `research.timings_ms` stage timings;
   - pre-NMS crop count, post-NMS crop count, detection count;
   - serialized response byte size;
   - output key list and crop image count.
2. Add representative fixture images or support an external fixture directory via an environment variable so large sample data does not need to be committed.
3. Add initial tests before risky optimization:
   - slicer crop bounds and minimum-size behavior;
   - crop NMS determinism;
   - validation failures for invalid image content and oversized uploads;
   - API response contract shape using a mocked or stubbed pipeline service;
   - optional model-backed integration benchmark for local machines with weights available.
4. Store baseline benchmark JSON artifacts outside committed source by default, with an option to keep curated summaries in `benchmarks/baselines/`.

## Phase 1: Backend Runtime Optimizations

1. Refactor multi-layer CAM into one forward/backward pass.
   - Current risk: `MultiLayerCAM.generate_combined_cam` calls one `LayerCAM.generate` per target layer, and each call performs its own model forward/backward pass.
   - Plan: attach hooks for all target layers at once, run one forward/backward pass, compute each layer heatmap, then compute the weighted fusion using the existing `CAM_FUSION_WEIGHTS`.
   - Verification: unit-test with a small fake CNN to prove one model pass, output shapes, normalized ranges, and unchanged fusion weights.
2. Batch YOLO inference across retained crops.
   - Current risk: `_run_yolo` loops through crops and calls `detector.predict` once per crop.
   - Plan: pass a list of crop images to YOLO in one call, then map each result back to its crop id and global offset.
   - Preserve `conf`, `iou`, `augment=True`, `agnostic_nms=True`, and verbose behavior.
   - Verification: compare detections from sequential and batched paths on the same crops within coordinate/confidence tolerance.
3. Add global NMS for remapped detections.
   - Current risk: overlapping crops can produce duplicate global boxes.
   - Plan: run a final `torchvision.ops.nms` on global coordinates after crop-local remapping.
   - Keep this behind a config or request flag during validation, then enable only when fixture comparisons show duplicate reduction without losing legitimate neighboring objects.
4. Avoid temporary file round trips for uploads.
   - Current risk: `run_from_bytes` writes a temp file, then `DroneImageLoader.load` reopens it after upload validation already decoded the image once.
   - Plan: add a byte/PIL loading path while keeping `run_from_path` for compatibility.
   - Verification: compare the tensor shape, original array, scale factor, and pipeline output against the existing file path flow.
5. Make CAM input max dimension configurable while preserving the current default.
   - Current code uses `RESNET_INPUT_MAX_DIM = 1800`; docs mention a fixed max dimension elsewhere.
   - Plan: add a `SEARCHLIGHT_RESNET_INPUT_MAX_DIM` setting with default `1800`, then update docs.
   - Do not change the default until benchmark and quality evidence supports it.

## Phase 2: Payload And Memory Optimizations

1. Add explicit response profiles.
   - `full`: current behavior, all outputs and crop images.
   - `display`: all metadata and detections, but visual images resized to frontend display needs and crop images limited to what the UI renders.
   - `metadata`: no image payloads, intended for API clients and benchmark automation.
2. Keep `full` as the default initially to preserve the API contract.
3. Let the frontend request `display` once visual parity is verified, because it currently renders only four main output tabs and the first six crop samples.
4. Add configurable image encoding controls:
   - output max dimension for display images;
   - output format, preserving PNG by default;
   - optional JPEG/WebP only for display mode if quality is acceptable.
5. Verification:
   - response schema remains backward compatible in `full`;
   - frontend visual tabs still render;
   - payload bytes drop materially in `display` mode, especially on 4K inputs;
   - no increased backend total time from resizing or encoding.

## Phase 3: Frontend Bundle And Runtime Optimizations

1. Add a build-size budget check using the current baseline as the starting budget: JS 287.15 kB raw / 92.91 kB gzip, CSS 26.06 kB raw / 6.00 kB gzip.
2. Audit unused frontend files before removal. `src/App.css` is not imported by `src/main.jsx` or `src/App.jsx`, but deletion should still be verified with build and visual smoke checks.
3. Keep `framer-motion` unless a replacement preserves the existing interactive behavior. Removing animation is a feature decision, not a safe optimization.
4. Consider route- or component-level lazy loading only if measured first paint or initial JS budget needs it. This can improve initial load without reducing total feature surface.
5. Move external font loading out of `@import` and into `index.html` with preconnect/preload, or self-host fonts, after measuring network impact in deployed environments.

## Phase 4: Repository And Deployment Footprint

1. Do not delete model weights as a first-pass optimization.
2. If clone size or deployment artifact size becomes the priority, evaluate moving optional weights such as `yolov9m.pt` to Git LFS, release assets, or a documented download/cache step.
3. Preserve offline/local startup for the default model unless the user explicitly accepts a network-dependent model fetch.
4. Keep ignored local artifacts out of commits: `webapp/frontend/node_modules`, `webapp/frontend/dist`, `__pycache__`, and the ignored root `package-lock.json`.

## Recommended Implementation Order

1. Phase 0 benchmark and regression harness.
2. Single-pass multi-layer CAM.
3. Batched YOLO crop inference.
4. Optional global NMS validation and rollout.
5. Upload byte-path loading to remove temp-file overhead.
6. Response profiles and display-mode frontend payload reduction.
7. Frontend bundle/runtime cleanup.
8. Repository weight/deployment artifact decisions.

## Required Gates For Each Optimization PR

- `python -m compileall -q Detector.py ImageLoader.py LayerCam.py Slicer.py webapp\backend`
- `python -m pytest -q` after tests are introduced.
- `npm.cmd run build` from `webapp/frontend`.
- Benchmark JSON before/after for at least one warm run.
- API contract comparison for default `full` response mode.
- Frontend smoke check confirming upload controls, result tabs, crop samples, detection overlay, and metrics still work.
