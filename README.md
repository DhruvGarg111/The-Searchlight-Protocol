# 🚁 Drone Forensics: The Searchlight Protocol 🔦

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

> **"Finding the needle in the haystack, from 400ft above."**

The **Searchlight Protocol** is an advanced computer vision pipeline designed for `Drone Forensics`. It solves the "small object detection" problem in high-resolution aerial imagery by using a **Saliency-Guided Zoom** approach.

Instead of resizing a 4K/8K image down to 640x640 (losing all detail), this system uses **LayerCAM** heatmaps to identify potential "hotspots", intelligently slices them out, and feeds high-resolution crops to an object detector.

---

## ✨ Features

- **🔍 LayerCAM-Guided ROI**: Uses deep feature activation maps (from ResNet/CNNs) to find areas of interest automatically.
- **🔪 Intelligent Slicing**:
  - Dynamically crops regions based on heatmap intensity.
  - Enforces **minimum crop sizes** to ensure valid detections.
  - Adds **adaptive padding** to provide context around objects.
- **🖼️ High-Res Image Engine**:
  - Custom `DroneImageLoader` handles massive TIFF/JPG files.
  - Smart downsampling for the "Searchlight" pass (LayerCAM) while preserving full resolution for the "Detection" pass (Slicing).
- **🚀 Modular Design**: Separate modules for Detection, Slicing, and Visualization.
- **⚡ GPU Accelerated**: Optimized for CUDA inference.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `The_searchlight_Protocol.ipynb` | 📓 **Main Notebook**. Runs the full pipeline: Load -> Search -> Slice -> Detect. |
| `Detector.py` | 🕵️ **YOLO Wrapper**. Manages the Ultralytics YOLOv8 model instance. |
| `LayerCam.py` | 🗺️ **Attention Engine**. Generates Class Activation Maps (CAM) from the backbone. |
| `Slicer.py` | 🔪 **Smart Slicer**. Converts heatmaps into coordinate crops with logic for padding/sizing. |
| `ImageLoader.py` | 🏗️ **I/O Utility**. Loader for high-res drone imagery with contrast enhancement. |
| `requirements.txt` | 📦 List of required Python libraries. |
| `DPP_00427.JPG` | 📷 **Sample Data**. High-resolution drone imagery for testing. |

---

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/drone-forensics.git
    cd drone-forensics
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Weights** (Auto-handled by Ultralytics):
    - The system defaults to `yolov8m.pt`. It will download automatically on first run.

---

## 💡 Usage

### Running the Searchlight Protocol

1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open **`The_searchlight_Protocol.ipynb`**.
3.  Execute the cells to run the pipeline on your own images.

### Pipeline Logic (Simplified)

```python
from ImageLoader import DroneImageLoader
from LayerCam import LayerCAM
from Slicer import IntelligentSlicer
from Detector import YOLODetector

# 1. Initialize
loader = DroneImageLoader(max_dim=2048)
slicer = IntelligentSlicer(padding_factor=0.2, info_threshold=0.4, min_crop_size=640)
detector = YOLODetector(model_variant="m")

# 2. Load & Search
original, tensor, _, _ = loader.load("DPP_00427.JPG")
# (Assume 'model' and 'target_layer' are defined from a backbone like ResNet)
heatmap = LayerCAM(model, target_layer).generate(tensor)

# 3. Slice & Detect
crops, _, _ = slicer.slice(original, heatmap)
for crop_data in crops:
    # Detection on the high-res crop
    results = detector.model(crop_data['image']) 
    print(f"Detected {len(results[0].boxes)} objects in crop {crop_data['id']}")
```

---

## ⚙️ Configuration Parameters

You can tune the system in `Slicer.py` instantiation:

- **`info_threshold`** (0.0 - 1.0): How "hot" a region must be to get sliced. Lower = more crops, Higher = generally only high-confidence areas.
- **`padding_factor`** (Float): How much extra context to add around the heatmap blob.
- **`min_crop_size`** (Pixels): Ensures crops are large enough for the YOLO model (e.g., 640px).

---

## 🛠️ Status

🚧 **Active Development**
- [x] LayerCAM Integration
- [x] Intelligent Slicing
- [x] YOLOv8 Support
- [ ] Real-time Video Support
- [ ] API Deployment

---

*Part of the Deepmind Forensics Initiative.*
