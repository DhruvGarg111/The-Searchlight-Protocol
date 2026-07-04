# Local Installation and Deployment Guide

This guide walks you through setting up and running **The Searchlight Protocol** on your local machine.

---

## 📋 System Prerequisites

*   **Operating System**: Linux, macOS, or Windows (via PowerShell/WSL).
*   **Python**: Python 3.10 or 3.11 (3.12 works but check PyTorch/CUDA wheels compatibility).
*   **Node.js**: Node 18+ (with npm).
*   **Hardware (Optional)**: NVIDIA GPU with CUDA support for accelerated model inference.

---

## 🛠️ Step-by-Step Local Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/DhruvGarg111/The-Searchlight-Protocol.git
cd The-Searchlight-Protocol
```

### Step 2: Configure and Boot the Backend API
1.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    ```
2.  **Activate the Virtual Environment**:
    *   *Linux/macOS*: `source .venv/bin/activate`
    *   *Windows (PowerShell)*: `.venv\Scripts\Activate.ps1`
3.  **Install Base & Web Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install -r webapp/backend/requirements.txt
    ```
4.  **Launch the Uvicorn Server**:
    ```bash
    cd webapp/backend
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    ```
    Verify it works by navigating to `http://127.0.0.1:8000/docs` in your browser. This will open the FastAPI Swagger interactive documentation interface.

### Step 3: Run the Frontend Console
1.  **Open a New Terminal Window** (leave the backend running).
2.  **Navigate to Frontend Directory**:
    ```bash
    cd webapp/frontend
    ```
3.  **Install Node Modules**:
    ```bash
    npm install
    ```
4.  **Boot the Vite Dev Server**:
    ```bash
    npm run dev
    ```
5.  **Open the Web Dashboard Console**:
    Navigate to `http://localhost:5173` in your browser to interact with the console interface.

---

## ⚙️ GPU Acceleration Verification

To verify that the backend is utilizing your local NVIDIA GPU (if available):
1.  Check the server console output during start-up. You should see a message:
    `Pipeline models warmed up on cuda in ... ms`.
2.  Alternatively, hit the health check endpoint:
    `curl http://127.0.0.1:8000/api/health`
    The response will indicate whether `cuda` or `cpu` is current model execution target.
