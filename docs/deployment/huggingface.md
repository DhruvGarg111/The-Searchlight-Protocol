# Hugging Face Spaces Deployment Guide

Deploying the FastAPI backend to Hugging Face Spaces provides a zero-cost cloud host for model inference.

---

## 🐋 Space Initialization

### 1. Create a New Space
1.  Navigate to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2.  Set the Owner and Name.
3.  Choose **Docker** as the SDK template.
4.  Choose **Blank** under the Docker templates list.
5.  Select **CPU Basic (Free)** (or GPU if you have access grants).
6.  Set Space visibility to Public.

---

## 🛠️ Docker Configuration

Ensure your repository has a `Dockerfile` at the root directory configured for loading the python stack. Below is the standard setup for Python 3.10 and PyTorch:

```dockerfile
FROM python:3.10-slim

WORKDIR /code

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/requirements.txt
COPY webapp/backend/requirements.txt /code/backend_requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/backend_requirements.txt

COPY . /code

# Hugging Face Spaces runs on port 7860 by default
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 🔒 Environment Variable Configuration

Navigate to your Space's **Settings** tab and configure the following variables under the **Variables and Secrets** block:

| Key | Value | Description |
| :--- | :--- | :--- |
| `SEARCHLIGHT_PRELOAD_MODELS` | `true` | Preloads models during container build/startup. |
| `SEARCHLIGHT_SERIAL_EXECUTION` | `true` | Essential for free-tier Spaces to prevent thread collisions. |
| `SEARCHLIGHT_LOG_LEVEL` | `INFO` | Standard INFO logging. |
| `SEARCHLIGHT_ALLOW_ORIGINS` | `https://your-vercel-domain.vercel.app` | Allows your deployed Vercel frontend to bypass CORS limits. |
