# Vercel Frontend Deployment Guide

Deploying the React frontend console to **Vercel** provides a high-availability serverless CDN hosting option.

---

## 🚀 Setup Steps

### Step 1: Import Repository
1.  Log in to your [Vercel Dashboard](https://vercel.com).
2.  Click **Add New...** and select **Project**.
3.  Import the target Git repository containing the Searchlight Protocol source code.

### Step 2: Build Options Configuration
Modify the Vercel default settings to point to the nested frontend directory structure:

1.  **Root Directory**: Click Edit and select `webapp/frontend`.
2.  **Build Command**: Verify it defaults to `npm run build` or `vite build`.
3.  **Output Directory**: Verify it is set to `dist`.

---

## 🔒 Environment Variable Configuration

Under the **Environment Variables** section, configure the key linking your frontend requests to the deployed backend server:

| Key | Value | Description |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `https://username-space.hf.space` | The target URL of your Hugging Face Space (or alternative backend host). |

*   *Note*: Ensure there is **no trailing slash** at the end of the URL (e.g. use `/api` resolution mappings in your fetch routines).

Click **Deploy** to publish the console. Once the build finishes, copy the generated `.vercel.app` URL and add it back to the backend CORS allowed origins list (`SEARCHLIGHT_ALLOW_ORIGINS` env var).
