import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn, spawnSync } from "node:child_process";

const HOST = "127.0.0.1";
const APP_PORT = Number.parseInt(process.env.SEARCHLIGHT_SMOKE_APP_PORT ?? "4173", 10);
const DEBUG_PORT = Number.parseInt(process.env.SEARCHLIGHT_SMOKE_DEBUG_PORT ?? "9222", 10);
const APP_URL = `http://${HOST}:${APP_PORT}`;
const TIMEOUT_MS = 15_000;

const PNG_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=";

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function browserCandidates() {
  const envPath = process.env.SEARCHLIGHT_BROWSER_PATH;
  if (envPath && existsSync(envPath)) {
    return [envPath];
  }

  if (process.platform === "win32") {
    return [
      "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
      "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
      "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
      "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    ].filter((candidate) => existsSync(candidate));
  }

  return ["google-chrome", "chromium", "chromium-browser", "microsoft-edge"].filter((candidate) => {
    const probe = spawnSync(candidate, ["--version"], { stdio: "ignore" });
    return probe.status === 0;
  });
}

function startPreview() {
  const viteBin = join(process.cwd(), "node_modules", "vite", "bin", "vite.js");
  return spawn(
    process.execPath,
    [viteBin, "preview", "--host", HOST, "--port", String(APP_PORT), "--strictPort"],
    {
      cwd: process.cwd(),
      stdio: "pipe",
      windowsHide: true,
    },
  );
}

function startBrowser(browserPath, userDataDir, debugPort) {
  return spawn(
    browserPath,
    [
      "--headless=new",
      "--disable-gpu",
      "--disable-background-networking",
      "--disable-dev-shm-usage",
      "--no-first-run",
      "--no-default-browser-check",
      `--remote-debugging-address=${HOST}`,
      `--remote-debugging-port=${debugPort}`,
      `--user-data-dir=${userDataDir}`,
      "about:blank",
    ],
    {
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    },
  );
}

async function waitForHttp(url, label) {
  const deadline = Date.now() + TIMEOUT_MS;
  let lastError = null;

  while (Date.now() < deadline) {
    const controller = new AbortController();
    const abortTimer = setTimeout(() => controller.abort(), 1_000);
    try {
      const response = await fetch(url, { signal: controller.signal });
      if (response.ok) {
        return response;
      }
    } catch (error) {
      lastError = error;
    } finally {
      clearTimeout(abortTimer);
    }
    await sleep(200);
  }

  throw new Error(`Timed out waiting for ${label}${lastError ? `: ${lastError.message}` : ""}`);
}

class DevToolsPage {
  constructor(webSocketUrl) {
    this.nextId = 1;
    this.pending = new Map();
    this.ws = new WebSocket(webSocketUrl);
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });

    this.ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (!message.id) {
        return;
      }

      const pending = this.pending.get(message.id);
      if (!pending) {
        return;
      }

      this.pending.delete(message.id);
      if (message.error) {
        pending.reject(new Error(message.error.message));
      } else {
        pending.resolve(message.result);
      }
    });
  }

  send(method, params = {}) {
    const id = this.nextId;
    this.nextId += 1;
    const payload = JSON.stringify({ id, method, params });

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(payload);
    });
  }

  close() {
    this.ws.close();
  }
}

async function evaluate(page, expression) {
  const result = await page.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  });

  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text ?? "Browser evaluation failed");
  }

  return result.result?.value;
}

async function waitForExpression(page, expression, label) {
  const deadline = Date.now() + TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (await evaluate(page, expression)) {
      return;
    }
    await sleep(100);
  }
  throw new Error(`Timed out waiting for ${label}`);
}

function mockApiScript() {
  const payload = {
    meta: {
      device: "cpu",
      original_size: [100, 80],
      scale_factor: 1,
      torch_version: "smoke",
      cuda_available: false,
    },
    settings: {
      padding_factor: 0.4,
      heatmap_threshold: 0.4,
      yolo_confidence: 0.3,
      min_crop_size: 120,
      nms_iou_threshold: 0.2,
      yolo_iou_threshold: 0.6,
      response_profile: "display",
      enable_global_nms: false,
      global_nms_iou_threshold: 0.5,
    },
    counts: {
      pre_nms_crops: 2,
      post_nms_crops: 2,
      detections: 1,
    },
    research: {
      run_id: "smoke1234",
      started_at_utc: "2026-07-06T00:00:00+00:00",
      experiment: "ui-smoke",
      objective: "frontend smoke",
      model_stack: {
        guide_backbone: "ResNet18 (ImageNet1K_V1)",
        guide_target_layers: ["layer2[-1]", "layer3[-1]", "layer4[-1]"],
        cam_fusion_weights: { layer2: 0.7, layer3: 0.9, layer4: 1.0 },
        detector: "YOLOv8-n",
      },
      detector_inference: {
        conf: 0.3,
        iou: 0.6,
        augment: true,
        agnostic_nms: true,
      },
      crop_selection: {
        padding_factor: 0.4,
        heatmap_threshold: 0.4,
        min_crop_size: 120,
        nms_iou_threshold: 0.2,
        pre_nms_count: 2,
        post_nms_count: 2,
      },
      timings_ms: { total_pipeline: 1200 },
      detection_meta: { batched: true },
    },
    outputs: {
      original_image: PNG_DATA_URL,
      weighted_fusion_cam: PNG_DATA_URL,
      post_nms_boundaries: PNG_DATA_URL,
      final_detections: PNG_DATA_URL,
    },
    crops: [
      { id: 1, score: 0.91, bbox: [0, 0, 20, 20], image: PNG_DATA_URL },
      { id: 2, score: 0.82, bbox: [30, 20, 20, 20], image: PNG_DATA_URL },
    ],
    detections: [
      {
        crop_id: 1,
        class: "target",
        confidence: 0.91,
        global_bbox: [10, 12, 30, 36],
      },
    ],
    input_filename: "smoke.png",
  };

  return `
    window.__searchlightSmokeRequests = [];
    window.fetch = async (url, options = {}) => {
      const fields = {};
      if (options.body && typeof options.body.forEach === "function") {
        options.body.forEach((value, key) => {
          fields[key] = value instanceof File ? value.name : String(value);
        });
      }
      window.__searchlightSmokeRequests.push({ url: String(url), fields });
      return new Response(${JSON.stringify(JSON.stringify(payload))}, {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };
  `;
}

async function runSmoke(page) {
  await page.send("Runtime.enable");
  await page.send("Page.enable");
  await page.send("Page.addScriptToEvaluateOnNewDocument", { source: mockApiScript() });
  await page.send("Page.navigate", { url: APP_URL });

  await waitForExpression(
    page,
    "document.body && document.body.textContent.includes('TACTICAL CONTROL PANEL')",
    "initial app render",
  );

  await evaluate(
    page,
    `
      (() => {
        const input = document.querySelector('input[type="file"]');
        if (!input) return false;
        const file = new File(['smoke'], 'smoke.png', { type: 'image/png' });
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()
    `,
  );

  await waitForExpression(page, "document.body.textContent.includes('smoke.png')", "file selection");
  await evaluate(
    page,
    `
      (() => {
        const button = [...document.querySelectorAll('button')]
          .find((item) => item.textContent.includes('EXECUTE PIPELINE'));
        if (!button || button.disabled) return false;
        button.click();
        return true;
      })()
    `,
  );

  await waitForExpression(
    page,
    "window.__searchlightSmokeRequests.length === 1 && window.__searchlightSmokeRequests[0].fields.response_profile === 'display'",
    "mocked display-profile API call",
  );
  await waitForExpression(page, "document.body.textContent.includes('FRAME: SMOKE123')", "result render");

  await evaluate(
    page,
    `
      (() => {
        const button = [...document.querySelectorAll('button')]
          .find((item) => item.textContent.trim() === 'CROPS');
        if (!button) return false;
        button.click();
        return true;
      })()
    `,
  );
  await waitForExpression(page, "document.body.textContent.includes('CROP 1')", "crop sample render");

  await evaluate(
    page,
    `
      (() => {
        const rawButton = [...document.querySelectorAll('button')]
          .find((item) => item.textContent.trim() === 'RAW');
        if (!rawButton) return false;
        rawButton.click();
        return true;
      })()
    `,
  );
  await waitForExpression(
    page,
    "[...document.querySelectorAll('button')].some((item) => item.textContent.includes('OVERLAY') && !item.disabled)",
    "enabled overlay control",
  );
  await evaluate(
    page,
    `
      (() => {
        const overlayButton = [...document.querySelectorAll('button')]
          .find((item) => item.textContent.includes('OVERLAY'));
        if (!overlayButton || overlayButton.disabled) return false;
        overlayButton.click();
        return true;
      })()
    `,
  );
  await waitForExpression(page, "document.body.textContent.includes('OVERLAY ON')", "detection overlay toggle");
  await waitForExpression(
    page,
    "document.body.textContent.includes('Detections Found') && document.body.textContent.includes('Crops Generated')",
    "metrics panel render",
  );
}

async function main() {
  if (typeof WebSocket !== "function") {
    throw new Error("This smoke check requires a Node runtime with a built-in WebSocket implementation.");
  }

  const candidates = browserCandidates();
  if (!candidates.length) {
    throw new Error("Could not find Chrome or Edge. Set SEARCHLIGHT_BROWSER_PATH to run the UI smoke check.");
  }

  const preview = startPreview();
  let browser = null;
  let page = null;
  let userDataDir = null;
  let debugUrl = null;
  const browserErrors = [];

  try {
    console.log(`Starting Vite preview on ${APP_URL}`);
    await waitForHttp(APP_URL, "Vite preview server");

    for (const [index, browserPath] of candidates.entries()) {
      const debugPort = DEBUG_PORT + index;
      debugUrl = `http://${HOST}:${debugPort}`;
      userDataDir = mkdtempSync(join(tmpdir(), "searchlight-browser-"));
      console.log(`Starting browser for smoke check: ${browserPath}`);
      browser = startBrowser(browserPath, userDataDir, debugPort);
      let stderr = "";
      browser.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });

      try {
        await waitForHttp(`${debugUrl}/json/version`, `browser debugging endpoint (${browserPath})`);
        break;
      } catch (error) {
        browserErrors.push(`${browserPath}: ${error.message}${stderr ? `\n${stderr.trim()}` : ""}`);
        browser.kill();
        rmSync(userDataDir, { recursive: true, force: true });
        browser = null;
        userDataDir = null;
        debugUrl = null;
      }
    }

    if (!browser || !debugUrl) {
      throw new Error(`Could not start a headless browser:\n${browserErrors.join("\n")}`);
    }

    const targets = await (await fetch(`${debugUrl}/json/list`)).json();
    const target = targets.find((item) => item.type === "page");
    if (!target?.webSocketDebuggerUrl) {
      throw new Error("Could not find a debuggable browser page.");
    }

    page = new DevToolsPage(target.webSocketDebuggerUrl);
    await page.open();
    await runSmoke(page);
    console.log("Frontend UI smoke check passed.");
  } finally {
    page?.close();
    browser?.kill();
    preview.kill();
    await sleep(500);
    if (userDataDir) {
      try {
        rmSync(userDataDir, { recursive: true, force: true });
      } catch (error) {
        console.warn(`Could not remove temporary browser profile ${userDataDir}: ${error.message}`);
      }
    }
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
