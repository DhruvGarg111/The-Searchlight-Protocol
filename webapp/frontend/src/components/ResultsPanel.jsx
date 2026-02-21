import { memo, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Crop, Crosshair, Flame, Image as ImageIcon, Layers3 } from "lucide-react";
import CornerFrame from "./ui/CornerFrame";
import MicroLabel from "./ui/MicroLabel";

const RESULT_TABS = [
  { id: "raw", label: "RAW", icon: ImageIcon, outputKey: "original_image", accent: "tab-accent-guide" },
  { id: "heatmap", label: "HEATMAP", icon: Flame, outputKey: "weighted_fusion_cam", accent: "tab-accent-slicer" },
  { id: "crops", label: "CROPS", icon: Crop, outputKey: "post_nms_boundaries", accent: "tab-accent-slicer" },
  { id: "detections", label: "DETECTIONS", icon: Crosshair, outputKey: "final_detections", accent: "tab-accent-detector" },
];

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function DetectionOverlay({ detections, originalSize }) {
  const imageHeight = toNumber(originalSize?.[0]);
  const imageWidth = toNumber(originalSize?.[1]);

  if (!imageHeight || !imageWidth || !detections.length) {
    return null;
  }

  return (
    <div className="pointer-events-none absolute inset-0">
      {detections.map((det, index) => {
        const [x1, y1, x2, y2] = (det.global_bbox ?? []).map(toNumber);
        const left = (x1 / imageWidth) * 100;
        const top = (y1 / imageHeight) * 100;
        const width = ((x2 - x1) / imageWidth) * 100;
        const height = ((y2 - y1) / imageHeight) * 100;

        if (!Number.isFinite(left + top + width + height)) {
          return null;
        }

        return (
          <motion.div
            key={`${det.crop_id}-${det.class}-${index}`}
            className="absolute border border-accent-detector/90"
            style={{
              left: `${Math.max(0, left)}%`,
              top: `${Math.max(0, top)}%`,
              width: `${Math.max(0.8, width)}%`,
              height: `${Math.max(0.8, height)}%`,
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2, delay: index * 0.03 }}
          >
            <span className="absolute -top-5 left-0 bg-slate-950/90 px-1 py-0.5 font-mono text-[10px] text-accent-detector">
              {det.class} {toNumber(det.confidence).toFixed(2)}
            </span>
          </motion.div>
        );
      })}
    </div>
  );
}

function ResultsPanel({ result, loading }) {
  const [activeTab, setActiveTab] = useState("raw");
  const [overlayEnabled, setOverlayEnabled] = useState(true);

  const tabMeta = RESULT_TABS.find((tab) => tab.id === activeTab) ?? RESULT_TABS[0];
  const imageSrc = result?.outputs?.[tabMeta.outputKey] ?? null;
  const detections = result?.detections ?? [];
  const cropSamples = useMemo(() => (result?.crops ?? []).slice(0, 6), [result]);

  const runShort = result?.research?.run_id?.slice(0, 8)?.toUpperCase() ?? "N/A";
  const runtimeMs = toNumber(result?.research?.timings_ms?.total_pipeline);

  const canOverlay = (activeTab === "raw" || activeTab === "detections") && detections.length > 0;

  return (
    <CornerFrame className="panel-shell rounded-sm px-5 py-6 md:px-8 md:py-7" markerTopLeft="SEC:04" markerBottomRight="VIS-D">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <MicroLabel className="mb-2">GLOBAL COORDINATE REMAP</MicroLabel>
          <p className="section-label">Results Panel</p>
          <h2 className="mt-1 font-display text-xl tracking-[0.1em] text-slate-100 md:text-2xl">VISUAL OUTPUTS</h2>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="results-runtime-badge">
            <span>FRAME: {runShort}</span>
            <span>{runtimeMs ? `${runtimeMs.toFixed(1)}ms` : "RUNTIME:N/A"}</span>
          </div>

          <button
            type="button"
            className="inline-flex items-center gap-2 border border-slate-600/70 bg-slate-950/75 px-3 py-2 text-xs text-slate-300 transition duration-200 hover:border-slate-500"
            onClick={() => setOverlayEnabled((prev) => !prev)}
            disabled={!canOverlay}
          >
            <Layers3 className="h-3.5 w-3.5 text-accent-slicer" />
            OVERLAY {overlayEnabled ? "ON" : "OFF"}
          </button>
        </div>
      </div>

      <div className="mb-4 grid gap-2 md:grid-cols-4">
        {RESULT_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = tab.id === activeTab;

          return (
            <button
              key={tab.id}
              type="button"
              className={`result-tab ${tab.accent} ${isActive ? "result-tab-active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon className="h-3.5 w-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {!result ? (
        <div className="border border-slate-700/80 bg-slate-950/60 p-8 text-center text-sm text-slate-400">
          {loading ? "Pipeline execution in progress." : "Awaiting inference run."}
        </div>
      ) : (
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.22 }}
            className="space-y-4"
          >
            <div className="relative aspect-[16/10] overflow-hidden border border-slate-700/80 bg-slate-950/75">
              {imageSrc ? (
                <img src={imageSrc} alt={tabMeta.label} className="h-full w-full object-contain" />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">No output for this tab.</div>
              )}

              {overlayEnabled && canOverlay ? (
                <DetectionOverlay detections={detections} originalSize={result?.meta?.original_size} />
              ) : null}
            </div>

            {activeTab === "crops" ? (
              cropSamples.length ? (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {cropSamples.map((crop) => (
                    <div key={crop.id} className="border border-slate-700/80 bg-slate-950/60 p-2">
                      <img src={crop.image} alt={`Crop ${crop.id}`} className="h-24 w-full border border-slate-700/80 object-cover" />
                      <div className="mt-2 flex items-center justify-between text-xs">
                        <span className="font-display tracking-[0.12em] text-slate-100">CROP {crop.id}</span>
                        <span className="font-mono text-slate-300">{toNumber(crop.score).toFixed(3)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-400">No retained crops after NMS.</p>
              )
            ) : null}
          </motion.div>
        </AnimatePresence>
      )}
    </CornerFrame>
  );
}

export default memo(ResultsPanel);
