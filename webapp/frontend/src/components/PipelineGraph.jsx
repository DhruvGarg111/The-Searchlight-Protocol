import { memo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Binary, Crop, ScanSearch } from "lucide-react";
import CornerFrame from "./ui/CornerFrame";
import MicroLabel from "./ui/MicroLabel";

const STAGES = [
  {
    code: "GUIDE",
    title: "ResNet50 + LayerCAM",
    summary: "Generates semantic attention heatmaps.",
    detail: "Semantic activations provide priors for high-value regions before any detector pass.",
    icon: Binary,
    accentClass: "stage-guide",
  },
  {
    code: "SLICER",
    title: "Heatmap-Based Contouring",
    summary: "Contours high-activation regions.",
    detail: "Activation contours are padded with context to retain geometry around small aerial targets.",
    icon: Crop,
    accentClass: "stage-slicer",
  },
  {
    code: "DETECTOR",
    title: "YOLOv8 + Fusion",
    summary: "Runs inference on selected crops.",
    detail: "Crop-local predictions are projected to global coordinates and fused for mission-ready output.",
    icon: ScanSearch,
    accentClass: "stage-detector",
  },
];

function PipelineGraph() {
  const [hoveredIndex, setHoveredIndex] = useState(-1);

  return (
    <CornerFrame className="panel-shell rounded-sm px-5 py-6 md:px-8 md:py-7" markerTopLeft="SEC:02" markerBottomRight="PIPE-B">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <MicroLabel className="mb-2">L3 FEATURE EXTRACTION</MicroLabel>
          <p className="section-label">Pipeline Section</p>
          <h2 className="mt-1 font-display text-xl tracking-[0.1em] text-slate-100 md:text-2xl">STAGE ARCHITECTURE</h2>
        </div>
        <p className="body-copy max-w-lg text-sm">
          Each stage is independently tuned and contributes to end-to-end tactical localization quality.
        </p>
      </div>

      <div className={`pipeline-connector ${hoveredIndex >= 0 ? "pipeline-connector-active" : ""}`} aria-hidden="true" />

      <div className="grid gap-4 md:grid-cols-3">
        {STAGES.map((stage, index) => {
          const Icon = stage.icon;
          const active = hoveredIndex === index;
          const dimmed = hoveredIndex >= 0 && !active;

          return (
            <motion.article
              key={stage.code}
              className={`stage-card ${stage.accentClass} ${active ? "stage-card-active" : ""} ${dimmed ? "stage-card-dim" : ""}`}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(-1)}
              onFocus={() => setHoveredIndex(index)}
              onBlur={() => setHoveredIndex(-1)}
              whileHover={{ y: -4 }}
              transition={{ duration: 0.2 }}
              tabIndex={0}
            >
              <div className="mb-2 flex items-center justify-between">
                <span className="font-display text-[11px] tracking-[0.18em] text-slate-100">[ {stage.code} ]</span>
                <Icon className="h-4 w-4" />
              </div>

              <h3 className="font-display text-sm uppercase tracking-[0.08em] text-slate-100">{stage.title}</h3>
              <p className="mt-2 body-copy text-sm">{stage.summary}</p>

              <AnimatePresence initial={false}>
                {active ? (
                  <motion.p
                    className="mt-3 border-t border-current/30 pt-2 text-xs text-slate-300"
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 3 }}
                    transition={{ duration: 0.2 }}
                  >
                    {stage.detail}
                  </motion.p>
                ) : null}
              </AnimatePresence>
            </motion.article>
          );
        })}
      </div>
    </CornerFrame>
  );
}

export default memo(PipelineGraph);
