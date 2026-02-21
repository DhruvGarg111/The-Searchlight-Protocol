import { memo, useMemo, useRef, useState } from "react";
import { Cpu, SlidersHorizontal, Upload } from "lucide-react";
import CornerFrame from "./ui/CornerFrame";
import MicroLabel from "./ui/MicroLabel";

const PARAM_GROUPS = [
  {
    title: "GUIDE PARAMETERS",
    values: [
      {
        key: "heatmap_threshold",
        label: "Heatmap Threshold",
        min: 0,
        max: 1,
        step: 0.01,
        precision: 2,
        accent: "#22d3ee",
      },
      {
        key: "padding_factor",
        label: "Padding Factor",
        min: 0,
        max: 1,
        step: 0.01,
        precision: 2,
        accent: "#7c5cff",
      },
      {
        key: "max_layercam_dim",
        label: "Max Image Dimension",
        min: 1024,
        max: 6144,
        step: 256,
        precision: 0,
        accent: "#7c5cff",
      },
    ],
  },
  {
    title: "DETECTOR PARAMETERS",
    values: [
      {
        key: "yolo_confidence",
        label: "YOLO Confidence",
        min: 0.05,
        max: 1,
        step: 0.01,
        precision: 2,
        accent: "#f5bd5a",
      },
      {
        key: "nms_iou_threshold",
        label: "NMS IoU Threshold",
        min: 0.05,
        max: 1,
        step: 0.01,
        precision: 2,
        accent: "#f5bd5a",
      },
    ],
  },
];

function toDisplaySize(bytes) {
  if (!Number.isFinite(bytes)) {
    return "0 KB";
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function formatParamValue(value, precision) {
  return Number(value ?? 0).toFixed(precision);
}

function ResearchConsole({ imageFile, params, onFileChange, onParamChange, onExecute, loading, error }) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const fileLabel = useMemo(() => {
    if (!imageFile) {
      return "Drop an aerial frame or click to select.";
    }
    return `${imageFile.name} (${toDisplaySize(imageFile.size)})`;
  }, [imageFile]);

  const handleSubmit = (event) => {
    event.preventDefault();
    onExecute();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);

    const dropped = event.dataTransfer.files?.[0] ?? null;
    if (dropped && dropped.type?.startsWith("image/")) {
      onFileChange(dropped);
    }
  };

  return (
    <CornerFrame className="panel-shell panel-main-control rounded-sm px-5 py-6 md:px-8 md:py-7" markerTopLeft="SEC:03" markerBottomRight="CTRL-C">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="mb-2 flex flex-wrap gap-2">
            <MicroLabel>MODEL VERSION: v0.3.2 EXPERIMENTAL</MicroLabel>
            <MicroLabel>BACKBONE: RESNET50 (IMAGENET PRETRAINED)</MicroLabel>
          </div>
          <p className="section-label">Research Console</p>
          <h2 className="mt-1 font-display text-xl tracking-[0.1em] text-slate-50 md:text-2xl">TACTICAL CONTROL PANEL</h2>
        </div>

        <div className="inline-flex items-center gap-2 border border-slate-600/70 bg-slate-950/75 px-3 py-2 text-xs text-slate-300">
          <Cpu className="h-3.5 w-3.5 text-accent-slicer" />
          Lightweight inference runtime controls
        </div>
      </div>

      <form className="grid gap-6 lg:grid-cols-[1.05fr_1fr]" onSubmit={handleSubmit}>
        <div className="control-focus-shell space-y-4">
          <p className="section-label">INPUT</p>

          <div
            className={`upload-shell ${dragActive ? "upload-shell-active" : ""}`}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={(event) => {
              event.preventDefault();
              setDragActive(false);
            }}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                inputRef.current?.click();
              }
            }}
            role="button"
            tabIndex={0}
          >
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              accept="image/*"
              onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
            />

            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center border border-slate-500 bg-slate-900">
                <Upload className="h-5 w-5 text-accent-slicer" />
              </div>

              <div>
                <p className="font-display text-xs tracking-[0.15em] text-slate-100">INPUT FEED</p>
                <p className="body-copy mt-1 text-xs">{fileLabel}</p>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={!imageFile || loading}
            className={`execute-btn ${loading ? "execute-btn-loading" : ""}`}
          >
            <span className="execute-btn-rail" aria-hidden="true" />
            <span className="execute-btn-main">{loading ? "EXECUTING PIPELINE" : "EXECUTE PIPELINE"}</span>
            <span className="execute-btn-sub">{loading ? "RUN SEQUENCE INITIATED" : "RUNTIME ESTIMATE: ~420ms"}</span>
            {loading ? <span className="execute-loader" aria-hidden="true" /> : null}
          </button>

          {error ? <p className="text-xs text-rose-300">{error}</p> : null}
        </div>

        <div className="space-y-4">
          {PARAM_GROUPS.map((group) => (
            <div key={group.title} className="border border-slate-700/80 bg-slate-950/60 px-4 py-4">
              <p className="mb-3 flex items-center gap-2 font-display text-xs tracking-[0.16em] text-slate-100">
                <SlidersHorizontal className="h-3.5 w-3.5 text-accent-slicer" />
                {group.title}
              </p>

              <div className="space-y-3">
                {group.values.map((param) => {
                  const rawValue = Number(params[param.key] ?? 0);
                  return (
                    <label key={param.key} className="block">
                      <div className="mb-1.5 flex items-center justify-between gap-3 text-xs">
                        <span className="text-slate-300">{param.label}</span>
                        <span className="font-mono text-slate-100">{formatParamValue(rawValue, param.precision)}</span>
                      </div>

                      <input
                        type="range"
                        className="range-input"
                        style={{ "--slider-accent": param.accent }}
                        min={param.min}
                        max={param.max}
                        step={param.step}
                        value={rawValue}
                        onChange={(event) => {
                          const nextValue = Number(event.target.value);
                          onParamChange(param.key, param.precision === 0 ? Math.round(nextValue) : nextValue);
                        }}
                      />
                    </label>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </form>
    </CornerFrame>
  );
}

export default memo(ResearchConsole);
