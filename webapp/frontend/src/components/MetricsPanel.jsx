import { memo, useEffect, useMemo, useState } from "react";
import { Clock3, Crop, Crosshair, Percent } from "lucide-react";
import CornerFrame from "./ui/CornerFrame";
import MicroLabel from "./ui/MicroLabel";

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function mergedYLength(segments) {
  if (!segments.length) {
    return 0;
  }

  const sorted = [...segments].sort((a, b) => a[0] - b[0]);
  let total = 0;
  let currentStart = sorted[0][0];
  let currentEnd = sorted[0][1];

  for (let index = 1; index < sorted.length; index += 1) {
    const [start, end] = sorted[index];
    if (start > currentEnd) {
      total += currentEnd - currentStart;
      currentStart = start;
      currentEnd = end;
    } else {
      currentEnd = Math.max(currentEnd, end);
    }
  }

  total += currentEnd - currentStart;
  return total;
}

function rectanglesUnionArea(rectangles) {
  if (!rectangles.length) {
    return 0;
  }

  const events = [];
  rectangles.forEach((rect, id) => {
    events.push({ x: rect[0], id, type: 1, y1: rect[1], y2: rect[3] });
    events.push({ x: rect[2], id, type: -1, y1: rect[1], y2: rect[3] });
  });

  events.sort((a, b) => (a.x === b.x ? b.type - a.type : a.x - b.x));

  const active = new Map();
  let area = 0;
  let previousX = events[0].x;

  for (const event of events) {
    const deltaX = event.x - previousX;
    if (deltaX > 0 && active.size > 0) {
      const activeSegments = Array.from(active.values());
      area += deltaX * mergedYLength(activeSegments);
    }

    if (event.type === 1) {
      active.set(event.id, [event.y1, event.y2]);
    } else {
      active.delete(event.id);
    }

    previousX = event.x;
  }

  return area;
}

function resolveImageDimensions(originalSize, crops) {
  let width = toNumber(originalSize?.[0]);
  let height = toNumber(originalSize?.[1]);
  if (width <= 0 || height <= 0) {
    return { width: 0, height: 0 };
  }

  if (Array.isArray(crops) && crops.length > 0) {
    let maxX2 = 0;
    let maxY2 = 0;
    for (const crop of crops) {
      const [x = 0, y = 0, w = 0, h = 0] = Array.isArray(crop?.bbox) ? crop.bbox : [];
      maxX2 = Math.max(maxX2, toNumber(x) + toNumber(w));
      maxY2 = Math.max(maxY2, toNumber(y) + toNumber(h));
    }

    if (maxX2 > width && maxX2 <= height && maxY2 <= width) {
      const tmp = width;
      width = height;
      height = tmp;
    }
  }

  return { width, height };
}

function computeBackgroundSkippedPercent(result) {
  const crops = Array.isArray(result?.crops) ? result.crops : [];
  const { width, height } = resolveImageDimensions(result?.meta?.original_size, crops);
  if (width <= 0 || height <= 0) {
    return null;
  }

  const imageArea = width * height;
  if (imageArea <= 0) {
    return null;
  }

  const rectangles = crops
    .map((crop) => {
      const [x = 0, y = 0, w = 0, h = 0] = Array.isArray(crop?.bbox) ? crop.bbox : [];
      const x1 = Math.max(0, toNumber(x));
      const y1 = Math.max(0, toNumber(y));
      const x2 = Math.min(width, x1 + Math.max(0, toNumber(w)));
      const y2 = Math.min(height, y1 + Math.max(0, toNumber(h)));
      return [x1, y1, x2, y2];
    })
    .filter(([x1, y1, x2, y2]) => x2 > x1 && y2 > y1);

  const coveredArea = rectanglesUnionArea(rectangles);
  const skipped = ((imageArea - coveredArea) / imageArea) * 100;
  return Math.min(100, Math.max(0, skipped));
}

function useSteppedCounter(target, trigger) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    const numericTarget = toNumber(target);
    const steps = 16;
    const tickMs = 18;
    let cancelled = false;
    let timer = null;
    let step = 0;

    setValue(0);

    const advance = () => {
      if (cancelled) {
        return;
      }

      step += 1;
      const next = (numericTarget * step) / steps;
      setValue(step >= steps ? numericTarget : next);

      if (step < steps) {
        timer = setTimeout(advance, tickMs);
      }
    };

    timer = setTimeout(advance, 0);

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [target, trigger]);

  return value;
}

const METRIC_DEFS = [
  { key: "inference", label: "Inference Time", suffix: " ms", decimals: 1, icon: Clock3, accent: "text-accent-slicer" },
  { key: "crops", label: "Crops Generated", suffix: "", decimals: 0, icon: Crop, accent: "text-accent-guide" },
  { key: "skipped", label: "Background Skipped", suffix: "%", decimals: 1, icon: Percent, accent: "text-accent-detector" },
  { key: "detections", label: "Detections Found", suffix: "", decimals: 0, icon: Crosshair, accent: "text-accent-detector" },
];

function MetricsPanel({ result }) {
  const runKey = result?.research?.run_id ?? "idle";

  const values = useMemo(() => {
    const pre = toNumber(result?.counts?.pre_nms_crops);
    const post = toNumber(result?.counts?.post_nms_crops);
    const areaBasedSkipped = computeBackgroundSkippedPercent(result);
    const nmsBasedSkipped = pre > 0 ? Math.max(0, ((pre - post) / pre) * 100) : 0;

    return {
      inference: toNumber(result?.research?.timings_ms?.total_pipeline),
      crops: pre,
      skipped: areaBasedSkipped ?? nmsBasedSkipped,
      detections: toNumber(result?.counts?.detections),
    };
  }, [result]);

  const counters = {
    inference: useSteppedCounter(values.inference, runKey),
    crops: useSteppedCounter(values.crops, runKey),
    skipped: useSteppedCounter(values.skipped, runKey),
    detections: useSteppedCounter(values.detections, runKey),
  };

  return (
    <CornerFrame className="panel-shell rounded-sm px-5 py-6 md:px-8 md:py-7" markerTopLeft="SEC:05" markerBottomRight="MET-E">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <MicroLabel className="mb-2">INFERENCE PROFILING STREAM</MicroLabel>
          <p className="section-label">Metrics Panel</p>
          <h2 className="mt-1 font-display text-xl tracking-[0.1em] text-slate-100 md:text-2xl">INFERENCE METRICS</h2>
        </div>
        {!result ? <p className="body-copy text-xs">Run inference to populate metrics.</p> : null}
      </div>

      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
        {METRIC_DEFS.map((metric) => {
          const Icon = metric.icon;
          const currentValue = counters[metric.key];

          return (
            <article key={metric.key} className="border border-slate-700/80 bg-slate-950/60 px-4 py-4">
              <div className="mb-3 flex items-center justify-between text-slate-300">
                <span className="font-display text-[11px] tracking-[0.16em]">{metric.label}</span>
                <Icon className={`h-4 w-4 ${metric.accent}`} />
              </div>

              <span className="font-mono text-2xl text-slate-100 md:text-3xl">
                {currentValue.toFixed(metric.decimals)}
                {metric.suffix}
              </span>
            </article>
          );
        })}
      </div>
    </CornerFrame>
  );
}

export default memo(MetricsPanel);
