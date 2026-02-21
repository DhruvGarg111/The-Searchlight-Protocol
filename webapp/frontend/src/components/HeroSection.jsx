import { memo } from "react";
import { Eye, Radar, ScanLine } from "lucide-react";
import CornerFrame from "./ui/CornerFrame";
import MicroLabel from "./ui/MicroLabel";

const STAGE_NODES = [
  {
    id: "guide",
    label: "GUIDE",
    title: "ResNet50 + LayerCAM",
    top: "18%",
    left: "10%",
    accentClass: "node-guide",
  },
  {
    id: "slicer",
    label: "SLICER",
    title: "Heatmap-Based Slicing",
    top: "52%",
    left: "40%",
    accentClass: "node-slicer",
  },
  {
    id: "detector",
    label: "DETECTOR",
    title: "YOLOv8 + Global Fusion",
    top: "26%",
    left: "70%",
    accentClass: "node-detector",
  },
];

function HeroSection() {
  return (
    <CornerFrame className="panel-shell hero-shell rounded-sm px-5 py-6 md:px-8 md:py-8" markerTopLeft="SEC:01" markerBottomRight="HERO-A">
      <div className="grid items-start gap-7 lg:grid-cols-[1.2fr_0.82fr]">
        <div className="hero-diagram-wrap md:-ml-2">
          <MicroLabel>SEMANTIC ACTIVATION MAP</MicroLabel>

          <div className="hero-diagram-shell relative h-[332px] overflow-hidden p-4 md:p-5">
            <div className="absolute inset-0 bg-[linear-gradient(160deg,rgba(124,92,255,0.08),transparent_44%,rgba(34,211,238,0.05)_72%,rgba(251,191,36,0.05))]" />

            <div className="hero-link hero-link-a" aria-hidden="true" />
            <div className="hero-link hero-link-b" aria-hidden="true" />

            <div className="hero-radar" aria-hidden="true">
              <div className="hero-radar-ring" />
              <div className="hero-radar-ring hero-radar-ring-inner" />
              <div className="hero-radar-sweep" />
            </div>

            {STAGE_NODES.map((node) => (
              <div
                key={node.id}
                className={`hero-node ${node.accentClass}`}
                style={{ top: node.top, left: node.left }}
              >
                <span className="hero-node-label">{node.label}</span>
                <p className="hero-node-title">{node.title}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4 lg:-mt-2 lg:pt-2">
          <p className="hero-kicker">EXPERIMENTAL AERIAL INTELLIGENCE STACK</p>
          <p className="section-label">LayerCAM-Guided Aerial Intelligence</p>

          <h1 className="font-display text-4xl leading-[0.98] tracking-[0.1em] text-slate-100 md:text-6xl">
            THE SEARCHLIGHT PROTOCOL
          </h1>

          <p className="body-copy max-w-xl text-slate-400/80">
            Research interface for a three-stage aerial intelligence stack that fuses semantic attention guidance,
            intelligent slicing, and global detection fusion for high-resolution tactical scenes.
          </p>

          <div className="flex flex-wrap gap-2 text-xs">
            <div className="tag-chip">
              <Radar className="h-3.5 w-3.5 text-accent-slicer" />
              ATTENTION
            </div>
            <div className="tag-chip">
              <Eye className="h-3.5 w-3.5 text-accent-guide" />
              LOCALIZATION
            </div>
            <div className="tag-chip">
              <ScanLine className="h-3.5 w-3.5 text-accent-detector" />
              GLOBAL FUSION
            </div>
          </div>

          <div className="status-line">
            <span className="status-dot" aria-hidden="true" />
            SYSTEM STATUS: ACTIVE
          </div>

        </div>
      </div>
    </CornerFrame>
  );
}

export default memo(HeroSection);
