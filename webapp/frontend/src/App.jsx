import { useCallback, useState } from "react";
import HeroSection from "./components/HeroSection";
import MetricsPanel from "./components/MetricsPanel";
import PipelineGraph from "./components/PipelineGraph";
import ResearchConsole from "./components/ResearchConsole";
import ResultsPanel from "./components/ResultsPanel";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const DEFAULT_PARAMS = {
  heatmap_threshold: 0.4,
  padding_factor: 0.4,
  yolo_confidence: 0.3,
  nms_iou_threshold: 0.2,
};

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleParamChange = useCallback((name, value) => {
    setParams((prev) => {
      if (prev[name] === value) {
        return prev;
      }
      return {
        ...prev,
        [name]: value,
      };
    });
  }, []);

  const handleExecute = useCallback(async () => {
    if (!imageFile || loading) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("image", imageFile);

    Object.entries(params).forEach(([key, value]) => {
      formData.append(key, String(value));
    });

    try {
      const response = await fetch(`${API_BASE_URL}/api/run-pipeline`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Pipeline execution failed.");
      }

      setResult(payload);
    } catch (runError) {
      setError(runError.message || "Pipeline execution failed.");
    } finally {
      setLoading(false);
    }
  }, [imageFile, loading, params]);

  return (
    <div className="app-background relative min-h-screen overflow-x-clip">
      <div className="svg-grid-overlay pointer-events-none absolute inset-0 opacity-35" aria-hidden="true" />
      <div className="scanline-overlay pointer-events-none absolute inset-0" aria-hidden="true" />

      <main className="relative z-10 mx-auto flex w-full max-w-7xl flex-col px-4 py-6 md:px-8 md:py-8">
        <div className="surrounding-panel md:translate-x-3">
          <HeroSection />
        </div>

        <div className="surrounding-panel -mt-1 md:-mt-2 md:-ml-4">
          <PipelineGraph />
        </div>

        <div className="mt-5 md:mt-7 md:mr-5">
          <ResearchConsole
            imageFile={imageFile}
            params={params}
            onFileChange={setImageFile}
            onParamChange={handleParamChange}
            onExecute={handleExecute}
            loading={loading}
            error={error}
          />
        </div>

        <div className="surrounding-panel mt-4 md:mt-8 md:ml-2">
          <ResultsPanel result={result} loading={loading} />
        </div>

        <div className="surrounding-panel mt-3 md:mt-6 md:-mr-3">
          <MetricsPanel result={result} />
        </div>
      </main>
    </div>
  );
}

export default App;
