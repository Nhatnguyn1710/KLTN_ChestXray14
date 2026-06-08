import { useCallback, useEffect, useRef, useState } from "react";
import { XRayHeader } from "./components/XRayHeader.jsx";
import { XRayUploadPanel } from "./components/XRayUploadPanel.jsx";
import { XRayResultsPanel } from "./components/XRayResultsPanel.jsx";
import { useHealth } from "./hooks/health/useHealth.js";
import { usePredict } from "./hooks/predict/usePredict.js";
import { VI_LABELS } from "./configs/labels.js";

const HISTORY_KEY = "xray_scan_history_v2";
const HISTORY_MAX = 30;

const STAGE_PROGRESS_TARGETS = [25, 65, 85];
const STAGE_INTERVAL_MS = 1100;

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr.slice(0, HISTORY_MAX) : [];
  } catch {
    return [];
  }
}

function saveHistory(items) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX)));
  } catch {
    /* quota exceeded — ignore */
  }
}

async function makeThumbnail(file, size = 90) {
  return new Promise((resolve) => {
    try {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          URL.revokeObjectURL(url);
          return resolve(null);
        }
        const min = Math.min(img.width, img.height);
        const sx = (img.width - min) / 2;
        const sy = (img.height - min) / 2;
        ctx.drawImage(img, sx, sy, min, min, 0, 0, size, size);
        URL.revokeObjectURL(url);
        try {
          resolve(canvas.toDataURL("image/jpeg", 0.7));
        } catch {
          resolve(null);
        }
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(null);
      };
      img.src = url;
    } catch {
      resolve(null);
    }
  });
}

export default function App() {
  /* Toast (inline) */
  const [toastMsg, setToastMsg] = useState(null);
  const toastTimer = useRef(null);
  const toast = useCallback((message, type = "info") => {
    setToastMsg({ message, type, id: Date.now() });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToastMsg(null), 3500);
  }, []);

  /* Hooks */
  const { health } = useHealth(toast);
  const { predicting, result, runPredict } = usePredict(toast);

  /* Phase + UI state */
  const [phase, setPhase] = useState("idle");
  const [activeTab, setActiveTab] = useState("classification");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [filename, setFilename] = useState("");
  const [dragging, setDragging] = useState(false);

  const [sensitivity, setSensitivity] = useState(0.5);
  const [posture, setPosture] = useState("auto");
  const [gradCamEnabled, setGradCamEnabled] = useState(true);

  /* Stage animation */
  const [analysisStage, setAnalysisStage] = useState(0);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const stageTimers = useRef([]);

  /* History */
  const [history, setHistory] = useState(() => loadHistory());

  const onClearHistory = useCallback(() => {
    setHistory([]);
    saveHistory([]);
    toast("Đã xoá toàn bộ lịch sử", "info");
  }, [toast]);

  const onDeleteItem = useCallback((id) => {
    setHistory((prev) => {
      const next = prev.filter((item) => item.id !== id);
      saveHistory(next);
      return next;
    });
  }, []);

  const clearStageTimers = () => {
    stageTimers.current.forEach(clearTimeout);
    stageTimers.current = [];
  };

  /* File handlers */
  const onFile = useCallback((newFile) => {
    if (!newFile) return;
    const maxBytes = (health.maxUploadMb || 10) * 1024 * 1024;
    if (newFile.size > maxBytes) {
      toast(`Ảnh vượt quá ${health.maxUploadMb || 10} MB`, "error");
      return;
    }
    if (!/^image\//.test(newFile.type) && !/\.(png|jpe?g|bmp|webp)$/i.test(newFile.name)) {
      toast("Chỉ chấp nhận file ảnh (PNG/JPG/BMP/WEBP)", "error");
      return;
    }
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(newFile);
    });
    setFile(newFile);
    setFilename(newFile.name);
    setPhase("uploaded");
    setActiveTab("classification");
  }, [health.maxUploadMb, toast]);

  const onClearFile = useCallback(() => {
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setFile(null);
    setFilename("");
    setPhase("idle");
  }, []);

  /* Analyze */
  const startAnalysis = useCallback(async () => {
    if (!file || predicting) return;
    if (!health.cnn) {
      toast("CNN chưa sẵn sàng", "error");
      return;
    }
    clearStageTimers();
    setPhase("analyzing");
    setAnalysisStage(0);
    setAnalysisProgress(5);

    STAGE_PROGRESS_TARGETS.forEach((target, idx) => {
      stageTimers.current.push(
        setTimeout(() => {
          setAnalysisStage(idx);
          setAnalysisProgress(target);
        }, idx * STAGE_INTERVAL_MS)
      );
    });

    const thumbnailPromise = makeThumbnail(file).catch(() => null);
    let success = false;

    await runPredict({
      file,
      threshold: sensitivity,
      enableGradcam: gradCamEnabled,
      viewPosition: posture,
      onSuccess: async (data) => {
        success = true;
        clearStageTimers();
        setAnalysisStage(3);
        setAnalysisProgress(100);

        const thumbnail = await thumbnailPromise;
        const detectedCount = typeof data.detected_count === "number"
          ? data.detected_count
          : (data.classifications || []).filter((c) => c.detected).length;
        const sortedDetected = [...(data.classifications || [])]
          .filter((c) => c.detected)
          .sort((a, b) => {
            const bm = typeof b.margin === "number" ? b.margin : b.probability - b.threshold;
            const am = typeof a.margin === "number" ? a.margin : a.probability - a.threshold;
            return bm - am;
          });
        const top = sortedDetected[0];
        const now = new Date();
        const newScan = {
          id: `scan-${Date.now()}`,
          date: now.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" }),
          time: now.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" }),
          filename: file.name,
          studyTriage: data.study_triage || null,
          detected: detectedCount,
          total: (data.classifications || []).length || 14,
          topFinding: top?.label || null,
          topFindingVi: top?.label_vi || (top ? VI_LABELS[top.label] : null),
          topProb: top ? top.probability * 100 : 0,
          posture: posture === "auto" ? (data.image_metadata?.view_position || "Auto") : posture,
          thumbnail: thumbnail || null,
        };
        setHistory((prev) => {
          const next = [newScan, ...prev].slice(0, HISTORY_MAX);
          saveHistory(next);
          return next;
        });
        setTimeout(() => setPhase("done"), 250);
      },
    });

    if (!success) {
      clearStageTimers();
      setPhase("uploaded");
    }
  }, [file, predicting, health.cnn, runPredict, sensitivity, gradCamEnabled, posture, toast]);

  /* Cleanup */
  useEffect(() => () => {
    clearStageTimers();
    if (toastTimer.current) clearTimeout(toastTimer.current);
  }, []);

  /* Keyboard shortcut */
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        startAnalysis();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [startAnalysis]);

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <XRayHeader health={health} />

      <div className="flex flex-1 overflow-hidden">
        <XRayUploadPanel
          phase={phase}
          file={file}
          previewUrl={previewUrl}
          filename={filename}
          onFile={onFile}
          onClearFile={onClearFile}
          sensitivity={sensitivity}
          setSensitivity={setSensitivity}
          posture={posture}
          setPosture={setPosture}
          gradCamEnabled={gradCamEnabled}
          setGradCamEnabled={setGradCamEnabled}
          onAnalyze={startAnalysis}
          analysisStage={analysisStage}
          analysisProgress={analysisProgress}
          maxUploadMb={health.maxUploadMb || 10}
          cnnReady={health.cnn}
          dragging={dragging}
          setDragging={setDragging}
        />

        <XRayResultsPanel
          phase={phase}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          result={result}
          history={history}
          onClearHistory={onClearHistory}
          onDeleteItem={onDeleteItem}
          analysisStage={analysisStage}
          analysisProgress={analysisProgress}
          filename={filename}
          posture={posture}
          previewUrl={previewUrl}
        />
      </div>

      {toastMsg && <Toast {...toastMsg} />}
    </div>
  );
}

function Toast({ message, type }) {
  const config = {
    success: { bg: "#ECFDF5", color: "#065F46", border: "#A7F3D0" },
    error:   { bg: "#FEF2F2", color: "#B91C1C", border: "#FECACA" },
    info:    { bg: "#EFF6FF", color: "#1E40AF", border: "#BFDBFE" },
  }[type] || { bg: "#EFF6FF", color: "#1E40AF", border: "#BFDBFE" };

  return (
    <div
      className="fixed bottom-6 right-6 px-4 py-3 rounded-2xl text-sm shadow-lg z-50"
      style={{
        background: config.bg,
        color: config.color,
        border: `1px solid ${config.border}`,
        fontWeight: 600,
        maxWidth: "400px",
        animation: "medFadeInUp 0.3s ease",
      }}
    >
      {message}
    </div>
  );
}
