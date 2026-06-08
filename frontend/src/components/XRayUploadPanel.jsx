import { useRef } from "react";
import { Upload, Settings2, Zap, ScanLine, X } from "lucide-react";

const ANALYSIS_STAGES = [
  "Tiền xử lý ảnh",
  "CNN phân tích",
  "Tạo Grad-CAM",
  "Tổng hợp kết quả",
];

const ACCEPTED_TYPES = "image/png,image/jpeg,image/jpg,image/bmp,image/webp";

export function XRayUploadPanel({
  phase,
  file,
  previewUrl,
  filename,
  onFile,
  onClearFile,
  sensitivity,
  setSensitivity,
  posture,
  setPosture,
  gradCamEnabled,
  setGradCamEnabled,
  onAnalyze,
  analysisStage,
  analysisProgress,
  maxUploadMb,
  cnnReady,
  dragging,
  setDragging,
}) {
  const inputRef = useRef(null);
  const isAnalyzing = phase === "analyzing";
  const isDone = phase === "done";
  const hasImage = !!previewUrl;
  const pct = ((sensitivity - 0.1) / 0.8) * 100;

  const handleFiles = (files) => {
    if (!files || !files[0]) return;
    onFile(files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleClickZone = () => {
    if (isAnalyzing) return;
    inputRef.current?.click();
  };

  const canAnalyze = hasImage && cnnReady && !isAnalyzing;

  return (
    <div
      className="flex flex-col flex-shrink-0 h-full"
      style={{
        width: "300px",
        background: "#FFFFFF",
        borderRight: "1px solid #E8EFF6",
      }}
    >
      {/* ── Panel header ── */}
      <div
        className="flex items-center gap-2.5 px-4 py-3.5 flex-shrink-0"
        style={{ borderBottom: "1px solid #F0F5FA" }}
      >
        <div
          className="w-8 h-8 rounded-xl flex items-center justify-center"
          style={{ background: "#EFF6FF", border: "1px solid #BFDBFE" }}
        >
          <Upload className="w-4 h-4" style={{ color: "#1D72F5" }} />
        </div>
        <div>
          <div className="text-sm" style={{ color: "#0F172A", fontWeight: 700 }}>
            Tải ảnh X-quang
          </div>
          <div className="text-xs" style={{ color: "#94A3B8" }}>
            PNG · JPG · BMP · Tối đa {maxUploadMb} MB
          </div>
        </div>
      </div>

      {/* ── X-ray preview / upload zone ── */}
      <div className="px-4 pt-4 pb-2 flex-shrink-0">
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES}
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div
          onClick={handleClickZone}
          onDragOver={(e) => { e.preventDefault(); if (!isAnalyzing) setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={`relative rounded-2xl overflow-hidden upload-zone ${hasImage ? 'upload-zone-loaded' : 'upload-zone-empty'}`}
          style={{
            background: hasImage ? "#000" : dragging ? "#EFF6FF" : "#F8FAFC",
            aspectRatio: "1 / 1",
            maxHeight: "252px",
            cursor: isAnalyzing ? "default" : "pointer",
            border: dragging
              ? "2px solid #1D72F5"
              : hasImage
              ? "2px solid #1D72F5"
              : "2px dashed #CBD5E1",
            boxShadow: hasImage || dragging ? "0 0 0 3px rgba(29,114,245,0.1)" : "none",
          }}
        >
          {hasImage ? (
            <>
              <img
                src={previewUrl}
                alt="Chest X-Ray"
                className="w-full h-full object-cover"
                style={{
                  filter: "contrast(1.12) brightness(0.95)",
                  opacity: isAnalyzing ? 0.82 : 1,
                  transition: "opacity 0.3s",
                }}
              />

              {isAnalyzing && (
                <div className="absolute inset-0">
                  <div className="absolute inset-0" style={{ background: "rgba(6, 18, 38, 0.45)" }} />

                  <div
                    className="absolute left-0 right-0"
                    style={{
                      height: "3px",
                      background:
                        "linear-gradient(90deg, transparent 0%, #06B6D4 15%, rgba(255,255,255,0.9) 50%, #06B6D4 85%, transparent 100%)",
                      boxShadow:
                        "0 0 18px rgba(6,182,212,0.9), 0 0 36px rgba(6,182,212,0.45), 0 -1px 8px rgba(255,255,255,0.3)",
                      animation: "xrayScanBeam 1.8s linear infinite",
                    }}
                  />

                  {[
                    { top: 10, left: 10 },
                    { top: 10, right: 10 },
                    { bottom: 10, left: 10 },
                    { bottom: 10, right: 10 },
                  ].map((pos, i) => (
                    <div
                      key={i}
                      className="absolute w-5 h-5"
                      style={{
                        ...pos,
                        borderTop: i === 0 || i === 1 ? "2px solid rgba(6,182,212,0.85)" : "none",
                        borderBottom: i === 2 || i === 3 ? "2px solid rgba(6,182,212,0.85)" : "none",
                        borderLeft: i === 0 || i === 2 ? "2px solid rgba(6,182,212,0.85)" : "none",
                        borderRight: i === 1 || i === 3 ? "2px solid rgba(6,182,212,0.85)" : "none",
                        borderRadius: "2px",
                      }}
                    />
                  ))}

                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-1">
                    <div
                      className="text-4xl text-white"
                      style={{
                        fontWeight: 800,
                        fontVariantNumeric: "tabular-nums",
                        textShadow: "0 2px 12px rgba(6,182,212,0.7)",
                        letterSpacing: "-0.03em",
                        animation: "medFadeInUp 0.3s ease",
                      }}
                    >
                      {analysisProgress}
                      <span className="text-xl" style={{ fontWeight: 600 }}>%</span>
                    </div>
                    <div
                      className="text-xs px-2 py-0.5 rounded-full text-center"
                      style={{
                        color: "#67E8F9",
                        background: "rgba(6,182,212,0.15)",
                        border: "1px solid rgba(6,182,212,0.3)",
                        fontWeight: 600,
                      }}
                    >
                      {ANALYSIS_STAGES[Math.min(analysisStage, ANALYSIS_STAGES.length - 1)]}
                    </div>
                  </div>

                  <div className="absolute bottom-0 left-0 right-0 h-1" style={{ background: "rgba(0,0,0,0.5)" }}>
                    <div
                      style={{
                        height: "100%",
                        width: `${analysisProgress}%`,
                        background: "linear-gradient(90deg, #06B6D4, #6366F1)",
                        transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
                      }}
                    />
                  </div>
                </div>
              )}

              {isDone && (
                <div
                  className="absolute top-2 right-2 flex items-center gap-1 px-2 py-0.5 rounded-full text-xs"
                  style={{
                    background: "rgba(16,185,129,0.18)",
                    border: "1px solid rgba(16,185,129,0.45)",
                    color: "#10B981",
                    fontWeight: 700,
                    animation: "medBadgeIn 0.4s ease",
                  }}
                >
                  ✓ Đã phân tích
                </div>
              )}

              {!isAnalyzing && (
                <button
                  onClick={(e) => { e.stopPropagation(); onClearFile(); }}
                  className="btn-icon absolute top-2 left-2 w-6 h-6 rounded-full flex items-center justify-center"
                  style={{
                    background: "rgba(0,0,0,0.6)",
                    backdropFilter: "blur(6px)",
                    color: "#fff",
                  }}
                  title="Xoá ảnh"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}

              <div
                className="absolute bottom-2 left-2 right-2 flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl"
                style={{ background: "rgba(0,0,0,0.68)", backdropFilter: "blur(6px)" }}
              >
                <ScanLine className="w-3 h-3 flex-shrink-0" style={{ color: "#67E8F9" }} />
                <span
                  className="text-xs truncate"
                  style={{ color: "rgba(255,255,255,0.75)", fontFamily: "monospace" }}
                >
                  {filename || "image.png"}
                </span>
              </div>
            </>
          ) : (
            <div
              className="absolute inset-0 flex flex-col items-center justify-center gap-2"
              style={{ background: "transparent" }}
            >
              <div
                className="upload-icon-box w-14 h-14 rounded-2xl flex items-center justify-center mb-1"
                style={{
                  background: dragging ? "rgba(29,114,245,0.12)" : "#EFF6FF",
                  border: `1.5px solid ${dragging ? "#1D72F5" : "#BFDBFE"}`,
                  transition: "background 0.2s, border-color 0.2s",
                }}
              >
                <Upload className="w-6 h-6" style={{ color: "#1D72F5" }} />
              </div>
              <div className="text-sm text-center" style={{ color: "#1E293B", fontWeight: 600 }}>
                Kéo thả ảnh vào đây
              </div>
              <div className="text-xs text-center" style={{ color: "#94A3B8" }}>
                hoặc nhấn để chọn file
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Settings (scrollable) ── */}
      <div className="flex-1 overflow-y-auto px-4 py-3 scrollbar-none">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-px flex-1" style={{ background: "#E8EFF6" }} />
          <div className="flex items-center gap-1.5">
            <Settings2 className="w-3.5 h-3.5" style={{ color: "#64748B" }} />
            <span
              className="text-xs uppercase tracking-wider"
              style={{ color: "#64748B", fontWeight: 700, letterSpacing: "0.08em" }}
            >
              Cài đặt lâm sàng
            </span>
          </div>
          <div className="h-px flex-1" style={{ background: "#E8EFF6" }} />
        </div>

        {/* Sensitivity */}
        <div className="mb-5">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs" style={{ color: "#475569", fontWeight: 600 }}>
              Độ nhạy phát hiện
            </span>
            <div
              className="px-2 py-0.5 rounded-md text-xs"
              style={{
                background: "#EFF6FF",
                color: "#1D4ED8",
                fontWeight: 800,
                fontVariantNumeric: "tabular-nums",
                border: "1px solid #BFDBFE",
              }}
            >
              {sensitivity.toFixed(2)}
            </div>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.01"
            value={sensitivity}
            onChange={(e) => setSensitivity(parseFloat(e.target.value))}
            className="w-full"
            style={{
              background: `linear-gradient(to right, #1D72F5 ${pct}%, #E2E8F0 ${pct}%)`,
            }}
          />
          <div className="flex justify-between mt-1">
            <span className="text-xs" style={{ color: "#94A3B8" }}>Nhạy cao</span>
            <span className="text-xs" style={{ color: "#94A3B8" }}>Chặt chẽ</span>
          </div>
        </div>

        {/* Posture */}
        <div className="mb-5">
          <div className="text-xs mb-2" style={{ color: "#475569", fontWeight: 600 }}>
            Tư thế chụp
          </div>
          <div className="relative">
            <select
              value={posture}
              onChange={(e) => setPosture(e.target.value)}
              className="select-field w-full text-sm outline-none appearance-none cursor-pointer"
              style={{
                background: "#F8FAFC",
                border: "1.5px solid #E2E8F0",
                color: "#1E293B",
                borderRadius: "12px",
                padding: "10px 36px 10px 12px",
              }}
            >
              <option value="auto">Tự động nhận diện</option>
              <option value="PA">PA (Posteroanterior)</option>
              <option value="AP">AP (Anteroposterior)</option>
              <option value="Lateral">Lateral (Bên)</option>
            </select>
            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                <path d="M6 9l6 6 6-6" stroke="#64748B" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
          </div>
        </div>

        {/* Grad-CAM toggle */}
        <div
          className="flex items-center justify-between p-3 rounded-xl"
          style={{ background: "#F8FAFC", border: "1px solid #E8EFF6" }}
        >
          <div>
            <div className="text-xs" style={{ color: "#1E293B", fontWeight: 600 }}>
              Kích hoạt Grad-CAM
            </div>
            <div className="text-xs mt-0.5" style={{ color: "#94A3B8" }}>
              Bản đồ nhiệt vùng bất thường
            </div>
          </div>
          <button
            onClick={() => setGradCamEnabled(!gradCamEnabled)}
            className="toggle-track relative flex-shrink-0 w-11 h-6 rounded-full"
            style={{
              background: gradCamEnabled
                ? "linear-gradient(135deg, #1D72F5, #06B6D4)"
                : "#CBD5E1",
              boxShadow: gradCamEnabled ? "0 2px 8px rgba(29,114,245,0.4)" : "none",
            }}
          >
            <div
              className="toggle-thumb absolute top-0.5 w-5 h-5 rounded-full bg-white"
              style={{
                left: gradCamEnabled ? "22px" : "2px",
                boxShadow: "0 1px 4px rgba(0,0,0,0.2)",
              }}
            />
          </button>
        </div>
      </div>

      {/* ── Analyze button ── */}
      <div className="px-4 pb-5 pt-3 flex-shrink-0" style={{ borderTop: "1px solid #F0F5FA" }}>
        <button
          onClick={onAnalyze}
          disabled={!canAnalyze}
          className="btn-primary ripple w-full flex items-center justify-center gap-2.5 py-3 rounded-2xl"
          style={{
            background: !canAnalyze
              ? "#94A3B8"
              : "linear-gradient(135deg, #1B6FE8 0%, #0284C7 100%)",
            color: "#fff",
            fontWeight: 700,
            fontSize: "14px",
            letterSpacing: "-0.01em",
            boxShadow: !canAnalyze
              ? "none"
              : "0 4px 20px rgba(29,114,245,0.45), 0 1px 0 rgba(255,255,255,0.15) inset",
            cursor: !canAnalyze ? "not-allowed" : "pointer",
            opacity: !canAnalyze ? 0.7 : 1,
          }}
        >
          {isAnalyzing ? (
            <>
              <div
                className="w-4 h-4 rounded-full border-2 border-white"
                style={{ borderTopColor: "transparent", animation: "medSpinRing 0.7s linear infinite" }}
              />
              Đang phân tích...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              Phân tích ảnh
            </>
          )}
        </button>
        <div className="flex items-center justify-center gap-1.5 mt-2">
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: "#1D72F5" }} />
          <span className="text-xs" style={{ color: "#64748B" }}>
            Ngưỡng phát hiện: <strong style={{ color: "#1D4ED8" }}>{(sensitivity * 100).toFixed(0)}%</strong>
          </span>
        </div>
      </div>
    </div>
  );
}
