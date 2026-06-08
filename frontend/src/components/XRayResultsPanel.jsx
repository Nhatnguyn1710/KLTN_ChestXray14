import { useMemo, useState } from "react";
import {
  BarChart2, Scan, History, AlertTriangle, CheckCircle, Info,
  Clock, FileText, Image as ImageIcon, ChevronLeft, ChevronRight,
  Trash2, Download, X,
} from "lucide-react";

const STAGES_INFO = [
  { label: "Tiền xử lý ảnh",     sublabel: "Chuẩn hoá & tăng cường đầu vào",   color: "#6366F1" },
  { label: "CNN phân tích",      sublabel: "DenseNet-121 xử lý 14 bệnh lý",   color: "#0EA5E9" },
  { label: "Tạo Grad-CAM",       sublabel: "Trực quan hoá vùng bất thường",    color: "#8B5CF6" },
  { label: "Tổng hợp kết quả",   sublabel: "Đối chiếu ngưỡng phát hiện",       color: "#10B981" },
];

const POSTURE_LABEL = {
  auto: "Chưa xác định",
  PA: "PA (Posteroanterior)",
  AP: "AP (Anteroposterior)",
  Lateral: "Lateral",
};
/** Phân tầng hai ngưỡng (API: tri_grade) + cận ngưỡng do giới hạn top-N */
const TRIAGE_LEVELS = {
  negative: { label: "Âm tính", color: "#059669", bg: "#ECFDF5", border: "#A7F3D0", rowBg: "transparent" },
  equivocal: { label: "Nghi ngờ", color: "#D97706", bg: "#FFFBEB", border: "#FDE68A", rowBg: "rgba(255,251,235,0.5)" },
  positive: { label: "Phát hiện", color: "#DC2626", bg: "#FEF2F2", border: "#FECACA", rowBg: "rgba(254,242,242,0.55)" },
  borderline: { label: "Cận ngưỡng", color: "#92400E", bg: "#FEF3C7", border: "#FDE68A", rowBg: "rgba(254,243,199,0.35)" },
};

function getDetectionMargin(item) {
  if (!item) return 0;
  if (typeof item.margin === "number") return item.margin;
  if (typeof item.probability === "number" && typeof item.threshold === "number") {
    return item.probability - item.threshold;
  }
  return 0;
}

function getTriDisplay(item) {
  const margin = getDetectionMargin(item);
  if (item?.borderline) return { ...TRIAGE_LEVELS.borderline, margin };
  const g = item?.tri_grade;
  if (g === "positive") return { ...TRIAGE_LEVELS.positive, margin };
  if (g === "equivocal") return { ...TRIAGE_LEVELS.equivocal, margin };
  return { ...TRIAGE_LEVELS.negative, margin };
}

export function XRayResultsPanel({
  phase,
  activeTab,
  setActiveTab,
  result,
  history,
  onClearHistory,
  onDeleteItem,
  analysisStage,
  analysisProgress,
  filename,
  posture,
  previewUrl,
}) {
  const isAnalyzing = phase === "analyzing";
  const isDone = phase === "done" && !!result;

  const classifications = result?.classifications || [];
  const detectedCount = result?.detected_count
    ?? classifications.filter((c) => c.detected).length;
  const totalCount = classifications.length || 14;
  const positiveCount = typeof result?.positive_count === "number"
    ? result.positive_count
    : classifications.filter((c) => c.tri_grade === "positive").length;
  const equivocalCount = typeof result?.equivocal_count === "number"
    ? result.equivocal_count
    : classifications.filter((c) => c.tri_grade === "equivocal").length;
  const negativeCount = typeof result?.negative_count === "number"
    ? result.negative_count
    : classifications.filter((c) => c.tri_grade === "negative").length;
  const thresholdDelta = (() => {
    const v = result?.threshold_delta;
    if (typeof v !== "number") return 0;
    return v * 100;
  })();
  const topFinding = useMemo(() => {
    const positive = classifications.filter((c) => c.tri_grade === "positive");
    if (positive.length > 0) {
      return [...positive].sort((a, b) => getDetectionMargin(b) - getDetectionMargin(a))[0];
    }
    const eq = classifications.filter((c) => c.tri_grade === "equivocal");
    if (eq.length === 0) return null;
    return [...eq].sort((a, b) => b.probability - a.probability)[0];
  }, [classifications]);

  const tabs = [
    { id: "classification", label: "Phân loại",    Icon: BarChart2, badge: isDone && positiveCount > 0 ? positiveCount : undefined, badgeAlert: positiveCount > 0 },
    { id: "gradcam",        label: "Grad-CAM",     Icon: Scan,       badge: undefined,                                                badgeAlert: false },
    { id: "history",        label: "Lịch sử quét", Icon: History,    badge: history.length || undefined,                              badgeAlert: false },
  ];

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden" style={{ background: "#F1F5F9" }}>
      {/* Tab bar */}
      <div
        className="flex items-end gap-0 px-5 flex-shrink-0"
        style={{ background: "#FFFFFF", borderBottom: "1px solid #E2EBF4", borderLeft: "1px solid #E8EFF6" }}
      >
        {tabs.map(({ id, label, Icon, badge, badgeAlert }) => {
          const active = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className="tab-btn relative flex items-center gap-2 px-4 py-3.5"
              style={{
                color: active ? "#1D72F5" : "#64748B",
                fontWeight: active ? 700 : 500,
                fontSize: "13px",
                borderBottom: active ? "2px solid #1D72F5" : "2px solid transparent",
                marginBottom: "-1px",
              }}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
              {badge !== undefined && (
                <span
                  className="px-1.5 py-0.5 rounded-full text-xs"
                  style={{
                    background: badgeAlert ? "#FEE2E2" : "#EFF6FF",
                    color: badgeAlert ? "#DC2626" : "#1D4ED8",
                    fontWeight: 700,
                    fontSize: "11px",
                    minWidth: "20px",
                    textAlign: "center",
                  }}
                >
                  {badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
        <div
          className="min-h-full rounded-2xl overflow-hidden"
          style={{ background: "#FFFFFF", border: "1px solid #E2EBF4", boxShadow: "0 2px 12px rgba(0,0,0,0.04)" }}
        >
        {isAnalyzing && (
          <AnalyzingView analysisStage={analysisStage} analysisProgress={analysisProgress} />
        )}

        {!isAnalyzing && activeTab === "classification" && (
          <ClassificationView
            result={result}
            classifications={classifications}
            positiveCount={positiveCount}
            equivocalCount={equivocalCount}
            negativeCount={negativeCount}
            thresholdDelta={thresholdDelta}
            topFinding={topFinding}
            filename={filename}
            posture={posture}
          />
        )}

        {!isAnalyzing && activeTab === "gradcam" && (
          <GradCamView
            result={result}
            previewUrl={previewUrl}
          />
        )}

        {!isAnalyzing && activeTab === "history" && (
          <HistoryView history={history} onClearHistory={onClearHistory} onDeleteItem={onDeleteItem} />
        )}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════ Analyzing ════════════════════ */
function AnalyzingView({ analysisStage, analysisProgress }) {
  const stageIdx = Math.min(analysisStage, STAGES_INFO.length - 1);
  return (
    <div className="flex flex-col items-center justify-center h-full px-8 py-10" style={{ animation: "medFadeInUp 0.4s ease" }}>
      <div className="relative mb-8">
        <div
          className="w-24 h-24 rounded-full border-4 flex items-center justify-center"
          style={{ borderColor: "rgba(6,182,212,0.2)", animation: "medPulse 2s ease-in-out infinite" }}
        >
          <div
            className="w-20 h-20 rounded-full border-4 flex items-center justify-center"
            style={{
              borderColor: "transparent",
              borderTopColor: "#06B6D4",
              borderRightColor: "#6366F1",
              animation: "medSpinRing 1.2s linear infinite",
            }}
          >
            <Scan className="w-8 h-8" style={{ color: "#06B6D4" }} />
          </div>
        </div>
        <div
          className="absolute inset-0 rounded-full"
          style={{
            border: "1px solid rgba(6,182,212,0.3)",
            animation: "medPulse 2s ease-in-out infinite",
            transform: "scale(1.15)",
          }}
        />
      </div>

      <div className="text-base mb-1.5 text-center" style={{ color: "#0F172A", fontWeight: 700 }}>
        {STAGES_INFO[stageIdx].label}
      </div>
      <div className="text-sm mb-8 text-center" style={{ color: "#64748B" }}>
        {STAGES_INFO[stageIdx].sublabel}
      </div>

      <div className="w-full max-w-md mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs" style={{ color: "#64748B", fontWeight: 600 }}>Tiến trình phân tích</span>
          <span className="text-xs" style={{ color: "#1D72F5", fontWeight: 800, fontVariantNumeric: "tabular-nums" }}>
            {analysisProgress}%
          </span>
        </div>
        <div className="relative h-2 rounded-full overflow-hidden" style={{ background: "#E2E8F0" }}>
          <div
            style={{
              height: "100%",
              width: `${analysisProgress}%`,
              background: "linear-gradient(90deg, #1D72F5 0%, #06B6D4 50%, #6366F1 100%)",
              borderRadius: "9999px",
              transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
              boxShadow: "0 0 8px rgba(6,182,212,0.6)",
            }}
          />
          <div
            className="absolute inset-0 rounded-full"
            style={{
              background: "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.35) 50%, transparent 100%)",
              animation: "xrayScanBeam 1.5s linear infinite",
              backgroundSize: "200% 100%",
            }}
          />
        </div>
      </div>

      <div className="w-full max-w-md space-y-2">
        {STAGES_INFO.map((s, idx) => {
          const done = idx < stageIdx;
          const current = idx === stageIdx;
          return (
            <div
              key={idx}
              className="flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all"
              style={{
                background: current ? "rgba(29,114,245,0.07)" : done ? "rgba(16,185,129,0.05)" : "transparent",
                border: current ? "1px solid rgba(29,114,245,0.2)" : "1px solid transparent",
              }}
            >
              <div
                className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
                style={{ background: done ? "#10B981" : current ? s.color : "#E2E8F0" }}
              >
                {done ? (
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d="M2 6l3 3 5-5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : current ? (
                  <div className="w-2 h-2 rounded-full bg-white" style={{ animation: "medPulse 1s infinite" }} />
                ) : (
                  <span style={{ fontSize: "10px", color: "#94A3B8", fontWeight: 700 }}>{idx + 1}</span>
                )}
              </div>
              <div className="flex-1">
                <div
                  className="text-xs"
                  style={{
                    color: done ? "#059669" : current ? "#0F172A" : "#94A3B8",
                    fontWeight: current || done ? 600 : 400,
                  }}
                >
                  {s.label}
                </div>
                {current && <div className="text-xs" style={{ color: "#64748B" }}>{s.sublabel}</div>}
              </div>
              {current && (
                <div className="text-xs" style={{ color: s.color, fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>
                  {analysisProgress}%
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ════════════════════ Classification ════════════════════ */
function ClassificationView({
  result,
  classifications,
  positiveCount,
  equivocalCount,
  negativeCount,
  thresholdDelta,
  topFinding,
  filename,
  posture,
}) {
  if (!result) {
    return <EmptyState icon={BarChart2} title="Chưa có kết quả" sub="Tải ảnh và nhấn Phân tích để bắt đầu" />;
  }

  const sorted = [...classifications].sort((a, b) => b.probability - a.probability);
  const topLevel = topFinding ? getTriDisplay(topFinding) : null;
  const metadataView = result?.image_metadata?.view_position;
  const displayPosture = posture === "auto" ? (metadataView || POSTURE_LABEL.auto) : (POSTURE_LABEL[posture] || posture);
  const studyTriage = result?.study_triage
    || (positiveCount > 0 ? "positive" : equivocalCount > 0 ? "equivocal" : "negative");

  return (
    <div className="p-5" style={{ animation: "medFadeInUp 0.4s ease" }}>
      {/* Stat cards — phân tầng hai ngưỡng */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        <div
          className="rounded-2xl p-4 text-center"
          style={{
            background: positiveCount > 0 ? "#FEF2F2" : "#F8FAFC",
            border: `1.5px solid ${positiveCount > 0 ? "#FECACA" : "#E2E8F0"}`,
            animation: "medBadgeIn 0.5s ease",
          }}
        >
          <div
            className="text-3xl"
            style={{
              fontWeight: 800,
              color: positiveCount > 0 ? "#DC2626" : "#94A3B8",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {positiveCount}
          </div>
          <div
            className="text-xs mt-1"
            style={{ color: positiveCount > 0 ? "#EF4444" : "#94A3B8", fontWeight: 600 }}
          >
            Phát hiện
          </div>
        </div>
        <div
          className="rounded-2xl p-4 text-center"
          style={{
            background: equivocalCount > 0 ? "#FFFBEB" : "#F8FAFC",
            border: `1.5px solid ${equivocalCount > 0 ? "#FDE68A" : "#E2E8F0"}`,
            animation: "medBadgeIn 0.5s 0.05s ease both",
          }}
        >
          <div
            className="text-3xl"
            style={{ fontWeight: 800, color: equivocalCount > 0 ? "#D97706" : "#94A3B8", fontVariantNumeric: "tabular-nums" }}
          >
            {equivocalCount}
          </div>
          <div className="text-xs mt-1" style={{ color: equivocalCount > 0 ? "#CA8A04" : "#94A3B8", fontWeight: 600 }}>Nghi ngờ</div>
        </div>
        <div
          className="rounded-2xl p-4 text-center"
          style={{ background: "#F0FDF4", border: "1.5px solid #BBF7D0", animation: "medBadgeIn 0.5s 0.1s ease both" }}
        >
          <div className="text-3xl" style={{ fontWeight: 800, color: "#16A34A", fontVariantNumeric: "tabular-nums" }}>
            {negativeCount}
          </div>
          <div className="text-xs mt-1" style={{ color: "#22C55E", fontWeight: 600 }}>Âm tính</div>
        </div>
        <div
          className="rounded-2xl p-4 text-center"
          style={{ background: "#EFF6FF", border: "1.5px solid #BFDBFE", animation: "medBadgeIn 0.5s 0.15s ease both" }}
        >
          <div className="text-3xl" style={{ fontWeight: 800, color: "#1D4ED8", fontVariantNumeric: "tabular-nums" }}>
            {thresholdDelta >= 0 ? "+" : ""}{thresholdDelta.toFixed(2)}
          </div>
          <div className="text-xs mt-1" style={{ color: "#3B82F6", fontWeight: 600 }}>Điều chỉnh ngưỡng</div>
        </div>
      </div>

      {/* Clinical summary */}
      {studyTriage === "positive" && topFinding && (
        <div
          className="rounded-2xl p-4 mb-5"
          style={{
            background: "linear-gradient(135deg, #FFFBEB 0%, #FEF9EC 100%)",
            border: "1.5px solid #FDE68A",
            animation: "medFadeInUp 0.5s 0.3s ease both",
          }}
        >
          <div className="flex items-start gap-3">
            <div
              className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: "#FEF3C7", border: "1px solid #FDE68A" }}
            >
              <AlertTriangle className="w-4 h-4" style={{ color: "#D97706" }} />
            </div>
            <div>
              <div className="text-sm mb-1" style={{ color: "#92400E", fontWeight: 700 }}>
                Tóm tắt lâm sàng
              </div>
              <p className="text-xs leading-relaxed" style={{ color: "#78350F" }}>
                Có dấu hiệu bất thường, cần kết hợp triệu chứng và tiền sử để xác nhận.
                <br />
                <span style={{ fontWeight: 600 }}>
                  Ưu tiên xem xét: {topFinding.label_vi || topFinding.label} ({(topFinding.probability * 100).toFixed(1)}%{topLevel ? ` · ${topLevel.label}` : ""})
                </span>
              </p>
              <div className="flex flex-wrap gap-1.5 mt-2">
                <span
                  className="px-2 py-0.5 rounded-full text-xs"
                  style={{
                    background: "#FEF3C7",
                    color: "#92400E",
                    fontWeight: 600,
                    border: "1px solid #FDE68A",
                  }}
                >
                  Tư thế: {displayPosture}
                </span>
                {filename && (
                  <span
                    className="px-2 py-0.5 rounded-full text-xs font-mono"
                    style={{
                      background: "#E0F2FE",
                      color: "#0369A1",
                      fontWeight: 600,
                      border: "1px solid #BAE6FD",
                    }}
                  >
                    {filename}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {studyTriage === "equivocal" && (
        <div
          className="rounded-2xl p-4 mb-5 flex items-start gap-3"
          style={{
            background: "linear-gradient(135deg, #FFFBEB 0%, #FEF9C3 100%)",
            border: "1.5px solid #FDE68A",
            animation: "medFadeInUp 0.5s 0.3s ease both",
          }}
        >
          <div
            className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ background: "#FEF3C7", border: "1px solid #FDE68A" }}
          >
            <AlertTriangle className="w-4 h-4" style={{ color: "#D97706" }} />
          </div>
          <div>
            <div className="text-sm mb-1" style={{ color: "#92400E", fontWeight: 700 }}>
              {result?.study_triage_vi || "Cần xem xét thêm (vùng trung gian)"}
            </div>
            <p className="text-xs leading-relaxed" style={{ color: "#78350F" }}>
              Có ít nhất một nhãn nằm giữa ngưỡng dưới và ngưỡng phát hiện. Nên đối chiếu lâm sàng hoặc xét thêm hình ảnh.
              {topFinding && (
                <>
                  <br />
                  <span style={{ fontWeight: 600 }}>
                    Gần ngưỡng nhất: {topFinding.label_vi || topFinding.label} ({(topFinding.probability * 100).toFixed(1)}%)
                  </span>
                </>
              )}
            </p>
          </div>
        </div>
      )}

      {studyTriage === "negative" && (
        <div
          className="rounded-2xl p-4 mb-5 flex items-start gap-3"
          style={{
            background: "linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%)",
            border: "1.5px solid #BBF7D0",
            animation: "medFadeInUp 0.5s 0.3s ease both",
          }}
        >
          <div
            className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ background: "#DCFCE7", border: "1px solid #BBF7D0" }}
          >
            <CheckCircle className="w-4 h-4" style={{ color: "#059669" }} />
          </div>
          <div>
            <div className="text-sm mb-1" style={{ color: "#065F46", fontWeight: 700 }}>
              Không phát hiện bất thường rõ
            </div>
            <p className="text-xs leading-relaxed" style={{ color: "#047857" }}>
              Tất cả nhãn dưới ngưỡng nghi ngờ (xác suất đã hiệu chỉnh). Vui lòng đối chiếu với lâm sàng và tiền sử bệnh.
            </p>
          </div>
        </div>
      )}

      {/* Disease table */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{
          background: "#F8FAFC",
          border: "1px solid #E2EBF4",
          animation: "medFadeInUp 0.5s 0.4s ease both",
        }}
      >
        <div
          className="grid gap-2 px-4 py-3 text-xs uppercase tracking-wider"
          style={{
            gridTemplateColumns: "1fr 1fr 1.65fr 1.05fr 1.05fr 1.15fr",
            background: "#F8FAFC",
            borderBottom: "1px solid #E2EBF4",
            color: "#64748B",
            fontWeight: 700,
            letterSpacing: "0.07em",
          }}
        >
          <span>Bệnh lý</span>
          <span>Tiếng Việt</span>
          <span>Xác suất</span>
          <span className="text-center">Ngưỡng nghi ngờ</span>
          <span className="text-center">Ngưỡng phát hiện</span>
          <span className="text-center">Phân loại</span>
        </div>

        {sorted.map((d, idx) => {
          const probPct = d.probability * 100;
          const thrHiPct = d.threshold * 100;
          const thrLo = typeof d.threshold_low === "number" ? d.threshold_low : d.threshold - 0.08;
          const thrLoPct = thrLo * 100;
          const detected = d.detected;
          const tri = getTriDisplay(d);
          const marginPct = Math.max(0, tri.margin * 100);
          const highlight = d.tri_grade === "positive" || d.tri_grade === "equivocal";
          return (
            <div
              key={d.label}
              className="disease-row grid items-center gap-2 px-4 py-3"
              style={{
                gridTemplateColumns: "1fr 1fr 1.65fr 1.05fr 1.05fr 1.15fr",
                borderBottom: idx < sorted.length - 1 ? "1px solid #F0F5FA" : "none",
                background: tri.rowBg,
                animation: `medSlideInRow 0.3s ${idx * 0.04}s ease both`,
              }}
            >
              <div className="flex items-center gap-2">
                {highlight && <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: tri.color }} />}
                <span
                  className="text-xs"
                  style={{
                    color: highlight ? tri.color : "#334155",
                    fontWeight: highlight ? 700 : 500,
                  }}
                >
                  {d.label}
                </span>
              </div>

              <span className="text-xs" style={{ color: "#64748B" }}>
                {d.label_vi || "—"}
              </span>

              <div className="flex items-center gap-2">
                <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "#E2E8F0" }}>
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.min(100, probPct)}%`,
                      background: d.tri_grade === "positive"
                        ? `linear-gradient(90deg, ${tri.color}, #F97316)`
                        : d.tri_grade === "equivocal"
                        ? "linear-gradient(90deg, #F59E0B, #EAB308)"
                        : "linear-gradient(90deg, #10B981, #06B6D4)",
                      borderRadius: "9999px",
                      transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
                    }}
                  />
                </div>
                <span
                  className="text-xs flex-shrink-0"
                  style={{
                    color: highlight ? tri.color : "#475569",
                    fontWeight: highlight ? 700 : 500,
                    fontVariantNumeric: "tabular-nums",
                    minWidth: "44px",
                  }}
                >
                  {probPct.toFixed(1)}%
                </span>
              </div>

              <span className="text-xs text-center" style={{ color: "#94A3B8", fontVariantNumeric: "tabular-nums" }}>
                {thrLoPct.toFixed(1)}%
              </span>
              <span className="text-xs text-center" style={{ color: "#64748B", fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>
                {thrHiPct.toFixed(1)}%
              </span>
              <div className="flex flex-col items-center justify-center gap-1">
                <span
                  className="px-2 py-0.5 rounded-full text-xs text-center"
                  style={{
                    background: tri.bg,
                    color: tri.color,
                    fontWeight: 700,
                    border: `1px solid ${tri.border}`,
                    minWidth: "82px",
                  }}
                >
                  {d.tri_label_vi || tri.label}
                </span>
                {detected && d.tri_grade === "positive" && (
                  <span
                    className="text-xs"
                    style={{ color: "#94A3B8", fontSize: "10px", fontVariantNumeric: "tabular-nums" }}
                  >
                    +{marginPct.toFixed(1)}% so với ngưỡng cao
                  </span>
                )}
                {d.tri_grade === "equivocal" && !d.borderline && (
                  <span className="text-xs" style={{ color: "#94A3B8", fontSize: "10px", fontVariantNumeric: "tabular-nums" }}>
                    Vùng trung gian
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ════════════════════ GradCAM ════════════════════ */
function GradCamView({ result, previewUrl }) {
  const items = result?.gradcam_items || [];
  const [idx, setIdx] = useState(0);

  if (!result) {
    return <EmptyState icon={Scan} title="Chưa có bản đồ nhiệt" sub="Thực hiện phân tích để tạo Grad-CAM" />;
  }
  if (items.length === 0) {
    return <EmptyState
      icon={Scan}
      title="Không có Grad-CAM"
      sub="Không có nhãn nào được phát hiện hoặc Grad-CAM bị tắt"
    />;
  }

  const safeIdx = Math.min(idx, items.length - 1);
  const current = items[safeIdx];
  const probPct = (current.probability * 100).toFixed(1);
  const camUrl = absoluteUrl(current.url);
  const currentFinding = (result?.classifications || []).find((c) => c.label === current.label) || current;
  const currentLevel = getTriDisplay(currentFinding);
  const currentMarginPct = Math.max(0, currentLevel.margin * 100);

  return (
    <div className="p-5" style={{ animation: "medFadeInUp 0.4s ease" }}>
      <div
        className="rounded-2xl overflow-hidden mb-5"
        style={{ background: "#F8FAFC", border: "1px solid #E2EBF4" }}
      >
        <div
          className="px-4 py-3 flex items-center justify-between"
          style={{ borderBottom: "1px solid #F0F5FA" }}
        >
          <div className="flex items-center gap-2">
            <Scan className="w-4 h-4" style={{ color: "#6366F1" }} />
            <span className="text-sm" style={{ fontWeight: 700, color: "#0F172A" }}>
              Bản đồ nhiệt Grad-CAM
            </span>
          </div>
          <span
            className="text-xs px-2 py-0.5 rounded-full"
            style={{
              background: "#EDE9FE",
              color: "#7C3AED",
              fontWeight: 600,
              border: "1px solid #DDD6FE",
            }}
          >
            {current.label_vi || current.label} · {probPct}%
          </span>
        </div>

        <div className="grid grid-cols-2 gap-0">
          <div className="relative" style={{ borderRight: "1px solid #F0F5FA" }}>
            <div
              className="px-3 py-2 text-xs"
              style={{ color: "#64748B", fontWeight: 600, background: "#F8FAFC", borderBottom: "1px solid #F0F5FA" }}
            >
              Ảnh gốc
            </div>
            <div className="relative overflow-hidden" style={{ aspectRatio: "1", background: "#000" }}>
              {previewUrl && (
                <img
                  src={previewUrl}
                  alt="Original"
                  className="w-full h-full object-cover"
                  style={{ filter: "contrast(1.1)" }}
                />
              )}
            </div>
          </div>

          <div className="relative">
            <div
              className="px-3 py-2 text-xs"
              style={{ color: "#64748B", fontWeight: 600, background: "#F8FAFC", borderBottom: "1px solid #F0F5FA" }}
            >
              Grad-CAM Overlay
            </div>
            <div className="relative overflow-hidden" style={{ aspectRatio: "1", background: "#000" }}>
              <img
                src={camUrl}
                alt={current.label}
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        </div>
        <div
          className="px-4 py-3 flex items-center gap-3"
          style={{ borderTop: "1px solid #F0F5FA" }}
        >
          <span className="text-xs" style={{ color: "#64748B", fontWeight: 500 }}>
            Mức nghi ngờ:
          </span>
          <span
            className="px-2.5 py-1 rounded-full text-xs"
            style={{
              background: currentLevel.bg,
              color: currentLevel.color,
              border: `1px solid ${currentLevel.border}`,
              fontWeight: 800,
            }}
          >
            {currentLevel.label}
          </span>
          <span className="text-xs" style={{ color: "#94A3B8", fontVariantNumeric: "tabular-nums" }}>
            Vượt ngưỡng {currentMarginPct.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Carousel of multiple labels */}
      {items.length > 1 && (
        <div
          className="rounded-2xl p-4 mb-5"
          style={{ background: "#FFFFFF", border: "1px solid #E2EBF4", boxShadow: "0 2px 12px rgba(0,0,0,0.05)" }}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <ImageIcon className="w-4 h-4" style={{ color: "#6366F1" }} />
              <span className="text-sm" style={{ fontWeight: 700, color: "#0F172A" }}>
                Các nhãn được giải thích ({items.length})
              </span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setIdx((safeIdx - 1 + items.length) % items.length)}
                className="btn-nav w-7 h-7 rounded-lg flex items-center justify-center"
                style={{ background: "#F1F5F9", color: "#475569" }}
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setIdx((safeIdx + 1) % items.length)}
                className="btn-nav w-7 h-7 rounded-lg flex items-center justify-center"
                style={{ background: "#F1F5F9", color: "#475569" }}
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {items.map((it, i) => {
              const active = i === safeIdx;
              return (
                <button
                  key={it.label + i}
                  onClick={() => setIdx(i)}
                  className="rounded-xl overflow-hidden text-left transition-all"
                  style={{
                    border: active ? "2px solid #1D72F5" : "1px solid #E2EBF4",
                    boxShadow: active ? "0 0 0 3px rgba(29,114,245,0.15)" : "none",
                    background: "#000",
                  }}
                >
                  <div className="relative" style={{ aspectRatio: "1" }}>
                    <img
                      src={absoluteUrl(it.url)}
                      alt={it.label}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="px-2 py-1.5" style={{ background: "#FFFFFF" }}>
                    <div className="text-xs truncate" style={{ color: "#0F172A", fontWeight: 600 }}>
                      {it.label_vi || it.label}
                    </div>
                    <div className="text-xs" style={{ color: "#64748B", fontVariantNumeric: "tabular-nums" }}>
                      {(it.probability * 100).toFixed(1)}%
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Region annotations */}
      <div
        className="rounded-2xl p-4"
        style={{ background: "#FFFFFF", border: "1px solid #E2EBF4", boxShadow: "0 2px 12px rgba(0,0,0,0.05)" }}
      >
        <div className="flex items-center gap-2 mb-3">
          <Info className="w-4 h-4" style={{ color: "#6366F1" }} />
          <span className="text-sm" style={{ fontWeight: 700, color: "#0F172A" }}>
            Các vùng được mô hình tập trung
          </span>
        </div>
        <div className="space-y-2">
          {items.slice(0, 5).map((r, i) => {
            const probPct2 = r.probability * 100;
            const relatedFinding = (result?.classifications || []).find((c) => c.label === r.label) || r;
            const intensity = getTriDisplay(relatedFinding);
            return (
              <div
                key={r.label + i}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl"
                style={{ background: "#F8FAFC", border: "1px solid #F0F5FA" }}
              >
                <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: intensity.color }} />
                <div className="flex-1">
                  <span className="text-xs" style={{ color: "#334155", fontWeight: 600 }}>
                    {r.label_vi || r.label}
                  </span>
                  <span className="text-xs ml-2" style={{ color: "#64748B" }}>
                    → {r.label}
                  </span>
                </div>
                <span
                  className="text-xs px-2 py-0.5 rounded-full"
                  style={{
                    background: intensity.bg,
                    color: intensity.color,
                    border: `1px solid ${intensity.border}`,
                    fontWeight: 700,
                  }}
                >
                  {intensity.label}
                </span>
                <span className="text-xs" style={{ color: "#94A3B8", fontVariantNumeric: "tabular-nums" }}>
                  {probPct2.toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════ History ════════════════════ */
function exportPdf(item) {
  const statusColor = item.detected > 0 ? "#DC2626" : "#059669";
  const statusBg    = item.detected > 0 ? "#FEF2F2" : "#F0FDF4";
  const statusText  = item.detected > 0
    ? `Phát hiện ${item.detected} bất thường`
    : "Không phát hiện bất thường";
  const findingHtml = item.detected > 0 && item.topFindingVi
    ? `<p style="margin:6px 0 0"><strong>Ưu tiên:</strong> ${item.topFindingVi} (${item.topProb.toFixed(1)}%)</p>`
    : "";
  const thumbHtml = item.thumbnail
    ? `<img src="${item.thumbnail}" style="width:120px;height:120px;object-fit:cover;border-radius:8px;border:1px solid #E2EBF4;" />`
    : "";

  const html = `<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>Báo cáo X-quang – ${item.filename}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Inter, sans-serif; background: #F8FAFC; color: #0F172A; padding: 0; }
    .page { max-width: 680px; margin: 0 auto; background: #fff; min-height: 100vh; padding: 40px 48px; }
    .header { background: linear-gradient(90deg,#2563EB,#3B82F6); color:#fff; border-radius:14px; padding:24px 28px; margin-bottom:28px; display:flex; align-items:center; justify-content:space-between; }
    .header h1 { font-size:20px; font-weight:800; }
    .header p  { font-size:12px; opacity:.8; margin-top:4px; }
    .badge { background:rgba(255,255,255,0.2); border:1px solid rgba(255,255,255,0.35); border-radius:20px; padding:4px 12px; font-size:12px; font-weight:600; }
    .section { margin-bottom:24px; }
    .section-title { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#64748B; margin-bottom:10px; }
    .info-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .info-box { background:#F8FAFC; border:1px solid #E2EBF4; border-radius:12px; padding:14px 16px; }
    .info-box .label { font-size:11px; color:#94A3B8; font-weight:600; margin-bottom:4px; }
    .info-box .value { font-size:14px; font-weight:700; color:#0F172A; }
    .result-box { border-radius:14px; padding:18px 20px; background:${statusBg}; border:1.5px solid ${statusColor}40; display:flex; gap:16px; align-items:center; }
    .result-dot { width:12px; height:12px; border-radius:50%; background:${statusColor}; flex-shrink:0; margin-top:2px; }
    .result-text { font-size:15px; font-weight:700; color:${statusColor}; }
    .result-sub { font-size:12px; color:#64748B; margin-top:4px; }
    .footer { margin-top:40px; padding-top:16px; border-top:1px solid #E2EBF4; font-size:11px; color:#94A3B8; text-align:center; }
    @media print {
      body { background:#fff; }
      .page { padding:20px 24px; box-shadow:none; }
      .no-print { display:none !important; }
    }
  </style>
</head>
<body>
<div class="page">
  <div class="header">
    <div>
      <h1>Báo cáo phân tích X-quang</h1>
      <p>AI X-Ray Diagnosis – DenseNet-121 / CheXpert 9 nhãn</p>
    </div>
    <span class="badge">${item.date}</span>
  </div>

  <div class="section">
    <div class="section-title">🗂️ Thông tin quét</div>
    <div class="info-grid">
      <div class="info-box">
        <div class="label">Tên file</div>
        <div class="value" style="font-family:monospace;font-size:12px">${item.filename}</div>
      </div>
      <div class="info-box">
        <div class="label">Thời gian</div>
        <div class="value">${item.date} · ${item.time}</div>
      </div>
      <div class="info-box">
        <div class="label">Tư thế chụp</div>
        <div class="value">${item.posture}</div>
      </div>
      <div class="info-box">
        <div class="label">Số nhãn phát hiện</div>
        <div class="value" style="color:${statusColor}">${item.detected} / ${item.total}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">🧠 Kết quả AI</div>
    <div class="result-box">
      ${thumbHtml ? `<div style="flex-shrink:0">${thumbHtml}</div>` : ""}
      <div>
        <div class="result-dot" style="display:inline-block;vertical-align:middle;margin-right:8px"></div>
        <span class="result-text">${statusText}</span>
        ${findingHtml}
        <div class="result-sub" style="margin-top:8px">Kết quả chỉ mang tính hỗ trợ, không thay thế chẩn đoán lâm sàng.</div>
      </div>
    </div>
  </div>

  <div class="footer">
    Tạo tự động bởi AI X-Ray Diagnosis &bull; ${new Date().toLocaleString("vi-VN")} &bull; Chỉ dùng tham khảo, không có giá trị chẩn đoán
  </div>

  <div class="no-print" style="text-align:center;margin-top:24px">
    <button onclick="window.print()" style="background:#2563EB;color:#fff;border:none;border-radius:10px;padding:12px 32px;font-size:14px;font-weight:700;cursor:pointer">In / Lưu PDF</button>
  </div>
</div>
</body></html>`;

  const w = window.open("", "_blank", "width=720,height=860");
  if (!w) return;
  w.document.write(html);
  w.document.close();
}

function HistoryView({ history, onClearHistory, onDeleteItem }) {
  const [confirmClear, setConfirmClear] = useState(false);

  if (!history || history.length === 0) {
    return <EmptyState icon={History} title="Chưa có lịch sử" sub="Lịch sử các lần phân tích sẽ hiển thị ở đây" />;
  }

  const abnormalCount = history.filter((h) => h.detected > 0).length;
  const normalCount   = history.filter((h) => h.detected === 0).length;

  return (
    <div className="p-5" style={{ animation: "medFadeInUp 0.4s ease" }}>
      {/* Stats + clear button row */}
      <div className="flex items-center gap-3 mb-5">
        <div className="grid grid-cols-3 gap-3 flex-1">
          {[
            { label: "Tổng lần quét", value: history.length, color: "#1D72F5", bg: "#EFF6FF", border: "#BFDBFE" },
            { label: "Có bất thường",  value: abnormalCount,  color: "#DC2626", bg: "#FEF2F2", border: "#FECACA" },
            { label: "Bình thường",    value: normalCount,    color: "#059669", bg: "#F0FDF4", border: "#BBF7D0" },
          ].map((s, i) => (
            <div
              key={i}
              className="rounded-2xl p-3 text-center"
              style={{ background: s.bg, border: `1.5px solid ${s.border}` }}
            >
              <div className="text-2xl" style={{ fontWeight: 800, color: s.color, fontVariantNumeric: "tabular-nums" }}>
                {s.value}
              </div>
              <div className="text-xs mt-0.5" style={{ color: s.color, fontWeight: 600 }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>

        {/* Clear all button */}
        {!confirmClear ? (
          <button
            onClick={() => setConfirmClear(true)}
            className="btn-icon flex-shrink-0 flex items-center gap-1.5 px-3 py-2 rounded-xl"
            style={{
              background: "#FEF2F2",
              border: "1px solid #FECACA",
              color: "#DC2626",
              fontSize: "12px",
              fontWeight: 600,
              transition: "background 0.15s, transform 0.15s",
            }}
            title="Xoá toàn bộ lịch sử"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Xoá tất cả</span>
          </button>
        ) : (
          <div
            className="flex-shrink-0 flex items-center gap-1.5 px-3 py-2 rounded-xl"
            style={{ background: "#FEF2F2", border: "1.5px solid #EF4444" }}
          >
            <span className="text-xs" style={{ color: "#DC2626", fontWeight: 700 }}>Xác nhận?</span>
            <button
              onClick={() => { onClearHistory(); setConfirmClear(false); }}
              className="text-xs px-2 py-0.5 rounded-lg"
              style={{ background: "#EF4444", color: "#fff", fontWeight: 700 }}
            >
              Xoá
            </button>
            <button
              onClick={() => setConfirmClear(false)}
              className="text-xs px-2 py-0.5 rounded-lg"
              style={{ background: "#E2E8F0", color: "#475569", fontWeight: 600 }}
            >
              Huỷ
            </button>
          </div>
        )}
      </div>

      <div className="space-y-3">
        {history.map((item, idx) => (
          <div
            key={item.id}
            className="history-card rounded-2xl overflow-hidden"
            style={{
              background: "#FFFFFF",
              border: "1px solid #E2EBF4",
              boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
              animation: `medSlideInRow 0.3s ${idx * 0.06}s ease both`,
            }}
          >
            <div className="flex items-stretch">
              <div
                className="relative flex-shrink-0 overflow-hidden flex items-center justify-center"
                style={{
                  width: "90px",
                  background: item.detected > 0
                    ? "linear-gradient(135deg, #1F2937 0%, #4B1418 100%)"
                    : "linear-gradient(135deg, #1F2937 0%, #134E4A 100%)",
                }}
              >
                {item.thumbnail ? (
                  <img
                    src={item.thumbnail}
                    alt={`Scan ${item.id}`}
                    className="w-full h-full object-cover"
                    style={{ filter: "contrast(1.1) saturate(0)", opacity: 0.9, minHeight: "90px" }}
                  />
                ) : (
                  <ImageIcon className="w-7 h-7" style={{ color: "rgba(255,255,255,0.35)" }} />
                )}
                <div
                  className="absolute top-1.5 left-1.5 w-4 h-4 rounded-full flex items-center justify-center"
                  style={{
                    background: item.detected > 0 ? "#EF4444" : "#10B981",
                    color: "#fff",
                    fontWeight: 800,
                    fontSize: "9px",
                    boxShadow: "0 1px 4px rgba(0,0,0,0.3)",
                  }}
                >
                  {item.detected > 0 ? "!" : "✓"}
                </div>
              </div>

              <div className="flex-1 px-4 py-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <Clock className="w-3 h-3 flex-shrink-0" style={{ color: "#94A3B8" }} />
                      <span className="text-xs" style={{ color: "#64748B" }}>
                        {item.date} · {item.time}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <FileText className="w-3 h-3" style={{ color: "#94A3B8" }} />
                      <span className="text-xs font-mono truncate" style={{ color: "#94A3B8", maxWidth: 160 }}>
                        {item.filename}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
                        <rect x="3" y="3" width="18" height="18" rx="2" stroke="#94A3B8" strokeWidth="2"/>
                        <path d="M9 9h6M9 12h6M9 15h4" stroke="#94A3B8" strokeWidth="1.5" strokeLinecap="round"/>
                      </svg>
                      <span className="text-xs" style={{ color: "#64748B" }}>
                        Tư thế:{" "}
                        <span style={{ fontWeight: 600, color: "#475569" }}>
                          {item.posture === "Auto" || item.posture === "auto"
                            ? "Tự động nhận diện"
                            : item.posture === "PA"
                            ? "PA (Posteroanterior)"
                            : item.posture === "AP"
                            ? "AP (Anteroposterior)"
                            : item.posture === "Lateral"
                            ? "Lateral (Bên)"
                            : item.posture}
                        </span>
                      </span>
                    </div>
                  </div>
                  {/* Action buttons */}
                  <div className="flex items-center gap-1 flex-shrink-0">
                    <button
                      onClick={() => exportPdf(item)}
                      className="btn-nav w-7 h-7 rounded-lg flex items-center justify-center"
                      style={{ background: "#EFF6FF", color: "#1D72F5", border: "1px solid #BFDBFE" }}
                      title="Xuất PDF"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => onDeleteItem(item.id)}
                      className="btn-nav w-7 h-7 rounded-lg flex items-center justify-center"
                      style={{ background: "#FEF2F2", color: "#EF4444", border: "1px solid #FECACA" }}
                      title="Xoá mục này"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {item.detected > 0 ? (
                  <div className="flex items-center gap-2 flex-wrap">
                    <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "#EF4444" }} />
                    <span className="text-xs" style={{ color: "#DC2626", fontWeight: 700 }}>
                      {item.topFindingVi || item.topFinding} ({item.topProb.toFixed(1)}%)
                    </span>
                    <span
                      className="text-xs px-1.5 py-0.5 rounded-full"
                      style={{ background: "#FEE2E2", color: "#DC2626", fontWeight: 600 }}
                    >
                      +{item.detected} bất thường
                    </span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "#10B981" }} />
                    <span className="text-xs" style={{ color: "#059669", fontWeight: 600 }}>
                      Không phát hiện bất thường
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ════════════════════ Helpers ════════════════════ */
function EmptyState({ icon: Icon, title, sub }) {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div
        className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
        style={{ background: "#F0F5FA", border: "1px solid #E2EBF4" }}
      >
        <Icon className="w-8 h-8" style={{ color: "#94A3B8" }} />
      </div>
      <div className="text-sm" style={{ color: "#475569", fontWeight: 600 }}>{title}</div>
      <div className="text-xs mt-1" style={{ color: "#94A3B8" }}>{sub}</div>
    </div>
  );
}

function absoluteUrl(url) {
  if (!url) return "";
  if (/^https?:/i.test(url)) return url;
  const base = (typeof window !== "undefined" && window.API_BASE) ? window.API_BASE : "";
  return base + url;
}
