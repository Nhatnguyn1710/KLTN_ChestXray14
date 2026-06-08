import { ScanLine, Cpu, Shield } from "lucide-react";

export function XRayHeader({ health }) {
  const cnnReady = health?.cnn === true;
  const statusText = health?.text || "Đang kiểm tra...";
  const dotColor = cnnReady ? "#22C55E" : health?.loading ? "#F59E0B" : "#EF4444";

  return (
    <header
      className="flex items-center justify-between px-6 flex-shrink-0"
      style={{
        background: "linear-gradient(90deg, #2563EB 0%, #3B82F6 50%, #4F8EF6 100%)",
        height: "64px",
        borderBottom: "1px solid rgba(255,255,255,0.15)",
        boxShadow: "0 2px 12px rgba(37,99,235,0.18)",
      }}
    >
      {/* Left: Brand */}
      <div className="flex items-center gap-3">
        <div
          className="relative w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{
            background: "rgba(255,255,255,0.18)",
            border: "1px solid rgba(255,255,255,0.35)",
            backdropFilter: "blur(6px)",
          }}
        >
          <ScanLine className="w-5 h-5" style={{ color: "#FFFFFF" }} />
        </div>
        <div>
          <div
            className="text-base"
            style={{ color: "#FFFFFF", fontWeight: 700, letterSpacing: "-0.01em", lineHeight: 1.15 }}
          >
            AI X-Ray Diagnosis
          </div>
          <div className="text-xs" style={{ color: "rgba(255,255,255,0.78)" }}>
            Nền tảng hỗ trợ đọc X-quang tối ưu cho bác sĩ
          </div>
        </div>
      </div>

      {/* Right: badges */}
      <div className="flex items-center gap-2.5">
        <div
          className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg"
          style={{
            background: "rgba(255,255,255,0.15)",
            border: "1px solid rgba(255,255,255,0.28)",
          }}
        >
          <Cpu className="w-3.5 h-3.5" style={{ color: "#FFFFFF" }} />
          <span className="text-xs" style={{ color: "#FFFFFF", fontWeight: 600 }}>
            DenseNet-121{health.modelVersion ? ` · ${health.modelVersion}` : ""}
          </span>
        </div>

        <div
          className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg"
          style={{
            background: "rgba(255,255,255,0.15)",
            border: "1px solid rgba(255,255,255,0.28)",
          }}
        >
          <Shield className="w-3.5 h-3.5" style={{ color: "#FFFFFF" }} />
          <span className="text-xs" style={{ color: "#FFFFFF", fontWeight: 600 }}>
            NIH ChestX-ray14
          </span>
        </div>

        <div
          className="flex items-center gap-2 px-3 py-1.5 rounded-full"
          style={{
            background: "#FFFFFF",
            border: "1px solid rgba(255,255,255,0.6)",
            boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
          }}
        >
          <div
            className="w-2 h-2 rounded-full"
            style={{
              background: dotColor,
              boxShadow: `0 0 6px ${dotColor}`,
              animation: "medPulse 2s ease-in-out infinite",
            }}
          />
          <span className="text-xs" style={{ color: "#0F172A", fontWeight: 600 }}>
            {statusText}
          </span>
        </div>
      </div>
    </header>
  );
}
