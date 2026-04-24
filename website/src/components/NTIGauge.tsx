import { useMemo, useId, useState, useEffect } from "react";

interface NTIGaugeProps {
  score: number;
  zone: string;
  confidence: number;
  prevScore?: number;
  size?: number;
}

const ZONE_COLORS: Record<string, string> = {
  EXTREME_BUY: "var(--color-extreme-buy)",
  STRONG_BUY: "var(--color-strong-buy)",
  BUY_LEAN: "var(--color-buy-lean)",
  NEUTRAL: "var(--color-neutral)",
  SELL_LEAN: "var(--color-sell-lean)",
  STRONG_SELL: "var(--color-strong-sell)",
  EXTREME_SELL: "var(--color-extreme-sell)",
};

/* Hardcoded fallback colors for SVG gradient stops (can't use CSS vars in gradient stops) */
const ZONE_HEX_DARK: Record<string, string> = {
  EXTREME_BUY: "#00ff88",
  STRONG_BUY: "#22c55e",
  BUY_LEAN: "#4ade80",
  NEUTRAL: "#60a5fa",
  SELL_LEAN: "#fb923c",
  STRONG_SELL: "#ef4444",
  EXTREME_SELL: "#dc2626",
};

/* Lighter/stronger variants for light theme readability */
const ZONE_HEX_LIGHT: Record<string, string> = {
  EXTREME_BUY: "#059669",
  STRONG_BUY: "#16a34a",
  BUY_LEAN: "#22c55e",
  NEUTRAL: "#3b82f6",
  SELL_LEAN: "#ea580c",
  STRONG_SELL: "#dc2626",
  EXTREME_SELL: "#b91c1c",
};

const ZONE_LABELS: Record<string, string> = {
  EXTREME_BUY: "Extreme Buy",
  STRONG_BUY: "Strong Buy",
  BUY_LEAN: "Buy Lean",
  NEUTRAL: "Neutral",
  SELL_LEAN: "Sell Lean",
  STRONG_SELL: "Strong Sell",
  EXTREME_SELL: "Extreme Sell",
};

export default function NTIGauge({ score, zone, confidence, prevScore, size = 280 }: NTIGaugeProps) {
  const uid = useId().replace(/:/g, "");
  const [isLight, setIsLight] = useState(false);

  /* Detect theme changes */
  useEffect(() => {
    const check = () => setIsLight(document.documentElement.getAttribute("data-theme") === "light");
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  const color = ZONE_COLORS[zone] || "var(--color-text-muted)";
  const ZONE_HEX = isLight ? ZONE_HEX_LIGHT : ZONE_HEX_DARK;
  const hex = ZONE_HEX[zone] || "#64748b";
  const bgElevated = isLight ? "var(--color-border-subtle)" : "var(--color-bg-elevated)";
  const label = ZONE_LABELS[zone] || "Unknown";

  const scoreDelta = useMemo(() => {
    if (prevScore == null) return null;
    return score - prevScore;
  }, [score, prevScore]);

  const arcLength = 251.3;
  const offset = arcLength - (arcLength * Math.max(0, Math.min(100, score)) / 100);

  return (
    <div style={{ width: size, margin: "0 auto", textAlign: "center" }}>
      <svg viewBox="0 0 200 130" style={{ width: "100%" }}>
        <defs>
          {/* Gradient for the arc based on zone */}
          <linearGradient id={`gauge-grad-${uid}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={isLight ? "#059669" : "#00ff88"} stopOpacity="0.3" />
            <stop offset="50%" stopColor={hex} stopOpacity="0.6" />
            <stop offset="100%" stopColor={hex} />
          </linearGradient>
          {/* Glow filter */}
          <filter id={`gauge-glow-${uid}`} x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background arc */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke={bgElevated}
          strokeWidth="14"
          strokeLinecap="round"
        />
        {/* Colored arc with glow */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={arcLength}
          strokeDashoffset={offset}
          filter={`url(#gauge-glow-${uid})`}
          style={{
            transition: "stroke-dashoffset 0.8s ease, stroke 0.5s ease",
          }}
        />
        {/* Score text */}
        <text
          x="100"
          y="88"
          textAnchor="middle"
          fill={color}
          fontFamily="var(--font-mono)"
          fontSize="36"
          fontWeight="700"
        >
          {score.toFixed(1)}
        </text>
        {/* Zone label */}
        <text
          x="100"
          y="112"
          textAnchor="middle"
          fill={color}
          fontFamily="var(--font-body)"
          fontSize="11"
          fontWeight="600"
          letterSpacing="0.5"
        >
          {label.toUpperCase().replace(/_/g, " ")}
        </text>
        {/* Tick marks */}
        {[0, 15, 30, 45, 55, 69, 84, 100].map((tick) => {
          const angle = Math.PI - (tick / 100) * Math.PI;
          const x1 = 100 + 68 * Math.cos(angle);
          const y1 = 100 - 68 * Math.sin(angle);
          const x2 = 100 + 80 * Math.cos(angle);
          const y2 = 100 - 80 * Math.sin(angle);
          return (
            <line
              key={tick}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="var(--color-border)"
              strokeWidth="1.5"
            />
          );
        })}
      </svg>

      {/* Confidence */}
      <div style={{ marginTop: "-8px", fontSize: "0.75rem", color: "var(--color-text-muted)" }}>
        Confidence:{" "}
        <span
          style={{
            color: confidence >= 80 ? "var(--color-emerald)" : confidence >= 60 ? "var(--color-gold)" : "var(--color-crimson)",
            fontWeight: 600,
            fontFamily: "var(--font-mono)",
          }}
        >
          {confidence}%
        </span>
      </div>

      {/* Score delta */}
      {scoreDelta != null && (
        <div
          style={{
            marginTop: "4px",
            fontSize: "0.8rem",
            fontFamily: "var(--font-mono)",
            color: scoreDelta > 0 ? "var(--color-crimson)" : scoreDelta < 0 ? "var(--color-emerald)" : "var(--color-text-muted)",
          }}
        >
          {scoreDelta > 0 ? "↑" : scoreDelta < 0 ? "↓" : "→"}{" "}
          {Math.abs(scoreDelta).toFixed(1)} pts
        </div>
      )}
    </div>
  );
}
