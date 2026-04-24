import { useState, useEffect } from "react";

interface IndexScore {
  name: string;
  score: number;
  zone: string;
  confidence: number;
  price?: number;
  changePct?: number;
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

const ZONE_SHORT: Record<string, string> = {
  EXTREME_BUY: "ExtBuy",
  STRONG_BUY: "StrBuy",
  BUY_LEAN: "Buy",
  NEUTRAL: "Neutral",
  SELL_LEAN: "Sell",
  STRONG_SELL: "StrSell",
  EXTREME_SELL: "ExtSell",
};

/* Fallback hex colors for SVG / computed styles that need resolved values */
const ZONE_HEX_DARK: Record<string, string> = {
  EXTREME_BUY: "#00ff88",
  STRONG_BUY: "#22c55e",
  BUY_LEAN: "#4ade80",
  NEUTRAL: "#60a5fa",
  SELL_LEAN: "#fb923c",
  STRONG_SELL: "#ef4444",
  EXTREME_SELL: "#dc2626",
};

const ZONE_HEX_LIGHT: Record<string, string> = {
  EXTREME_BUY: "#059669",
  STRONG_BUY: "#16a34a",
  BUY_LEAN: "#22c55e",
  NEUTRAL: "#3b82f6",
  SELL_LEAN: "#ea580c",
  STRONG_SELL: "#dc2626",
  EXTREME_SELL: "#b91c1c",
};

interface IndexScoresProps {
  indices: Record<string, IndexScore>;
}

export default function IndexScores({ indices }: IndexScoresProps) {
  const entries = Object.entries(indices);
  const [isLight, setIsLight] = useState(false);

  /* Detect theme changes */
  useEffect(() => {
    const check = () => setIsLight(document.documentElement.getAttribute("data-theme") === "light");
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  const ZONE_HEX = isLight ? ZONE_HEX_LIGHT : ZONE_HEX_DARK;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      {entries.map(([key, idx]) => {
        const color = ZONE_COLORS[idx.zone] || "var(--color-text-muted)";
        const hex = ZONE_HEX[idx.zone] || (isLight ? "#94a3b8" : "#64748b");
        const shortZone = ZONE_SHORT[idx.zone] || idx.zone;
        const scorePct = Math.max(0, Math.min(100, idx.score));

        return (
          <div
            key={key}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "0.65rem 0.9rem",
              background: "var(--color-bg-card)",
              borderRadius: "var(--radius-md)",
              border: `1px solid var(--color-border-subtle)`,
              borderLeftWidth: "3px",
              borderLeftColor: hex,
              transition: "all var(--transition-fast)",
              cursor: "default",
              position: "relative",
              overflow: "hidden",
            }}
            onMouseEnter={(e) => {
              const el = e.currentTarget as HTMLElement;
              el.style.background = "var(--color-bg-elevated)";
              el.style.borderColor = "var(--color-border)";
              el.style.borderLeftColor = hex;
              el.style.transform = "translateX(2px)";
            }}
            onMouseLeave={(e) => {
              const el = e.currentTarget as HTMLElement;
              el.style.background = "var(--color-bg-card)";
              el.style.borderColor = "var(--color-border-subtle)";
              el.style.borderLeftColor = hex;
              el.style.transform = "translateX(0)";
            }}
          >
            {/* Subtle score bar background */}
            <div
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                bottom: 0,
                width: `${scorePct}%`,
                background: `${hex}08`,
                transition: "width var(--transition-normal)",
                pointerEvents: "none",
              }}
            />

            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", position: "relative", zIndex: 1 }}>
              <span style={{ fontSize: "0.82rem", color: "var(--color-text-secondary)", fontWeight: 500, minWidth: "80px" }}>
                {idx.name}
              </span>
              {idx.changePct != null && (
                <span
                  style={{
                    fontSize: "0.7rem",
                    fontFamily: "var(--font-mono)",
                    color: idx.changePct >= 0 ? "var(--color-emerald)" : "var(--color-crimson)",
                  }}
                >
                  {idx.changePct >= 0 ? "+" : ""}{idx.changePct.toFixed(2)}%
                </span>
              )}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", position: "relative", zIndex: 1 }}>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "1rem",
                  fontWeight: 700,
                  color,
                }}
              >
                {idx.score.toFixed(1)}
              </span>
              <span
                style={{
                  fontSize: "0.65rem",
                  fontWeight: 600,
                  color,
                  textTransform: "uppercase" as const,
                  letterSpacing: "0.5px",
                  padding: "2px 6px",
                  background: `${hex}15`,
                  borderRadius: "var(--radius-sm)",
                }}
              >
                {shortZone}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
