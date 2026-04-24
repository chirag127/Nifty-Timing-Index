import { useState, useMemo, useEffect } from "react";

interface Stock {
  rank: number;
  symbol: string;
  name: string;
  sector: string;
  is_psu: boolean;
  market_cap_cr: number;
  current_price: number;
  pe: number;
  pb: number;
  roe_pct: number;
  debt_equity: number;
  dividend_yield_pct: number;
  analyst_buy_pct: number;
  composite_score: number;
  value_score: number;
  quality_score: number;
  analyst_score: number;
  psu_boost: number;
  warnings: string[];
}

interface StockTableProps {
  stocks: Stock[];
  psuOnly?: boolean;
}

type SortKey = keyof Stock;
type SortDir = "asc" | "desc";

export default function StockTable({ stocks, psuOnly = false }: StockTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("composite_score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [filter, setFilter] = useState("");
  const [isLight, setIsLight] = useState(false);

  /* Detect theme changes */
  useEffect(() => {
    const check = () => setIsLight(document.documentElement.getAttribute("data-theme") === "light");
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  const filtered = useMemo(() => {
    let list = stocks;
    if (psuOnly) list = list.filter((s) => s.is_psu);
    if (filter) {
      const q = filter.toLowerCase();
      list = list.filter(
        (s) =>
          s.symbol.toLowerCase().includes(q) ||
          s.name.toLowerCase().includes(q) ||
          s.sector.toLowerCase().includes(q)
      );
    }
    return [...list].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      return sortDir === "asc"
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
  }, [stocks, psuOnly, filter, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const SortHeader = ({ label, k }: { label: string; k: SortKey }) => (
    <th
      onClick={() => toggleSort(k)}
      style={{
        cursor: "pointer",
        userSelect: "none",
        color: sortKey === k ? "var(--color-gold)" : "var(--color-text-muted)",
        whiteSpace: "nowrap",
        transition: "color var(--transition-fast)",
      }}
    >
      {label}
      {sortKey === k && (sortDir === "asc" ? " ↑" : " ↓")}
    </th>
  );

  const fmt = (v: number | undefined, decimals = 1) =>
    v != null ? v.toFixed(decimals) : "—";

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ marginBottom: "0.75rem", display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <input
          type="text"
          placeholder="Search symbol, name, sector..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{
            padding: "0.45rem 0.85rem",
            background: "var(--color-bg-elevated)",
            border: "1px solid var(--color-border-subtle)",
            borderRadius: "var(--radius-sm)",
            color: "var(--color-text-primary)",
            fontSize: "0.85rem",
            fontFamily: "var(--font-body)",
            width: "280px",
            maxWidth: "100%",
            transition: "border-color var(--transition-fast)",
            outline: "none",
          }}
          onFocus={(e) => {
            (e.currentTarget as HTMLInputElement).style.borderColor = "var(--color-gold)";
          }}
          onBlur={(e) => {
            (e.currentTarget as HTMLInputElement).style.borderColor = "var(--color-border-subtle)";
          }}
        />
        <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {filtered.length} stocks
        </span>
      </div>

      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "0.82rem",
        }}
      >
        <thead>
          <tr style={{ borderBottom: `2px solid var(--color-border)` }}>
            <SortHeader label="#" k="rank" />
            <SortHeader label="Symbol" k="symbol" />
            <th>Sector</th>
            <SortHeader label="PE" k="pe" />
            <SortHeader label="PB" k="pb" />
            <SortHeader label="ROE" k="roe_pct" />
            <SortHeader label="Div %" k="dividend_yield_pct" />
            <SortHeader label="Analyst" k="analyst_buy_pct" />
            <SortHeader label="Score" k="composite_score" />
            <th>MCap</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((s) => (
            <tr
              key={s.symbol}
              style={{
                borderBottom: `1px solid var(--color-border-subtle)`,
                transition: "background var(--transition-fast)",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.background = "var(--color-bg-elevated)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.background = "transparent";
              }}
            >
              <td style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>{s.rank}</td>
              <td>
                <a
                  href={`/stocks/${s.symbol}`}
                  style={{ color: "var(--color-gold)", fontWeight: 600, textDecoration: "none", transition: "color var(--transition-fast)" }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLElement).style.color = "var(--color-gold-bright)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.color = "var(--color-gold)";
                  }}
                >
                  {s.symbol}
                </a>
                {s.is_psu && (
                  <span
                    style={{
                      marginLeft: "4px",
                      fontSize: "0.6rem",
                      padding: "1px 5px",
                      background: "var(--color-gold-glow)",
                      color: "var(--color-gold-bright)",
                      borderRadius: "var(--radius-sm)",
                      fontWeight: 600,
                      border: `1px solid ${isLight ? "rgba(245,158,11,0.25)" : "rgba(245,158,11,0.2)"}`,
                    }}
                  >
                    PSU
                  </span>
                )}
              </td>
              <td style={{ color: "var(--color-text-secondary)", fontSize: "0.75rem", maxWidth: "120px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" as const }}>
                {s.sector}
              </td>
              <td style={{ fontFamily: "var(--font-mono)", color: s.pe < 15 ? "var(--color-emerald)" : "var(--color-text-primary)" }}>
                {fmt(s.pe)}
              </td>
              <td style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{fmt(s.pb)}</td>
              <td style={{ fontFamily: "var(--font-mono)", color: s.roe_pct >= 15 ? "var(--color-emerald)" : "var(--color-text-muted)" }}>
                {fmt(s.roe_pct)}%
              </td>
              <td style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{fmt(s.dividend_yield_pct)}%</td>
              <td style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                {s.analyst_buy_pct != null ? `${fmt(s.analyst_buy_pct, 0)}%` : "—"}
              </td>
              <td>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div
                    style={{
                      width: "48px",
                      height: "6px",
                      background: "var(--color-bg-elevated)",
                      borderRadius: "3px",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${s.composite_score}%`,
                        height: "100%",
                        background: "var(--color-gold)",
                        borderRadius: "3px",
                        transition: "width var(--transition-normal)",
                      }}
                    />
                  </div>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontWeight: 700,
                      color: "var(--color-gold)",
                    }}
                  >
                    {s.composite_score}
                  </span>
                </div>
              </td>
              <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>
                {s.market_cap_cr >= 1000
                  ? `₹${(s.market_cap_cr / 1000).toFixed(1)}K Cr`
                  : `₹${s.market_cap_cr} Cr`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
