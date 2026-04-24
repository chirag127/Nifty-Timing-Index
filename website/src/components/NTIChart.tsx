import { useEffect, useRef, useState } from "react";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

interface ScorePoint {
  date: string;
  score: number;
  nifty_price?: number;
  zone?: string;
}

interface NTIChartProps {
  scores: ScorePoint[];
  showNifty?: boolean;
  height?: number;
}

export default function NTIChart({ scores, showNifty = true, height = 300 }: NTIChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const [theme, setTheme] = useState<string>("dark");

  /* Listen for theme changes */
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setTheme(document.documentElement.getAttribute("data-theme") || "dark");
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    setTheme(document.documentElement.getAttribute("data-theme") || "dark");
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!canvasRef.current || scores.length === 0) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    /* Resolve CSS variables for chart theming */
    const style = getComputedStyle(document.documentElement);
    const gold = style.getPropertyValue("--color-gold").trim() || "#f59e0b";
    const blueSteel = style.getPropertyValue("--color-blue-steel").trim() || "#3b82f6";
    const textMuted = style.getPropertyValue("--color-text-muted").trim() || "#556580";
    const textSecondary = style.getPropertyValue("--color-text-secondary").trim() || "#8c9bb5";
    const border = style.getPropertyValue("--color-border").trim() || "#1e3050";
    const borderSubtle = style.getPropertyValue("--color-border-subtle").trim() || "#152240";
    const bgCard = style.getPropertyValue("--color-bg-card").trim() || "#111a2e";
    const bgElevated = style.getPropertyValue("--color-bg-elevated").trim() || "#162038";
    const textPrimary = style.getPropertyValue("--color-text-primary").trim() || "#eef2f7";
    const fontMono = style.getPropertyValue("--font-mono").trim() || "'JetBrains Mono', monospace";
    const fontBody = style.getPropertyValue("--font-body").trim() || "'DM Sans', sans-serif";

    const labels = scores.map((s) => {
      const d = new Date(s.date);
      return d.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
    });

    const datasets: any[] = [
      {
        label: "NTI Score",
        data: scores.map((s) => s.score),
        borderColor: gold,
        backgroundColor: `${gold}14`,
        fill: true,
        tension: 0.3,
        pointRadius: scores.length > 30 ? 0 : 3,
        pointHoverRadius: 5,
        pointBackgroundColor: gold,
        pointBorderColor: bgCard,
        pointBorderWidth: 2,
        borderWidth: 2.5,
        yAxisID: "y",
      },
    ];

    if (showNifty) {
      datasets.push({
        label: "Nifty 50",
        data: scores.map((s) => s.nifty_price ?? null),
        borderColor: blueSteel,
        backgroundColor: "transparent",
        borderDash: [5, 5],
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 1.5,
        yAxisID: "y1",
      });
    }

    chartRef.current = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            labels: {
              color: textSecondary,
              font: { size: 11, family: fontBody },
              usePointStyle: true,
              pointStyle: "circle",
            },
          },
          tooltip: {
            backgroundColor: bgElevated,
            borderColor: border,
            borderWidth: 1,
            titleColor: textPrimary,
            bodyColor: textSecondary,
            titleFont: { family: fontBody, weight: 600 },
            bodyFont: { family: fontMono, size: 12 },
            cornerRadius: 8,
            padding: 10,
          },
        },
        scales: {
          x: {
            ticks: { color: textMuted, maxTicksLimit: 8, font: { size: 10, family: fontBody } },
            grid: { color: `${border}40` },
          },
          y: {
            position: "left",
            min: 0,
            max: 100,
            ticks: { color: gold, font: { size: 10, family: fontMono } },
            grid: { color: `${border}40` },
            title: { display: true, text: "NTI Score", color: gold, font: { size: 11, family: fontBody, weight: 500 } },
          },
          ...(showNifty
            ? {
                y1: {
                  position: "right" as const,
                  ticks: { color: blueSteel, font: { size: 10, family: fontMono } },
                  grid: { drawOnChartArea: false },
                  title: { display: true, text: "Nifty 50", color: blueSteel, font: { size: 11, family: fontBody, weight: 500 } },
                },
              }
            : {}),
        },
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [scores, showNifty, theme]);

  return (
    <div style={{ position: "relative", height }}>
      <canvas ref={canvasRef} />
    </div>
  );
}
