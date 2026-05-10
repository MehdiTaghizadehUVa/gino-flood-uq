"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type RunData = {
  run_id: string;
  status: string;
  spec: Record<string, unknown>;
  failure_reason: string | null;
  pinned: boolean;
  created_at: string;
  updated_at: string;
};

type Artifact = {
  artifact_id: string;
  content_type: string;
  size_bytes: number;
};

type Summary = {
  label: string;
  n_members: number;
  n_time: number;
  n_cells: number;
  lead_time_hours: number[];
  mean_wd_overall_m: number;
  max_mean_wd_m: number;
  mean_spread_wd_m: number;
  mean_q05_wd_m: number;
  mean_q50_wd_m: number;
  mean_q95_wd_m: number;
  peak_mean_wd_by_time_m: number[];
  "inundated_cells_by_time_gt_0.05m": number[];
  max_inundated_cells_gt_0_05m?: number;
  "max_inundated_cells_gt_0.05m"?: number;
  mean_arrival_time_hours_gt_0_05m?: number | null;
  "mean_arrival_time_hours_gt_0.05m"?: number | null;
  isotonic_calibration_applied: boolean;
  [key: string]: unknown;
};

type ProductKey = "mean" | "spread" | "p95" | "p_gt_0p30m";

const PRODUCT_LABELS: Record<ProductKey, string> = {
  mean: "Mean WD (m)",
  spread: "Ensemble spread (m)",
  p95: "WD p95 (m)",
  p_gt_0p30m: "P(WD > 0.30 m)",
};

function formatNumber(value: unknown, digits = 3): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function exceedanceEntries(summary: Summary | null): { threshold: string; value: number }[] {
  if (!summary) return [];
  const entries: { threshold: string; value: number }[] = [];
  for (const [key, raw] of Object.entries(summary)) {
    const match = key.match(/^p_wd_gt_([0-9eE+\-.]+)m_mean$/);
    if (!match) continue;
    const value = typeof raw === "number" ? raw : Number(raw);
    if (!Number.isFinite(value)) continue;
    entries.push({ threshold: match[1], value });
  }
  return entries.sort((a, b) => Number(a.threshold) - Number(b.threshold));
}

function parseHydrograph(csvText: string): { t_h: number[]; stage: number[]; precip: number[] } {
  const lines = csvText.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) return { t_h: [], stage: [], precip: [] };
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const stageIdx = header.findIndex((h) => h.includes("stage"));
  const precipIdx = header.findIndex((h) => h.includes("precip"));
  const timeIdx = header.findIndex((h) => h.startsWith("time") || h === "t" || h === "hour" || h === "hours");
  if (stageIdx < 0 || precipIdx < 0) return { t_h: [], stage: [], precip: [] };
  const t_h: number[] = [];
  const stage: number[] = [];
  const precip: number[] = [];
  lines.slice(1).forEach((line, idx) => {
    const cols = line.split(",");
    const stageVal = Number(cols[stageIdx]);
    const precipVal = Number(cols[precipIdx]);
    if (!Number.isFinite(stageVal) || !Number.isFinite(precipVal)) return;
    let t = idx * (1200 / 3600);
    if (timeIdx >= 0) {
      const rawT = Number(cols[timeIdx]);
      if (Number.isFinite(rawT)) {
        const lower = header[timeIdx];
        t = lower.includes("hour") ? rawT : rawT / 3600;
      }
    }
    t_h.push(t);
    stage.push(stageVal);
    precip.push(precipVal);
  });
  return { t_h, stage, precip };
}

function LineSparkline({
  series,
  color,
  height = 80,
  yLabel,
}: {
  series: { t: number; y: number }[];
  color: string;
  height?: number;
  yLabel: string;
}) {
  if (series.length < 2) {
    return <p style={{ color: "#64748b", fontStyle: "italic" }}>Not enough samples to plot.</p>;
  }
  const padding = { left: 38, right: 12, top: 8, bottom: 22 };
  const innerW = 480;
  const innerH = height;
  const totalW = innerW + padding.left + padding.right;
  const totalH = innerH + padding.top + padding.bottom;
  const tMin = series[0].t;
  const tMax = series[series.length - 1].t;
  const yMin = Math.min(...series.map((s) => s.y));
  const yMax = Math.max(...series.map((s) => s.y));
  const yRange = yMax - yMin || 1;
  const tRange = tMax - tMin || 1;
  const points = series
    .map((s) => {
      const x = padding.left + ((s.t - tMin) / tRange) * innerW;
      const y = padding.top + (1 - (s.y - yMin) / yRange) * innerH;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg width={totalW} height={totalH} role="img" aria-label={yLabel} style={{ maxWidth: "100%" }}>
      <rect x={padding.left} y={padding.top} width={innerW} height={innerH} fill="#f1f5f9" stroke="#cbd5e1" />
      <polyline fill="none" stroke={color} strokeWidth={2} points={points} />
      <text x={padding.left} y={padding.top - 2} fontSize={10} fill="#475569">
        {yLabel}: {yMin.toFixed(3)} – {yMax.toFixed(3)}
      </text>
      <text x={padding.left} y={totalH - 6} fontSize={10} fill="#475569">
        {tMin.toFixed(2)} h
      </text>
      <text x={padding.left + innerW - 38} y={totalH - 6} fontSize={10} fill="#475569">
        {tMax.toFixed(2)} h
      </text>
    </svg>
  );
}

function findMatchingPng(
  artifacts: Artifact[],
  view: "calibrated" | "raw",
  product: ProductKey,
): { artifactId: string; tIndex: number }[] {
  const prefix = `${view}_${product}_t`;
  const matches: { artifactId: string; tIndex: number }[] = [];
  for (const artifact of artifacts) {
    if (!artifact.artifact_id.startsWith(prefix)) continue;
    if (!artifact.artifact_id.endsWith(".png")) continue;
    const stripped = artifact.artifact_id.slice(prefix.length, artifact.artifact_id.length - 4);
    const tIndex = parseInt(stripped, 10);
    if (Number.isFinite(tIndex)) {
      matches.push({ artifactId: artifact.artifact_id, tIndex });
    }
  }
  return matches.sort((a, b) => a.tIndex - b.tIndex);
}

export default function RunDetails({ params }: { params: { runId: string } }) {
  const runId = params.runId;
  const [run, setRun] = useState<RunData | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [error, setError] = useState("");
  const [calibratedSummary, setCalibratedSummary] = useState<Summary | null>(null);
  const [rawSummary, setRawSummary] = useState<Summary | null>(null);
  const [forcing, setForcing] = useState<{ t_h: number[]; stage: number[]; precip: number[] } | null>(null);
  const [view, setView] = useState<"calibrated" | "raw">("calibrated");
  const [product, setProduct] = useState<ProductKey>("mean");
  const [timeSlot, setTimeSlot] = useState(0);

  const loadJson = useCallback(
    async (artifactId: string) => {
      const res = await fetch(`/api/runs/${runId}/artifacts/${artifactId}`, { cache: "no-store" });
      if (!res.ok) return null;
      try {
        return (await res.json()) as Summary;
      } catch {
        return null;
      }
    },
    [runId],
  );

  const refresh = useCallback(async () => {
    const runRes = await fetch(`/api/runs/${runId}`, { cache: "no-store" });
    if (!runRes.ok) {
      const detail = await runRes.json().catch(() => ({}));
      setError(detail.detail ?? "Could not load run.");
      return;
    }
    setError("");
    setRun(await runRes.json());
    const artifactRes = await fetch(`/api/runs/${runId}/artifacts`, { cache: "no-store" });
    if (artifactRes.ok) {
      const list = (await artifactRes.json()) as Artifact[];
      setArtifacts(list);
      const ids = new Set(list.map((a) => a.artifact_id));
      if (ids.has("calibrated_summary.json")) {
        setCalibratedSummary((prev) => prev ?? null);
        loadJson("calibrated_summary.json").then(setCalibratedSummary).catch(() => undefined);
      }
      if (ids.has("raw_summary.json")) {
        loadJson("raw_summary.json").then(setRawSummary).catch(() => undefined);
      }
      if (ids.has("forcing.csv") && forcing === null) {
        fetch(`/api/runs/${runId}/artifacts/forcing.csv`, { cache: "no-store" })
          .then((res) => (res.ok ? res.text() : null))
          .then((text) => {
            if (text) setForcing(parseHydrograph(text));
          })
          .catch(() => undefined);
      }
    }
  }, [runId, loadJson, forcing]);

  useEffect(() => {
    refresh().catch((exc) => setError(String(exc)));
    const timer = setInterval(() => refresh().catch(() => undefined), 10000);
    return () => clearInterval(timer);
  }, [refresh]);

  const summary = view === "calibrated" ? calibratedSummary : rawSummary;
  const otherSummary = view === "calibrated" ? rawSummary : calibratedSummary;

  const pngSlots = useMemo(() => findMatchingPng(artifacts, view, product), [artifacts, view, product]);
  const safeTimeSlot = pngSlots.length === 0 ? 0 : Math.min(timeSlot, pngSlots.length - 1);
  const currentPng = pngSlots[safeTimeSlot];
  const currentLeadHours = useMemo(() => {
    if (!summary?.lead_time_hours || !currentPng) return null;
    const idx = currentPng.tIndex - 1;
    return summary.lead_time_hours[idx] ?? null;
  }, [summary, currentPng]);

  const arrivalTimeKey = "mean_arrival_time_hours_gt_0.05m";
  const maxInundatedKey = "max_inundated_cells_gt_0.05m";
  const exceedance = exceedanceEntries(summary);
  const otherExceedance = exceedanceEntries(otherSummary);
  const hasAnimation = artifacts.some((a) => a.artifact_id === "calibrated_mean_wd_animation.gif");
  const hasHdf5 = artifacts.some((a) => a.artifact_id === "forecast_members.h5");

  return (
    <main className="shell">
      <a className="back" href="/">← Back to runs</a>
      <header className="hero">
        <h1>Run {runId.slice(0, 10)}</h1>
        {run && (
          <p className="meta">
            <span className={`status status-${run.status}`}>{run.status}</span>
            <span> · created {new Date(run.created_at).toLocaleString()}</span>
            {run.pinned && <span className="pin"> · pinned</span>}
          </p>
        )}
      </header>
      {error && <p className="error">{error}</p>}
      {run?.failure_reason && (
        <section className="panel error-panel">
          <h2>Run failed</h2>
          <p>{run.failure_reason}</p>
        </section>
      )}

      {summary && (
        <section className="cards">
          <article className="card">
            <span className="card-label">Peak mean WD</span>
            <span className="card-value">{formatNumber(summary.max_mean_wd_m, 3)} m</span>
          </article>
          <article className="card">
            <span className="card-label">Max inundated cells (&gt;0.05 m)</span>
            <span className="card-value">
              {formatNumber(
                summary[maxInundatedKey] ?? summary.max_inundated_cells_gt_0_05m,
                0,
              )}
            </span>
          </article>
          <article className="card">
            <span className="card-label">Mean arrival (&gt;0.05 m)</span>
            <span className="card-value">
              {summary[arrivalTimeKey] === null || summary[arrivalTimeKey] === undefined
                ? "—"
                : `${formatNumber(summary[arrivalTimeKey], 2)} h`}
            </span>
          </article>
          <article className="card">
            <span className="card-label">Mean ensemble spread</span>
            <span className="card-value">{formatNumber(summary.mean_spread_wd_m, 3)} m</span>
          </article>
        </section>
      )}

      <section className="panel">
        <div className="row-between">
          <h2>Forecast maps</h2>
          <div className="toggle">
            <button
              type="button"
              className={view === "calibrated" ? "active" : ""}
              onClick={() => setView("calibrated")}
            >
              Calibrated
            </button>
            <button
              type="button"
              className={view === "raw" ? "active" : ""}
              onClick={() => setView("raw")}
            >
              Raw FGN
            </button>
          </div>
        </div>
        <div className="product-row">
          {(Object.keys(PRODUCT_LABELS) as ProductKey[]).map((p) => (
            <button
              key={p}
              type="button"
              className={`chip ${product === p ? "active" : ""}`}
              onClick={() => {
                setProduct(p);
                setTimeSlot(0);
              }}
            >
              {PRODUCT_LABELS[p]}
            </button>
          ))}
        </div>
        {currentPng ? (
          <>
            <img
              key={currentPng.artifactId}
              src={`/api/runs/${runId}/artifacts/${currentPng.artifactId}`}
              alt={`${view} ${product} at index ${currentPng.tIndex}`}
              className="map-image"
            />
            {pngSlots.length > 1 && (
              <div className="time-slider">
                <input
                  type="range"
                  min={0}
                  max={pngSlots.length - 1}
                  value={safeTimeSlot}
                  step={1}
                  onChange={(event) => setTimeSlot(Number(event.target.value))}
                />
                <span>
                  Slot {safeTimeSlot + 1} / {pngSlots.length}
                  {currentLeadHours !== null ? ` · lead ${currentLeadHours.toFixed(2)} h` : ""}
                </span>
              </div>
            )}
          </>
        ) : (
          <p className="muted">
            Map products are generated after inference completes. Once the run is COMPLETED, snapshots
            for {PRODUCT_LABELS[product]} ({view}) will appear here.
          </p>
        )}
        {hasAnimation && view === "calibrated" && (
          <a className="anim-link" href={`/api/runs/${runId}/artifacts/calibrated_mean_wd_animation.gif`}>
            Download calibrated mean-WD animation (GIF)
          </a>
        )}
      </section>

      <section className="panel">
        <h2>UQ summary</h2>
        {!summary && <p className="muted">Summary JSON will appear once postprocessing finishes.</p>}
        {summary && (
          <table className="summary-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>{view === "calibrated" ? "Calibrated" : "Raw FGN"}</th>
                {otherSummary && <th>{view === "calibrated" ? "Raw FGN" : "Calibrated"}</th>}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Members</td>
                <td>{summary.n_members}</td>
                {otherSummary && <td>{otherSummary.n_members}</td>}
              </tr>
              <tr>
                <td>Mean WD (m)</td>
                <td>{formatNumber(summary.mean_wd_overall_m)}</td>
                {otherSummary && <td>{formatNumber(otherSummary.mean_wd_overall_m)}</td>}
              </tr>
              <tr>
                <td>Mean spread (m)</td>
                <td>{formatNumber(summary.mean_spread_wd_m)}</td>
                {otherSummary && <td>{formatNumber(otherSummary.mean_spread_wd_m)}</td>}
              </tr>
              <tr>
                <td>p05 / p50 / p95 (m)</td>
                <td>
                  {formatNumber(summary.mean_q05_wd_m)} / {formatNumber(summary.mean_q50_wd_m)} /{" "}
                  {formatNumber(summary.mean_q95_wd_m)}
                </td>
                {otherSummary && (
                  <td>
                    {formatNumber(otherSummary.mean_q05_wd_m)} / {formatNumber(otherSummary.mean_q50_wd_m)} /{" "}
                    {formatNumber(otherSummary.mean_q95_wd_m)}
                  </td>
                )}
              </tr>
              <tr>
                <td>Isotonic exceedance applied</td>
                <td>{summary.isotonic_calibration_applied ? "yes" : "no"}</td>
                {otherSummary && <td>{otherSummary.isotonic_calibration_applied ? "yes" : "no"}</td>}
              </tr>
            </tbody>
          </table>
        )}
        {summary && exceedance.length > 0 && (
          <table className="summary-table">
            <thead>
              <tr>
                <th>Threshold (m)</th>
                <th>Mean P(WD &gt; thr)</th>
                {otherSummary && <th>{view === "calibrated" ? "Raw" : "Calibrated"}</th>}
              </tr>
            </thead>
            <tbody>
              {exceedance.map((row) => {
                const other = otherExceedance.find((entry) => entry.threshold === row.threshold);
                return (
                  <tr key={row.threshold}>
                    <td>{row.threshold}</td>
                    <td>{formatNumber(row.value)}</td>
                    {otherSummary && <td>{other ? formatNumber(other.value) : "—"}</td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h2>Forcing hydrograph</h2>
        {!forcing && <p className="muted">Forcing CSV is fetched once the run starts; refresh in a few seconds.</p>}
        {forcing && forcing.t_h.length > 0 && (
          <div className="hydrograph">
            <div>
              <h3>Stage</h3>
              <LineSparkline
                series={forcing.t_h.map((t, idx) => ({ t, y: forcing.stage[idx] }))}
                color="#0f766e"
                yLabel="stage"
              />
            </div>
            <div>
              <h3>Precipitation</h3>
              <LineSparkline
                series={forcing.t_h.map((t, idx) => ({ t, y: forcing.precip[idx] }))}
                color="#9333ea"
                yLabel="precip"
              />
            </div>
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Downloads</h2>
        {artifacts.length === 0 && <p className="muted">No artifacts available yet.</p>}
        {artifacts.length > 0 && (
          <ul className="downloads">
            {artifacts
              .slice()
              .sort((a, b) => a.artifact_id.localeCompare(b.artifact_id))
              .map((artifact) => (
                <li key={artifact.artifact_id}>
                  <a href={`/api/runs/${runId}/artifacts/${artifact.artifact_id}`}>{artifact.artifact_id}</a>
                  <span> · {Math.round(artifact.size_bytes / 1024)} KB · {artifact.content_type}</span>
                </li>
              ))}
          </ul>
        )}
        {hasHdf5 && (
          <p className="muted">Full ensemble HDF5 is available — open it with h5py for downstream analysis.</p>
        )}
      </section>

      <details className="panel">
        <summary>Run specification</summary>
        <pre>{run ? JSON.stringify(run.spec, null, 2) : "Loading…"}</pre>
      </details>

      <style jsx>{`
        .shell { max-width: 1080px; margin: 0 auto; padding: 48px 24px; font-family: ui-sans-serif, system-ui; color: #10231f; }
        .back { color: #0f766e; font-weight: 700; text-decoration: none; }
        .back:hover { text-decoration: underline; }
        .hero { margin-top: 12px; }
        h1 { font-size: 36px; margin: 8px 0 4px; }
        .meta { color: #475569; font-size: 14px; }
        .status { display: inline-block; padding: 2px 10px; border-radius: 999px; background: #e2e8f0; font-weight: 700; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; }
        .status-COMPLETED { background: #dcfce7; color: #166534; }
        .status-RUNNING { background: #fde68a; color: #78350f; }
        .status-POSTPROCESSING { background: #ddd6fe; color: #4c1d95; }
        .status-QUEUED { background: #e0e7ff; color: #312e81; }
        .status-FAILED { background: #fecaca; color: #991b1b; }
        .status-CANCELED { background: #e2e8f0; color: #475569; }
        .status-EXPIRED { background: #fde2e2; color: #7f1d1d; }
        .pin { color: #92400e; font-weight: 700; }
        .panel { border: 1px solid #cbd5e1; border-radius: 18px; padding: 22px; margin-top: 18px; background: #f8fafc; }
        .error-panel { background: #fee2e2; border-color: #fecaca; }
        .error { color: #991b1b; font-weight: 700; }
        .row-between { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
        .toggle button { border: 1px solid #cbd5e1; background: #ffffff; padding: 8px 14px; font-weight: 600; cursor: pointer; }
        .toggle button:first-child { border-radius: 999px 0 0 999px; }
        .toggle button:last-child { border-radius: 0 999px 999px 0; }
        .toggle button.active { background: #0f766e; color: white; border-color: #0f766e; }
        .product-row { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
        .chip { padding: 6px 12px; border-radius: 999px; border: 1px solid #cbd5e1; background: white; font-size: 13px; cursor: pointer; }
        .chip.active { background: #134e4a; color: white; border-color: #134e4a; }
        .map-image { display: block; margin: 16px auto 8px; max-width: 100%; border-radius: 12px; box-shadow: 0 1px 4px rgba(15,23,42,0.08); }
        .time-slider { display: flex; align-items: center; gap: 14px; margin-top: 6px; }
        .time-slider input[type="range"] { flex: 1; }
        .anim-link { display: inline-block; margin-top: 10px; color: #0f766e; font-weight: 700; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 20px; }
        .card { border: 1px solid #cbd5e1; border-radius: 14px; padding: 14px 18px; background: linear-gradient(180deg, #ffffff, #f8fafc); }
        .card-label { display: block; font-size: 12px; color: #475569; letter-spacing: 0.04em; text-transform: uppercase; }
        .card-value { display: block; margin-top: 6px; font-size: 22px; font-weight: 700; color: #0f172a; }
        .summary-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        .summary-table th, .summary-table td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }
        .summary-table th { color: #475569; font-weight: 700; }
        .hydrograph { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        .hydrograph h3 { margin: 0 0 6px; font-size: 14px; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; }
        .downloads { padding: 0; margin: 0; list-style: none; display: grid; gap: 6px; }
        .downloads a { color: #0f766e; font-weight: 700; }
        details > pre { overflow: auto; background: #0f172a; color: #e2e8f0; padding: 18px; border-radius: 14px; }
        .muted { color: #475569; font-style: italic; }
        @media (max-width: 720px) {
          .hydrograph { grid-template-columns: 1fr; }
        }
      `}</style>
    </main>
  );
}
