"use client";

import { useCallback, useEffect, useState } from "react";
import { Activity, Database, LineChart, ShieldCheck } from "lucide-react";
import { AppShell } from "../../components/AppShell";
import { DataTable } from "../../components/DataTable";
import { MetricCard } from "../../components/MetricCard";
import { PageHeader } from "../../components/PageHeader";
import { StatusBadge } from "../../components/StatusBadge";

type DriftResult = {
  test_id: string;
  test_type: string;
  descriptor_name: string | null;
  drift_detected: boolean;
  persistent_drift_detected?: boolean;
  test_statistic: number;
  threshold: number;
  n_observations?: number | null;
  created_at: string;
};
type DriftStatus = {
  message?: string;
  drift_detected?: boolean;
  persistent_drift_detected?: boolean;
  n_results?: number;
  n_detected?: number;
  n_persistent_detected?: number;
  results?: DriftResult[];
};
type TrendsData = {
  total_candidates: number;
  candidates_by_status: Record<string, number>;
  score_distribution?: { count: number; mean?: number | null; max?: number | null; p95?: number | null };
  top_flagged_descriptors?: Array<{ descriptor: string; count: number }>;
  candidate_counts_by_week?: Record<string, number>;
  recent_candidates: Array<{
    candidate_id: string;
    run_id: string;
    candidate_score: number;
    status: string;
    created_at: string;
  }>;
};
type ErrorRecord = {
  error_record_id: string;
  candidate_id: string;
  run_id: string;
  error_descriptors: Record<string, number>;
  created_at: string;
};
type HecrasData = { total: number; records: ErrorRecord[] };

const DESCRIPTOR_LABELS: Record<string, string> = {
  stage_max: "Peak coastal stage",
  precipitation_total: "Total precipitation",
  precipitation_mean: "Mean precipitation",
  precipitation_max: "Peak precipitation",
  "peak_expected_flooded_area_fraction_wettable_gt_0.05m": "Peak expected flooded fraction above 0.05 m",
  "peak_expected_flooded_area_km2_gt_0.05m": "Peak expected flooded area above 0.05 m",
  peak_area_weighted_iqr_wd_m: "Area-weighted uncertainty width",
  peak_area_weighted_total_ensemble_spread_wd_m: "Peak total ensemble spread",
  peak_area_weighted_between_checkpoint_spread_wd_m: "Peak checkpoint disagreement",
  peak_area_weighted_between_checkpoint_variance_share: "Checkpoint-disagreement share",
  peak_high_checkpoint_disagreement_area_fraction_wettable: "High-disagreement footprint",
  peak_between_checkpoint_disagreement_lead_hours: "Lead time to peak checkpoint disagreement",
  max_abs_calibration_shift_percentage_points_by_threshold: "Largest calibration shift",
  uncertainty_to_signal_ratio: "Uncertainty-to-signal ratio",
};

const DRIFT_TEST_LABELS: Record<string, string> = {
  CUSUM: "Sustained shift",
  WELCH_T: "Window mean shift",
  ENERGY_DISTANCE: "Distribution shift",
};

function formatDescriptorLabel(descriptor?: string | null): string {
  if (!descriptor) return "Joint monitored pattern";
  if (DESCRIPTOR_LABELS[descriptor]) return DESCRIPTOR_LABELS[descriptor];
  const threshold = descriptor.match(/_gt_([0-9.]+)m/);
  const thresholdText = threshold ? ` above ${threshold[1]} m` : "";
  return descriptor
    .replace(/_gt_[0-9.]+m/g, "")
    .replace(/wd/g, "water depth")
    .replace(/iqr/g, "IQR")
    .replace(/km2/g, "km²")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .concat(thresholdText);
}

function formatStatusLabel(status: string): string {
  return status.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (char) => char.toUpperCase());
}

export default function MonitoringDashboard() {
  const [drift, setDrift] = useState<DriftStatus | null>(null);
  const [trends, setTrends] = useState<TrendsData | null>(null);
  const [hecras, setHecras] = useState<HecrasData | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [driftRes, trendsRes, hecrasRes] = await Promise.all([
        fetch("/api/admin/drift-status", { cache: "no-store" }),
        fetch("/api/admin/monitoring-trends", { cache: "no-store" }),
        fetch("/api/admin/hecras-errors", { cache: "no-store" }),
      ]);
      if (driftRes.ok) setDrift(await driftRes.json());
      if (trendsRes.ok) setTrends(await trendsRes.json());
      if (hecrasRes.ok) setHecras(await hecrasRes.json());
    } catch (err) {
      setError("Failed to load monitoring data.");
    }
  }, []);

  useEffect(() => { refresh().catch(() => undefined); }, [refresh]);

  const persistentSignals = drift?.n_persistent_detected ?? 0;
  const latestSignals = drift?.n_detected ?? 0;
  const totalComparisons = hecras?.total ?? 0;
  const reviewItems = trends?.total_candidates ?? 0;

  return (
    <AppShell active="monitoring">
    <div className="monitoring-shell">
      <PageHeader
        kicker="Monitoring"
        title="Drift and review dashboard"
        subtitle="Population signals, selection trends, and HEC-RAS comparison records for research review."
        actions={<a className="button secondary" href="/admin">Admin dashboard</a>}
      />
      <section className="metric-grid">
        <MetricCard label="Persistent signals" value={persistentSignals} detail="Require repeated detection" icon={<ShieldCheck size={17} />} />
        <MetricCard label="Latest signals" value={latestSignals} detail="Most recent drift run" icon={<Activity size={17} />} />
        <MetricCard label="Review items" value={reviewItems} detail="Preserved candidate packages" icon={<Database size={17} />} />
        <MetricCard label="HEC-RAS comparisons" value={totalComparisons} detail="Completed high-fidelity records" icon={<LineChart size={17} />} />
      </section>
      {error && <p className="error">{error}</p>}

      <section className="panel">
        <div className="card-header">
          <div>
            <p className="eyebrow">Population monitoring</p>
            <h2 className="section-title">Drift status</h2>
            <p className="section-subtitle">These signals describe recent submission patterns, not proof of model error.</p>
          </div>
        </div>
        {drift ? (
          <>
            <div className="stats-row">
              <div className={`stat ${drift.persistent_drift_detected ? "danger" : drift.drift_detected ? "warn" : ""}`}>
                <span className="stat-value">{drift.n_persistent_detected ?? 0}</span>
                <span className="stat-label">Persistent Signals</span>
              </div>
              <div className="stat">
                <span className="stat-value">{drift.n_detected ?? 0}</span>
                <span className="stat-label">Latest Signals</span>
              </div>
              <div className="stat">
                <span className="stat-value">{drift.n_results ?? 0}</span>
                <span className="stat-label">Tests</span>
              </div>
            </div>
            <p>{drift.message || "No drift signals detected."}</p>
            <DataTable>
              <thead>
                <tr><th>Test</th><th>Monitored quantity</th><th>Signal</th><th>Statistic</th><th>Window</th></tr>
              </thead>
              <tbody>
                {(drift.results || []).slice(0, 12).map((r) => (
                  <tr key={r.test_id}>
                    <td>{DRIFT_TEST_LABELS[r.test_type] ?? formatStatusLabel(r.test_type)}</td>
                    <td>{formatDescriptorLabel(r.descriptor_name)}</td>
                    <td>
                      <StatusBadge tone={r.persistent_drift_detected ? "danger" : r.drift_detected ? "warning" : "neutral"}>
                        {r.persistent_drift_detected ? "Persistent" : r.drift_detected ? "Latest" : "None"}
                      </StatusBadge>
                    </td>
                    <td>{r.test_statistic.toFixed(3)}</td>
                    <td>{r.n_observations ?? "—"}</td>
                  </tr>
                ))}
                {(drift.results || []).length === 0 && (
                  <tr><td colSpan={5}>No drift-runner results yet.</td></tr>
                )}
              </tbody>
            </DataTable>
          </>
        ) : (
          <p className="loading">Loading drift status…</p>
        )}
      </section>

      <section className="panel">
        <div className="card-header">
          <div>
            <p className="eyebrow">Review queue</p>
            <h2 className="section-title">Selection trends</h2>
            <p className="section-subtitle">Review items are events preserved for later high-fidelity simulation and training-set review.</p>
          </div>
        </div>
        {trends ? (
          <>
            <div className="stats-row">
              <div className="stat">
                <span className="stat-value">{trends.total_candidates}</span>
                <span className="stat-label">Review Items</span>
              </div>
              <div className="stat">
                <span className="stat-value">{trends.score_distribution?.mean?.toFixed(2) ?? "—"}</span>
                <span className="stat-label">Mean Selection Score</span>
              </div>
              <div className="stat">
                <span className="stat-value">{trends.score_distribution?.p95?.toFixed(2) ?? "—"}</span>
                <span className="stat-label">95th-percentile Score</span>
              </div>
              {Object.entries(trends.candidates_by_status).map(([status, count]) => (
                <div className="stat" key={status}>
                  <span className="stat-value">{count}</span>
                  <span className="stat-label">{formatStatusLabel(status)}</span>
                </div>
              ))}
            </div>
            {trends.top_flagged_descriptors && trends.top_flagged_descriptors.length > 0 && (
              <>
                <h3>Most Frequent Monitoring Signals</h3>
                <div className="chip-row">
                  {trends.top_flagged_descriptors.slice(0, 8).map((item) => (
                    <span className="chip" key={item.descriptor}>{formatDescriptorLabel(item.descriptor)}: {item.count}</span>
                  ))}
                </div>
              </>
            )}
            <h3>Recent Review Items</h3>
            <DataTable>
              <thead>
                <tr><th>Run</th><th>Selection score</th><th>Status</th><th>Created</th></tr>
              </thead>
              <tbody>
                {trends.recent_candidates.map((c) => (
                  <tr key={c.candidate_id}>
                    <td><a href={`/runs/${c.run_id}`}>{c.run_id.slice(0, 10)}</a></td>
                    <td>{c.candidate_score.toFixed(2)}</td>
                    <td>{formatStatusLabel(c.status)}</td>
                    <td>{new Date(c.created_at).toLocaleString()}</td>
                  </tr>
                ))}
                {trends.recent_candidates.length === 0 && (
                  <tr><td colSpan={4}>No retraining-review items yet.</td></tr>
                )}
              </tbody>
            </DataTable>
          </>
        ) : (
          <p className="loading">Loading trends…</p>
        )}
      </section>

      <section className="panel">
        <div className="card-header">
          <div>
            <p className="eyebrow">High-fidelity loop</p>
            <h2 className="section-title">HEC-RAS comparison summary</h2>
            <p className="section-subtitle">Signed differences are reported after an admin attaches a completed HEC-RAS result.</p>
          </div>
        </div>
        {hecras ? (
          <>
            <p>{hecras.total} completed HEC-RAS comparison record(s)</p>
            <DataTable>
              <thead>
                <tr><th>Review item</th><th>Run</th><th>Leading differences</th><th>Created</th></tr>
              </thead>
              <tbody>
                {hecras.records.map((r) => (
                  <tr key={r.error_record_id}>
                    <td>{r.candidate_id.slice(0, 12)}</td>
                    <td><a href={`/runs/${r.run_id}`}>{r.run_id.slice(0, 10)}</a></td>
                    <td>
                      {Object.entries(r.error_descriptors)
                        .filter(([k]) => k.endsWith("_signed"))
                        .slice(0, 3)
                        .map(([k, v]) => `${formatDescriptorLabel(k.replace("_signed", ""))}: ${(v as number).toFixed(3)}`)
                        .join(", ") || "—"}
                    </td>
                    <td>{new Date(r.created_at).toLocaleString()}</td>
                  </tr>
                ))}
                {hecras.records.length === 0 && (
                  <tr><td colSpan={4}>No HEC-RAS comparison records yet.</td></tr>
                )}
              </tbody>
            </DataTable>
          </>
        ) : (
          <p className="loading">Loading HEC-RAS errors…</p>
        )}
      </section>

      <style jsx>{`
        .monitoring-shell { display: grid; gap: 16px; }
        h2 { font-size: 20px; margin-bottom: 8px; }
        h3 { font-size: 16px; margin-top: 16px; }
        .panel { padding: 18px; }
        .error { padding: 10px; border: 1px solid #fecaca; border-radius: 6px; background: #fef2f2; color: #991b1b; }
        .loading { color: #64748b; font-style: italic; }
        .stats-row { display: flex; gap: 24px; flex-wrap: wrap; }
        .stat { display: flex; flex-direction: column; align-items: center; padding: 12px 20px; background: white; border: 1px solid #e2e8f0; min-width: 100px; }
        .stat.warn { border-color: #d97706; background: #fffbeb; }
        .stat.danger { border-color: #dc2626; background: #fef2f2; }
        .stat-value { font-size: 28px; font-weight: 800; color: #0f766e; }
        .stat-label { font-size: 12px; color: #64748b; text-transform: uppercase; margin-top: 4px; }
        .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px; }
        .chip { background: #ecfdf5; color: #065f46; border: 1px solid #99f6e4; padding: 6px 10px; font-size: 12px; font-weight: 800; }
        a { color: var(--accent); font-weight: 800; }
      `}</style>
    </div>
    </AppShell>
  );
}
