"use client";

import { useCallback, useEffect, useState } from "react";

type DriftStatus = { message?: string };
type TrendsData = {
  total_candidates: number;
  candidates_by_status: Record<string, number>;
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

  return (
    <main className="shell">
      <a href="/admin">← Back to Admin</a>
      <h1>Monitoring Dashboard</h1>
      {error && <p className="error">{error}</p>}

      <section className="panel">
        <h2>Drift Status</h2>
        {drift ? (
          <p>{drift.message || "No drift signals detected."}</p>
        ) : (
          <p className="loading">Loading drift status…</p>
        )}
      </section>

      <section className="panel">
        <h2>Candidate Trends</h2>
        {trends ? (
          <>
            <div className="stats-row">
              <div className="stat">
                <span className="stat-value">{trends.total_candidates}</span>
                <span className="stat-label">Total Candidates</span>
              </div>
              {Object.entries(trends.candidates_by_status).map(([status, count]) => (
                <div className="stat" key={status}>
                  <span className="stat-value">{count}</span>
                  <span className="stat-label">{status}</span>
                </div>
              ))}
            </div>
            <h3>Recent Candidates</h3>
            <table>
              <thead>
                <tr><th>Run</th><th>Score</th><th>Status</th><th>Created</th></tr>
              </thead>
              <tbody>
                {trends.recent_candidates.map((c) => (
                  <tr key={c.candidate_id}>
                    <td><a href={`/runs/${c.run_id}`}>{c.run_id.slice(0, 10)}</a></td>
                    <td>{c.candidate_score.toFixed(2)}</td>
                    <td>{c.status}</td>
                    <td>{new Date(c.created_at).toLocaleString()}</td>
                  </tr>
                ))}
                {trends.recent_candidates.length === 0 && (
                  <tr><td colSpan={4}>No candidates yet.</td></tr>
                )}
              </tbody>
            </table>
          </>
        ) : (
          <p className="loading">Loading trends…</p>
        )}
      </section>

      <section className="panel">
        <h2>HEC-RAS Error Summary</h2>
        {hecras ? (
          <>
            <p>{hecras.total} error record(s)</p>
            <table>
              <thead>
                <tr><th>Candidate</th><th>Run</th><th>Errors</th><th>Created</th></tr>
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
                        .map(([k, v]) => `${k.replace("_signed", "")}: ${(v as number).toFixed(3)}`)
                        .join(", ") || "—"}
                    </td>
                    <td>{new Date(r.created_at).toLocaleString()}</td>
                  </tr>
                ))}
                {hecras.records.length === 0 && (
                  <tr><td colSpan={4}>No HEC-RAS error records yet.</td></tr>
                )}
              </tbody>
            </table>
          </>
        ) : (
          <p className="loading">Loading HEC-RAS errors…</p>
        )}
      </section>

      <style jsx>{`
        .shell { max-width: 1120px; margin: 0 auto; padding: 28px 22px; font-family: ui-sans-serif, system-ui; color: #10231f; }
        h1 { font-size: 28px; }
        h2 { font-size: 20px; margin-bottom: 8px; }
        h3 { font-size: 16px; margin-top: 16px; }
        .panel { border: 1px solid #cbd5e1; background: #f8fafc; padding: 18px; margin-top: 16px; }
        .error { padding: 10px; background: #fef2f2; color: #991b1b; }
        .loading { color: #64748b; font-style: italic; }
        .stats-row { display: flex; gap: 24px; flex-wrap: wrap; }
        .stat { display: flex; flex-direction: column; align-items: center; padding: 12px 20px; background: white; border: 1px solid #e2e8f0; min-width: 100px; }
        .stat-value { font-size: 28px; font-weight: 800; color: #0f766e; }
        .stat-label { font-size: 12px; color: #64748b; text-transform: uppercase; margin-top: 4px; }
        table { width: 100%; border-collapse: collapse; margin-top: 14px; }
        th, td { text-align: left; border-bottom: 1px solid #e2e8f0; padding: 10px; font-size: 14px; }
        a { color: #0f766e; font-weight: 800; }
        button { border: 0; background: #0f766e; color: white; padding: 8px 12px; margin-right: 8px; font-weight: 800; cursor: pointer; }
      `}</style>
    </main>
  );
}
