"use client";

import { useCallback, useEffect, useState } from "react";

type User = {
  email: string;
  is_admin: boolean;
  disclaimer_acknowledged: boolean;
};

type Bundle = {
  bundle_id: string;
  domain_name: string;
  max_forecast_steps: number;
  dt_seconds: number;
  total_members: number;
  research_disclaimer: string;
};

type RunRow = {
  run_id: string;
  label?: string;
  status: string;
  progress: number;
  created_at: string;
  pinned: boolean;
};

type ValidationState = {
  valid: boolean;
  messages: string[];
  summary?: Record<string, unknown> | null;
};

export default function Page() {
  const [user, setUser] = useState<User | null>(null);
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [label, setLabel] = useState("");
  const [forecastSteps, setForecastSteps] = useState("");
  const [thresholds, setThresholds] = useState("0.01,0.05,0.1,0.3,0.5");
  const [requestAnimation, setRequestAnimation] = useState(true);
  const [requestFullHdf5, setRequestFullHdf5] = useState(false);
  const [validation, setValidation] = useState<ValidationState | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    const [meRes, bundleRes, runsRes] = await Promise.all([
      fetch("/api/me", { cache: "no-store" }),
      fetch("/api/model-bundle", { cache: "no-store" }),
      fetch("/api/runs", { cache: "no-store" }),
    ]);
    if (meRes.ok) setUser(await meRes.json());
    if (bundleRes.ok) setBundle(await bundleRes.json());
    if (runsRes.ok) setRuns(await runsRes.json());
  }, []);

  useEffect(() => {
    refresh().catch(() => undefined);
    const timer = setInterval(() => refresh().catch(() => undefined), 10000);
    return () => clearInterval(timer);
  }, [refresh]);

  const validateFile = useCallback(async (nextFile: File | null, nextSteps = forecastSteps) => {
    setValidation(null);
    if (!nextFile) return;
    const form = new FormData();
    form.append("file", nextFile);
    if (nextSteps.trim()) form.append("forecast_steps", nextSteps.trim());
    const res = await fetch("/api/forcing/validate", { method: "POST", body: form });
    if (res.ok) setValidation(await res.json());
  }, [forecastSteps]);

  async function acknowledgeDisclaimer() {
    setBusy(true);
    try {
      const res = await fetch("/api/me/disclaimer", { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setUser(await res.json());
      setMessage("Research-only disclaimer acknowledged.");
    } catch (exc) {
      setMessage(`Could not acknowledge disclaimer: ${String(exc)}`);
    } finally {
      setBusy(false);
    }
  }

  async function submitRun() {
    if (!file) {
      setMessage("Choose a forcing CSV first.");
      return;
    }
    if (!user?.disclaimer_acknowledged) {
      setMessage("Acknowledge the research-only disclaimer before submitting.");
      return;
    }
    if (validation && !validation.valid) {
      setMessage(`Fix validation errors first: ${validation.messages.join("; ")}`);
      return;
    }
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      if (label.trim()) form.append("label", label.trim());
      if (forecastSteps.trim()) form.append("forecast_steps", forecastSteps.trim());
      form.append("output_detail", "standard");
      form.append("exceedance_thresholds_m", thresholds);
      form.append("request_animation", String(requestAnimation));
      form.append("request_full_hdf5", String(requestFullHdf5));
      const res = await fetch("/api/runs", { method: "POST", body: form });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
      setMessage(`Submitted run ${payload.run_id}.`);
      setFile(null);
      setLabel("");
      setValidation(null);
      await refresh();
    } catch (exc) {
      setMessage(`Submission rejected: ${String(exc)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>Coastal FGN UQ Console</h1>
          <p>{bundle ? `${bundle.domain_name} · ${bundle.total_members} members · max ${bundle.max_forecast_steps} steps` : "Loading model bundle..."}</p>
        </div>
        <div className="identity">
          <strong>{user?.email ?? "Unauthenticated"}</strong>
          {user?.is_admin && <a href="/admin">Admin</a>}
        </div>
      </header>

      <section className="notice">
        <strong>Research only.</strong> {bundle?.research_disclaimer ?? "Not for emergency or operational decision use."}
        {!user?.disclaimer_acknowledged && (
          <button type="button" onClick={acknowledgeDisclaimer} disabled={busy}>
            Acknowledge
          </button>
        )}
      </section>

      <section className="layout">
        <aside className="panel">
          <h2>Input Contract</h2>
          <dl>
            <dt>CSV cadence</dt>
            <dd>{bundle ? `${bundle.dt_seconds} seconds` : "-"}</dd>
            <dt>Required columns</dt>
            <dd><code>time_seconds</code>, <code>stage</code>, <code>precipitation</code></dd>
          </dl>
          <a className="button secondary" href="/api/forcing-template" download>
            Download valid CSV template
          </a>
        </aside>

        <section className="panel">
          <div className="row">
            <h2>Submit Scenario</h2>
            <button type="button" onClick={() => refresh()} disabled={busy}>Refresh</button>
          </div>
          <div className="form-grid">
            <label>
              Label
              <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Optional run label" />
            </label>
            <label>
              Forecast steps
              <input
                value={forecastSteps}
                onChange={(e) => {
                  setForecastSteps(e.target.value);
                  validateFile(file, e.target.value).catch(() => undefined);
                }}
                placeholder="Default: bundle max supported by CSV"
              />
            </label>
            <label className="wide">
              Exceedance thresholds (m)
              <input value={thresholds} onChange={(e) => setThresholds(e.target.value)} />
            </label>
          </div>
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => {
              const next = event.target.files?.[0] ?? null;
              setFile(next);
              validateFile(next).catch(() => undefined);
            }}
          />
          <div className="checks">
            <label><input type="checkbox" checked={requestAnimation} onChange={(e) => setRequestAnimation(e.target.checked)} /> GIF animation</label>
            <label><input type="checkbox" checked={requestFullHdf5} onChange={(e) => setRequestFullHdf5(e.target.checked)} /> Full HDF5 ensemble</label>
          </div>
          {validation && (
            <div className={validation.valid ? "valid" : "invalid"}>
              {validation.valid ? "CSV validation passed." : validation.messages.join("; ")}
            </div>
          )}
          <button className="button" type="button" onClick={submitRun} disabled={busy || !file}>
            Submit run
          </button>
          {message && <p className="message">{message}</p>}
        </section>
      </section>

      <section className="panel">
        <div className="row">
          <h2>Run Queue</h2>
          <span>{runs.length} visible</span>
        </div>
        <table>
          <thead>
            <tr><th>Run</th><th>Status</th><th>Progress</th><th>Created</th><th></th></tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id}>
                <td>{run.label || run.run_id.slice(0, 10)}{run.pinned ? " · pinned" : ""}</td>
                <td><span className={`status status-${run.status}`}>{run.status}</span></td>
                <td>{Math.round((run.progress ?? 0) * 100)}%</td>
                <td>{new Date(run.created_at).toLocaleString()}</td>
                <td><a href={`/runs/${run.run_id}`}>Open</a></td>
              </tr>
            ))}
            {runs.length === 0 && <tr><td colSpan={5}>No runs yet.</td></tr>}
          </tbody>
        </table>
      </section>

      <style jsx>{`
        .shell { max-width: 1180px; margin: 0 auto; padding: 28px 22px; font-family: ui-sans-serif, system-ui; color: #10231f; }
        .topbar, .row { display: flex; justify-content: space-between; gap: 16px; align-items: center; }
        h1 { font-size: 28px; margin: 0 0 4px; }
        h2 { margin: 0 0 14px; font-size: 18px; }
        p { margin: 0; color: #475569; }
        .identity { display: grid; gap: 4px; justify-items: end; font-size: 14px; }
        .notice { display: flex; justify-content: space-between; align-items: center; gap: 14px; margin: 20px 0; padding: 14px 16px; border: 1px solid #fbbf24; background: #fffbeb; }
        .layout { display: grid; grid-template-columns: 330px 1fr; gap: 16px; }
        .panel { border: 1px solid #cbd5e1; background: #f8fafc; padding: 18px; margin-top: 16px; }
        .form-grid { display: grid; grid-template-columns: 1fr 180px; gap: 12px; }
        .wide { grid-column: 1 / -1; }
        label { display: grid; gap: 6px; color: #334155; font-weight: 700; font-size: 13px; }
        input { border: 1px solid #cbd5e1; padding: 10px; background: white; }
        input[type="file"] { margin: 14px 0; width: 100%; }
        .checks { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 14px; }
        .checks label { display: flex; align-items: center; gap: 8px; }
        .button, button { border: 0; background: #0f766e; color: white; padding: 9px 14px; font-weight: 800; cursor: pointer; }
        .secondary { display: inline-block; text-decoration: none; background: #e2e8f0; color: #0f172a; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .valid, .invalid, .message { padding: 10px; margin-bottom: 12px; }
        .valid { background: #dcfce7; color: #166534; }
        .invalid { background: #fee2e2; color: #991b1b; }
        .message { background: #e0f2fe; color: #0c4a6e; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; border-bottom: 1px solid #e2e8f0; padding: 10px; font-size: 14px; }
        .status { font-weight: 900; font-size: 12px; }
        .status-COMPLETED { color: #166534; }
        .status-FAILED { color: #991b1b; }
        .status-RUNNING, .status-POSTPROCESSING { color: #1d4ed8; }
        .status-QUEUED, .status-VALIDATING, .status-SUBMITTED { color: #92400e; }
        dt { color: #64748b; font-size: 12px; font-weight: 800; text-transform: uppercase; }
        dd { margin: 3px 0 14px; }
        @media (max-width: 840px) { .layout { grid-template-columns: 1fr; } .topbar, .notice { align-items: flex-start; flex-direction: column; } }
      `}</style>
    </main>
  );
}
