"use client";

import { useCallback, useEffect, useState } from "react";

type UserRow = { email: string; is_admin: boolean; disclaimer_acknowledged?: boolean };
type RunRow = { run_id: string; label?: string; status: string; pinned: boolean; created_at: string };
type CandidateRow = {
  candidate_id: string;
  run_id: string;
  owner_email: string;
  candidate_score: number;
  reason: string;
  status: string;
  created_at: string;
};

const CANDIDATE_REASON_LABELS: Record<string, string> = {
  multivariate_outlier: "Joint pattern outside the reference population",
  high_uncertainty_to_signal: "High uncertainty relative to predicted signal",
  high_impact_high_uncertainty: "Broad affected area with elevated uncertainty",
  large_calibration_shift: "Large calibration shift",
  population_reinforced_candidate: "Persistent monitoring signal",
  deterministic_control_sample: "Selected for review-set balance",
  below_candidate_reference: "Historical reference-envelope-only selection",
  above_candidate_reference: "Historical reference-envelope-only selection",
};

const CANDIDATE_STATUS_LABELS: Record<string, string> = {
  NEW: "New review item",
  REVIEWED: "Reviewed",
  SELECTED_FOR_HECRAS: "Selected for HEC-RAS",
  REJECTED: "Rejected",
  SIMULATED: "HEC-RAS simulated",
};

function formatCandidateReason(reason: string): string {
  return CANDIDATE_REASON_LABELS[reason] ?? reason.replace(/_/g, " ").toLowerCase();
}

function formatCandidateStatus(status: string): string {
  return CANDIDATE_STATUS_LABELS[status] ?? status.replace(/_/g, " ").toLowerCase();
}

export default function AdminPage() {
  const [users, setUsers] = useState<UserRow[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [candidates, setCandidates] = useState<CandidateRow[]>([]);
  const [email, setEmail] = useState("");
  const [admin, setAdmin] = useState(false);
  const [message, setMessage] = useState("");

  const refresh = useCallback(async () => {
    const [usersRes, runsRes, candidatesRes] = await Promise.all([
      fetch("/api/admin/users", { cache: "no-store" }),
      fetch("/api/admin/runs", { cache: "no-store" }),
      fetch("/api/admin/retraining-candidates", { cache: "no-store" }),
    ]);
    if (usersRes.ok) setUsers(await usersRes.json());
    if (runsRes.ok) setRuns(await runsRes.json());
    if (candidatesRes.ok) setCandidates(await candidatesRes.json());
  }, []);

  useEffect(() => { refresh().catch(() => undefined); }, [refresh]);

  async function addUser() {
    const res = await fetch("/api/admin/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, is_admin: admin }),
    });
    setMessage(res.ok ? "User saved." : `User update failed: ${await res.text()}`);
    setEmail("");
    setAdmin(false);
    await refresh();
  }

  async function removeUser(target: string) {
    const res = await fetch(`/api/admin/users/${encodeURIComponent(target)}`, { method: "DELETE" });
    setMessage(res.ok ? "User removed." : `Remove failed: ${await res.text()}`);
    await refresh();
  }

  async function runAction(runId: string, action: "pin" | "unpin" | "cancel") {
    const res = await fetch(`/api/admin/runs/${runId}/${action}`, { method: "POST" });
    setMessage(res.ok ? `Run ${action} succeeded.` : `Run ${action} failed: ${await res.text()}`);
    await refresh();
  }

  async function candidateAction(candidateId: string, status: string) {
    const res = await fetch(`/api/admin/retraining-candidates/${encodeURIComponent(candidateId)}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    setMessage(res.ok ? `Review item marked ${formatCandidateStatus(status)}.` : `Review-item update failed: ${await res.text()}`);
    await refresh();
  }

  return (
    <main className="shell">
      <a href="/">Back to console</a>
      <h1>FGN Serving Admin</h1>
      <a href="/admin/monitoring" style={{marginLeft: 12}}>Monitoring Dashboard →</a>
      {message && <p className="message">{message}</p>}
      <section className="panel">
        <h2>Allowlist</h2>
        <div className="add-row">
          <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="researcher@example.edu" />
          <label><input type="checkbox" checked={admin} onChange={(e) => setAdmin(e.target.checked)} /> admin</label>
          <button onClick={addUser} disabled={!email.trim()}>Save</button>
        </div>
        <table>
          <thead><tr><th>Email</th><th>Admin</th><th>Disclaimer</th><th></th></tr></thead>
          <tbody>
            {users.map((user) => (
              <tr key={user.email}>
                <td>{user.email}</td>
                <td>{user.is_admin ? "yes" : "no"}</td>
                <td>{user.disclaimer_acknowledged ? "accepted" : "pending"}</td>
                <td><button onClick={() => removeUser(user.email)}>Remove</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="panel">
        <h2>Retraining Review Items</h2>
        <table>
          <thead><tr><th>Run</th><th>Owner</th><th>Selection score</th><th>Status</th><th>Reason</th><th>Actions</th></tr></thead>
          <tbody>
            {candidates.map((candidate) => (
              <tr key={candidate.candidate_id}>
                <td><a href={`/runs/${candidate.run_id}`}>{candidate.run_id.slice(0, 10)}</a></td>
                <td>{candidate.owner_email}</td>
                <td>{candidate.candidate_score.toFixed(2)}</td>
                <td>{formatCandidateStatus(candidate.status)}</td>
                <td>{formatCandidateReason(candidate.reason)}</td>
                <td>
                  <button onClick={() => candidateAction(candidate.candidate_id, "REVIEWED")}>Review</button>
                  <button onClick={() => candidateAction(candidate.candidate_id, "SELECTED_FOR_HECRAS")}>Select HEC-RAS</button>
                  <button onClick={() => candidateAction(candidate.candidate_id, "REJECTED")}>Reject</button>
                  <button onClick={() => candidateAction(candidate.candidate_id, "SIMULATED")}>Simulated</button>
                </td>
              </tr>
            ))}
            {candidates.length === 0 && (
              <tr><td colSpan={6}>No retraining-review items yet.</td></tr>
            )}
          </tbody>
        </table>
      </section>
      <section className="panel">
        <h2>Runs</h2>
        <table>
          <thead><tr><th>Run</th><th>Status</th><th>Created</th><th>Actions</th></tr></thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id}>
                <td>{run.label || run.run_id.slice(0, 10)}{run.pinned ? " · pinned" : ""}</td>
                <td>{run.status}</td>
                <td>{new Date(run.created_at).toLocaleString()}</td>
                <td>
                  <button onClick={() => runAction(run.run_id, run.pinned ? "unpin" : "pin")}>{run.pinned ? "Unpin" : "Pin"}</button>
                  <button onClick={() => runAction(run.run_id, "cancel")}>Cancel</button>
                  <a href={`/runs/${run.run_id}`}>Open</a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <style jsx>{`
        .shell { max-width: 1120px; margin: 0 auto; padding: 28px 22px; font-family: ui-sans-serif, system-ui; color: #10231f; }
        h1 { font-size: 28px; }
        .panel { border: 1px solid #cbd5e1; background: #f8fafc; padding: 18px; margin-top: 16px; }
        .add-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
        input { border: 1px solid #cbd5e1; padding: 9px; min-width: 260px; }
        button { border: 0; background: #0f766e; color: white; padding: 8px 12px; margin-right: 8px; font-weight: 800; cursor: pointer; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .message { padding: 10px; background: #e0f2fe; color: #0c4a6e; }
        table { width: 100%; border-collapse: collapse; margin-top: 14px; }
        th, td { text-align: left; border-bottom: 1px solid #e2e8f0; padding: 10px; font-size: 14px; }
        a { color: #0f766e; font-weight: 800; }
      `}</style>
    </main>
  );
}
