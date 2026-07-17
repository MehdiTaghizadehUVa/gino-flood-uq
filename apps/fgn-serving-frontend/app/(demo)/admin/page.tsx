"use client";

import { useCallback, useEffect, useState } from "react";
import { Activity, Database, ShieldCheck, UserPlus, Users } from "lucide-react";
import { AppShell } from "../../components/AppShell";
import { DataTable } from "../../components/DataTable";
import { MetricCard } from "../../components/MetricCard";
import { PageHeader } from "../../components/PageHeader";
import { StatusBadge, statusTone } from "../../components/StatusBadge";

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
  const [showDeletedRuns, setShowDeletedRuns] = useState(false);

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

  const activeRuns = runs.filter((run) => showDeletedRuns || run.status !== "DELETED");
  const newCandidates = candidates.filter((candidate) => candidate.status === "NEW").length;
  const adminUsers = users.filter((user) => user.is_admin).length;

  return (
    <AppShell active="admin">
    <div className="admin-shell">
      <PageHeader
        kicker="Administration"
        title="Service administration dashboard"
        subtitle="Manage collaborators, retraining-review items, and active run operations from one place."
        actions={<a className="button secondary" href="/admin/monitoring">Monitoring dashboard</a>}
      />
      <section className="metric-grid">
        <MetricCard label="Allowed users" value={users.length} detail={`${adminUsers} admin${adminUsers === 1 ? "" : "s"}`} icon={<Users size={17} />} />
        <MetricCard label="Review queue" value={newCandidates} detail="New retraining-review items" icon={<Database size={17} />} />
        <MetricCard label="Visible runs" value={activeRuns.length} detail={showDeletedRuns ? "Including deleted history" : "Deleted runs hidden"} icon={<Activity size={17} />} />
        <MetricCard label="Access mode" value="Allowlist" detail="Google OAuth + approved emails" icon={<ShieldCheck size={17} />} />
      </section>
      {message && <p className="message">{message}</p>}
      <section className="panel">
        <div className="card-header">
          <div>
            <p className="eyebrow">Access</p>
            <h2 className="section-title">Allowlist</h2>
            <p className="section-subtitle">Only approved collaborators can create runs or inspect artifacts.</p>
          </div>
        </div>
        <div className="add-row">
          <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="user@example.com" />
          <label><input type="checkbox" checked={admin} onChange={(e) => setAdmin(e.target.checked)} /> admin</label>
          <button className="primary" onClick={addUser} disabled={!email.trim()}><UserPlus size={15} /> Save</button>
        </div>
        <DataTable>
          <thead><tr><th>Email</th><th>Admin</th><th>Disclaimer</th><th></th></tr></thead>
          <tbody>
            {users.map((user) => (
              <tr key={user.email}>
                <td>{user.email}</td>
                <td><StatusBadge tone={user.is_admin ? "active" : "neutral"}>{user.is_admin ? "Admin" : "User"}</StatusBadge></td>
                <td><StatusBadge tone={user.disclaimer_acknowledged ? "success" : "warning"}>{user.disclaimer_acknowledged ? "Accepted" : "Pending"}</StatusBadge></td>
                <td><button className="danger" onClick={() => removeUser(user.email)}>Remove</button></td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      </section>
      <section className="panel">
        <div className="card-header">
          <div>
            <p className="eyebrow">Monitoring</p>
            <h2 className="section-title">Retraining-review items</h2>
            <p className="section-subtitle">Candidate packages preserved for later high-fidelity review.</p>
          </div>
        </div>
        <DataTable>
          <thead><tr><th>Run</th><th>Owner</th><th>Selection score</th><th>Status</th><th>Reason</th><th>Actions</th></tr></thead>
          <tbody>
            {candidates.map((candidate) => (
              <tr key={candidate.candidate_id}>
                <td><a href={`/demo/runs/${candidate.run_id}`}>{candidate.run_id.slice(0, 10)}</a></td>
                <td>{candidate.owner_email}</td>
                <td>{candidate.candidate_score.toFixed(2)}</td>
                <td><StatusBadge tone={statusTone(candidate.status)}>{formatCandidateStatus(candidate.status)}</StatusBadge></td>
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
        </DataTable>
      </section>
      <section className="panel">
        <div className="card-header">
          <div>
            <p className="eyebrow">Operations</p>
            <h2 className="section-title">Runs</h2>
            <p className="section-subtitle">Deleted runs are hidden by default to keep run history focused.</p>
          </div>
          <label className="toggle-row">
            <input type="checkbox" checked={showDeletedRuns} onChange={(event) => setShowDeletedRuns(event.target.checked)} />
            Show deleted runs
          </label>
        </div>
        <DataTable>
          <thead><tr><th>Run</th><th>Status</th><th>Created</th><th>Actions</th></tr></thead>
          <tbody>
            {activeRuns.map((run) => (
              <tr key={run.run_id}>
                <td>{run.label || run.run_id.slice(0, 10)}{run.pinned ? " · pinned" : ""}</td>
                <td><StatusBadge tone={statusTone(run.status)}>{run.status}</StatusBadge></td>
                <td>{new Date(run.created_at).toLocaleString()}</td>
                <td>
                  <button onClick={() => runAction(run.run_id, run.pinned ? "unpin" : "pin")}>{run.pinned ? "Unpin" : "Pin"}</button>
                  <button onClick={() => runAction(run.run_id, "cancel")}>Cancel</button>
                  <a href={`/demo/runs/${run.run_id}`}>Open</a>
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      </section>
      <style jsx>{`
        .admin-shell { display: grid; gap: 16px; }
        .panel { padding: 18px; }
        .add-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
        input { border: 1px solid var(--border); border-radius: 6px; padding: 9px; min-width: 260px; }
        .message { padding: 10px; border: 1px solid #9bd4f5; border-radius: 6px; background: #e0f2fe; color: #0c4a6e; }
        .toggle-row { display: inline-flex; gap: 8px; align-items: center; color: var(--muted); font-size: 13px; font-weight: 700; }
        a { color: var(--accent); font-weight: 800; }
      `}</style>
    </div>
    </AppShell>
  );
}
