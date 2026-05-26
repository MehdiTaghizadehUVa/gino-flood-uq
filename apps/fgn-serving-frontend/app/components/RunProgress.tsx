import { StatusBadge, statusTone } from "./StatusBadge";

const STAGES = ["SUBMITTED", "VALIDATING", "QUEUED", "RUNNING", "POSTPROCESSING", "COMPLETED"];

type RunProgressProps = {
  status?: string | null;
  progress?: number | null;
  detail?: string | null;
};

function normalizeProgress(status?: string | null, progress?: number | null) {
  if (typeof progress === "number" && Number.isFinite(progress)) {
    return Math.max(0, Math.min(100, progress));
  }
  const normalized = (status || "").toUpperCase();
  const index = STAGES.indexOf(normalized);
  if (normalized === "COMPLETED") return 100;
  if (index >= 0) return Math.round((index / (STAGES.length - 1)) * 100);
  return 0;
}

export function RunProgress({ status, progress, detail }: RunProgressProps) {
  const normalized = (status || "UNKNOWN").toUpperCase();
  const pct = normalizeProgress(status, progress);
  return (
    <div className="run-progress">
      <div className="run-progress-top">
        <StatusBadge tone={statusTone(normalized)}>{normalized}</StatusBadge>
        <strong>{pct}%</strong>
      </div>
      <div className="progress-bar" aria-label={`Run progress ${pct}%`}>
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      {detail ? <div className="metric-card-detail">{detail}</div> : null}
    </div>
  );
}
