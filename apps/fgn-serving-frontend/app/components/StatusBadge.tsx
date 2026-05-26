import type { ReactNode } from "react";

type StatusTone = "neutral" | "success" | "warning" | "danger" | "active";

type StatusBadgeProps = {
  children: ReactNode;
  tone?: StatusTone;
};

export function statusTone(status?: string | null): StatusTone {
  const normalized = (status || "").toUpperCase();
  if (["COMPLETED", "READY", "PASSED", "SIMULATED"].includes(normalized)) return "success";
  if (["RUNNING", "POSTPROCESSING", "VALIDATING", "QUEUED", "SUBMITTED"].includes(normalized)) {
    return "active";
  }
  if (["FAILED", "CANCELED", "EXPIRED", "DELETED", "REJECTED"].includes(normalized)) return "danger";
  if (["NEW", "REVIEWED", "SELECTED_FOR_HECRAS"].includes(normalized)) return "warning";
  return "neutral";
}

export function StatusBadge({ children, tone = "neutral" }: StatusBadgeProps) {
  return (
    <span className="status-badge" data-tone={tone}>
      {children}
    </span>
  );
}
