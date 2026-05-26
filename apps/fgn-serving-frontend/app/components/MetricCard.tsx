import type { ReactNode } from "react";

type MetricCardProps = {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  icon?: ReactNode;
};

export function MetricCard({ label, value, detail, icon }: MetricCardProps) {
  return (
    <section className="metric-card">
      <div className="metric-card-header">
        <span>{label}</span>
        {icon ? <span aria-hidden="true">{icon}</span> : null}
      </div>
      <div className="metric-card-value">{value}</div>
      {detail ? <div className="metric-card-detail">{detail}</div> : null}
    </section>
  );
}
