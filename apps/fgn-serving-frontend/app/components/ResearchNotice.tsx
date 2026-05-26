import type { ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

type ResearchNoticeProps = {
  title?: string;
  children: ReactNode;
};

export function ResearchNotice({ title = "Research only.", children }: ResearchNoticeProps) {
  return (
    <aside className="research-notice">
      <AlertTriangle size={18} aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        <span>{children}</span>
      </div>
    </aside>
  );
}
