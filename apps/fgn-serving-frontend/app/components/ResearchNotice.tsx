import type { ReactNode } from "react";
import { ShieldCheck } from "lucide-react";

type ResearchNoticeProps = {
  title?: string;
  children: ReactNode;
};

export function ResearchNotice({ title = "Model use and governance.", children }: ResearchNoticeProps) {
  return (
    <aside className="research-notice">
      <ShieldCheck size={18} aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        <span>{children}</span>
      </div>
    </aside>
  );
}
