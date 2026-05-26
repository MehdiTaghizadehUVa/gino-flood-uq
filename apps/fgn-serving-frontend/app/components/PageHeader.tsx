import type { ReactNode } from "react";

type PageHeaderProps = {
  kicker?: string;
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
};

export function PageHeader({ kicker, title, subtitle, actions }: PageHeaderProps) {
  return (
    <header className="app-topbar">
      <div>
        {kicker ? <div className="page-kicker">{kicker}</div> : null}
        <h1 className="page-title">{title}</h1>
        {subtitle ? <div className="page-subtitle">{subtitle}</div> : null}
      </div>
      {actions ? <div className="app-user">{actions}</div> : null}
    </header>
  );
}
