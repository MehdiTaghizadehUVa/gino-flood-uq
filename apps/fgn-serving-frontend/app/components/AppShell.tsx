import type { ReactNode } from "react";
import {
  Activity,
  Database,
  Home,
  LineChart,
  ListChecks,
  ShieldCheck,
  Waves
} from "lucide-react";

type AppSection = "home" | "runs" | "admin" | "monitoring";

type AppShellProps = {
  active?: AppSection;
  children: ReactNode;
  userEmail?: string | null;
};

const navItems: Array<{
  href: string;
  label: string;
  active: AppSection;
  icon: ReactNode;
}> = [
  { href: "/", label: "New run", active: "home", icon: <Home size={16} /> },
  { href: "/#runs", label: "Run queue", active: "runs", icon: <ListChecks size={16} /> },
  { href: "/admin", label: "Admin", active: "admin", icon: <ShieldCheck size={16} /> },
  {
    href: "/admin/monitoring",
    label: "Monitoring",
    active: "monitoring",
    icon: <LineChart size={16} />
  }
];

export function AppShell({ active = "home", children, userEmail }: AppShellProps) {
  return (
    <div className="app-shell">
      <aside className="app-sidebar" aria-label="Primary navigation">
        <div className="app-brand">
          <div className="app-brand-mark" aria-hidden="true">
            <Waves size={19} />
          </div>
          <div>
            <p className="app-brand-title">Coastal Flood-UQ Console</p>
            <p className="app-brand-subtitle">Fixed-domain FGN research server</p>
          </div>
        </div>
        <nav className="app-nav">
          {navItems.map((item) => (
            <a key={item.href} href={item.href} data-active={active === item.active}>
              {item.icon}
              <span>{item.label}</span>
            </a>
          ))}
        </nav>
        <div className="app-sidebar-foot">
          <strong>Research use only.</strong>
          <br />
          Results are calibrated uncertainty products for scientific review, not emergency or
          operational flood guidance.
          {userEmail ? (
            <>
              <br />
              <br />
              <Activity size={12} aria-hidden="true" /> Signed in as {userEmail}
            </>
          ) : null}
          <br />
          <Database size={12} aria-hidden="true" /> Bundle-backed artifact workflow
        </div>
      </aside>
      <main className="app-main">{children}</main>
    </div>
  );
}
