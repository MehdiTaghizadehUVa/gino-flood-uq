import type { ReactNode } from "react";
import {
  Activity,
  CircleDot,
  Database,
  Globe2,
  Home,
  LineChart,
  ListChecks,
  LogOut,
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
  active: AppSection | null;
  icon: ReactNode;
}> = [
  { href: "/demo?workspace=new", label: "New analysis", active: "home", icon: <Home size={16} /> },
  { href: "/demo?workspace=runs#runs", label: "Analysis queue", active: "runs", icon: <ListChecks size={16} /> },
  { href: "/admin", label: "Admin", active: "admin", icon: <ShieldCheck size={16} /> },
  {
    href: "/admin/monitoring",
    label: "Monitoring",
    active: "monitoring",
    icon: <LineChart size={16} />
  },
  { href: "/", label: "Product site", active: null, icon: <Globe2 size={16} /> }
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
            <p className="app-brand-subtitle">Fixed-domain probabilistic service</p>
          </div>
        </div>
        <nav className="app-nav">
          {navItems.map((item) => (
            <a key={item.href} href={item.href} data-active={active === item.active} title={item.label}>
              {item.icon}
              <span>{item.label}</span>
            </a>
          ))}
        </nav>
        <div className="app-sidebar-foot">
          <strong>Managed access.</strong>
          <br />
          Calibrated uncertainty products for scenario evaluation, asset review, and planning.
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
      <main className="app-main">
        <header className="app-command-bar" aria-label="Application status">
          <div className="command-left">
            <span className="command-chip health">
              <CircleDot size={12} aria-hidden="true" />
              Lab server
            </span>
            <div className="command-title">
              <strong>Coastal Flood-UQ Console</strong>
              <span>Fixed-domain probabilistic workspace</span>
            </div>
          </div>
          <div className="command-right">
            <span className="command-chip research">Managed workspace</span>
            <span className="command-chip">
              <Activity size={12} aria-hidden="true" />
              {userEmail ?? "Authentication required"}
            </span>
            <a className="command-link" href="/oauth2/sign_out">
              <LogOut size={13} aria-hidden="true" /> Sign out
            </a>
          </div>
        </header>
        {children}
      </main>
    </div>
  );
}
