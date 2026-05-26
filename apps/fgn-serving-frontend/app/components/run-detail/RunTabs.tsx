"use client";

import type { ReactNode } from "react";
import { GitCompareArrows, LineChart, Map, Waves } from "lucide-react";

export type RunTabKey = "hazard" | "uncertainty" | "drivers" | "compare";

type RunDetailTabsProps = {
  active: RunTabKey;
  onChange: (tab: RunTabKey) => void;
};

const tabs: Array<{ key: RunTabKey; label: string; icon: ReactNode }> = [
  { key: "hazard", label: "Hazard", icon: <Map size={15} /> },
  { key: "uncertainty", label: "Uncertainty", icon: <Waves size={15} /> },
  { key: "drivers", label: "Drivers", icon: <LineChart size={15} /> },
  { key: "compare", label: "Compare", icon: <GitCompareArrows size={15} /> },
];

export function RunDetailTabs({ active, onChange }: RunDetailTabsProps) {
  return (
    <nav className="tabs tabs-elevated" role="tablist" aria-label="Run views">
      {tabs.map((entry) => (
        <button
          key={entry.key}
          type="button"
          role="tab"
          aria-selected={active === entry.key}
          className={`tab ${active === entry.key ? "active" : ""}`}
          onClick={() => onChange(entry.key)}
        >
          {entry.icon}
          {entry.label}
        </button>
      ))}
    </nav>
  );
}

function TabFrame({
  children,
  label,
}: {
  children: ReactNode;
  label: string;
}) {
  return (
    <div className="run-tab-frame" role="tabpanel" aria-label={label}>
      {children}
    </div>
  );
}

export function HazardTab({ children }: { children: ReactNode }) {
  return <TabFrame label="Hazard">{children}</TabFrame>;
}

export function UncertaintyTab({ children }: { children: ReactNode }) {
  return <TabFrame label="Uncertainty">{children}</TabFrame>;
}

export function DriversTab({ children }: { children: ReactNode }) {
  return <TabFrame label="Drivers">{children}</TabFrame>;
}

export function CompareTab({ children }: { children: ReactNode }) {
  return <TabFrame label="Compare">{children}</TabFrame>;
}
