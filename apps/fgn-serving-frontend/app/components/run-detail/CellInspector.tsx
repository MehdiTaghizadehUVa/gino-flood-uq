import type { ReactNode } from "react";

type CellInspectorProps = {
  selected: boolean;
  children: ReactNode;
};

export function CellInspector({ selected, children }: CellInspectorProps) {
  return (
    <aside
      className={`panel poi-panel hazard-inspector ${selected ? "has-selection" : "is-empty"}`}
      aria-label="Per-cell ensemble inspector"
    >
      {children}
    </aside>
  );
}
