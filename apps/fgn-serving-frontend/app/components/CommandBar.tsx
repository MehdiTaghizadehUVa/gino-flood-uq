import type { ReactNode } from "react";

type CommandBarProps = {
  left?: ReactNode;
  right?: ReactNode;
};

export function CommandBar({ left, right }: CommandBarProps) {
  return (
    <div className="toolbar command-panel">
      <div>{left}</div>
      <div>{right}</div>
    </div>
  );
}
