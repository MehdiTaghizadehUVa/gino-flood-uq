import type { ReactNode } from "react";

type ToolbarProps = {
  children: ReactNode;
  className?: string;
  label?: string;
};

export function Toolbar({ children, className = "", label }: ToolbarProps) {
  return (
    <div className={`toolbar ${className}`.trim()} aria-label={label}>
      {children}
    </div>
  );
}
