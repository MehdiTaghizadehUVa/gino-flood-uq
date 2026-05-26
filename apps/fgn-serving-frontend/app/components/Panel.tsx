import type { ReactNode } from "react";

type PanelProps = {
  children: ReactNode;
  className?: string;
  ariaLabel?: string;
};

export function Panel({ children, className = "", ariaLabel }: PanelProps) {
  return (
    <section className={`panel ${className}`.trim()} aria-label={ariaLabel}>
      {children}
    </section>
  );
}
