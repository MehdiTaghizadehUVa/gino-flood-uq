import type { ReactNode } from "react";

type FigureCardProps = {
  title?: ReactNode;
  children: ReactNode;
  className?: string;
};

export function FigureCard({ title, children, className = "" }: FigureCardProps) {
  return (
    <figure className={`figure-card ${className}`.trim()}>
      {title ? <h3>{title}</h3> : null}
      {children}
    </figure>
  );
}
