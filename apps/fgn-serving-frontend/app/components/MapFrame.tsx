import type { ReactNode } from "react";

type MapFrameProps = {
  children: ReactNode;
  className?: string;
};

export function MapFrame({ children, className = "" }: MapFrameProps) {
  return <div className={`map-frame ${className}`.trim()}>{children}</div>;
}
