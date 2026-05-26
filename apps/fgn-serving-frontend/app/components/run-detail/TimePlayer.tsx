import type { ReactNode } from "react";

type TimePlayerProps = {
  children: ReactNode;
};

export function TimePlayer({ children }: TimePlayerProps) {
  return <div className="time-player">{children}</div>;
}
