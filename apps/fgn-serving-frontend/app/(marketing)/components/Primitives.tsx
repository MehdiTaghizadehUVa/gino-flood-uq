import type { ReactNode } from "react";

type SectionProps = {
  id?: string;
  eyebrow: string;
  title: ReactNode;
  intro?: ReactNode;
  children: ReactNode;
  className?: string;
};

export function Section({ id, eyebrow, title, intro, children, className = "" }: SectionProps) {
  return (
    <section id={id} className={`marketing-section ${className}`} data-reveal>
      <div className="marketing-section-inner">
        <header className="section-heading">
          <p className="marketing-eyebrow">{eyebrow}</p>
          <h2>{title}</h2>
          {intro ? <div className="section-intro">{intro}</div> : null}
        </header>
        {children}
      </div>
    </section>
  );
}

export function MicroLabel({ children }: { children: ReactNode }) {
  return <span className="micro-label">{children}</span>;
}
