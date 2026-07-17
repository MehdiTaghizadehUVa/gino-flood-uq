"use client";

import type { NumberedContentItem } from "../content";
import { MicroLabel } from "./Primitives";
import { useScrollScene } from "./useScrollScene";

export function UncertaintyStory({
  sources,
  equationHtml
}: {
  sources: readonly NumberedContentItem[];
  equationHtml: string;
}) {
  const { rootRef, enabled, activeStep } = useScrollScene({ stepCount: sources.length });

  return (
    <div
      ref={rootRef}
      className={`scroll-story uncertainty-story${enabled ? " is-scroll-enabled" : ""}`}
      data-scroll-scene="uncertainty"
      data-active-step={activeStep}
    >
      <div className="scroll-story-visual uncertainty-story-visual">
        <MicroLabel>Total forecast variance</MicroLabel>
        <div className="math-expression uncertainty-equation" dangerouslySetInnerHTML={{ __html: equationHtml }} />
        <div className="uncertainty-source-key" aria-live="polite">
          {sources.map((source, index) => (
            <div key={source.number} className={index === activeStep ? "active" : ""}>
              <span>{source.number}</span><strong>{source.title}</strong>
            </div>
          ))}
        </div>
        <div className="scroll-scene-meter" aria-hidden="true"><span /></div>
      </div>
      <ol className="scroll-story-steps uncertainty-story-steps">
        {sources.map((source, index) => (
          <li key={source.number} className={index === activeStep ? "active" : ""} aria-current={index === activeStep ? "step" : undefined}>
            <span>{source.number}</span>
            <h3>{source.title}</h3>
            <p>{source.body}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
