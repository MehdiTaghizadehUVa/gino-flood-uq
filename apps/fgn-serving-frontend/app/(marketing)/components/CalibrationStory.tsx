"use client";

import type { NumberedContentItem } from "../content";
import { CalibrationGraphic } from "./CalibrationGraphic";
import { useScrollScene } from "./useScrollScene";

export function CalibrationStory({ notes }: { notes: readonly NumberedContentItem[] }) {
  const { rootRef, enabled, activeStep } = useScrollScene({ stepCount: notes.length });

  return (
    <div
      ref={rootRef}
      className={`scroll-story calibration-story${enabled ? " is-scroll-enabled" : ""}`}
      data-scroll-scene="calibration"
      data-active-step={activeStep}
    >
      <div className="scroll-story-visual calibration-story-visual">
        <CalibrationGraphic stage={activeStep} />
        <div className="calibration-stage-label" aria-live="polite">
          <span>{notes[activeStep]?.number}</span>
          <strong>{notes[activeStep]?.title}</strong>
        </div>
        <div className="scroll-scene-meter" aria-hidden="true"><span /></div>
      </div>
      <ol className="scroll-story-steps calibration-story-steps">
        {notes.map((note, index) => (
          <li key={note.number} className={index === activeStep ? "active" : ""} aria-current={index === activeStep ? "step" : undefined}>
            <span>{note.number}</span>
            <h3>{note.title}</h3>
            <p>{note.body}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
