"use client";

import Image from "next/image";
import { caseStudyAsset } from "../caseStudyAsset";
import type { NumberedContentItem } from "../content";
import { useScrollScene } from "./useScrollScene";

const VISUALS = [
  {
    src: "/marketing/portsmouth/overview/mean_depth.webp",
    alt: "Ensemble-mean coastal water-depth map from the Portsmouth deployment",
    label: "Expected water depth",
    insight: "Orient the study around the central ensemble response."
  },
  {
    src: "/marketing/portsmouth/overview/arrival_time.webp",
    alt: "Forecast arrival-time map from the Portsmouth deployment",
    label: "Arrival and persistence",
    insight: "Move from a static footprint to the timing of the response."
  },
  {
    src: "/marketing/portsmouth/overview/probability.webp",
    alt: "Calibrated coastal water-depth exceedance probability map from the Portsmouth deployment",
    label: "Calibrated exceedance probability",
    insight: "Ask how often a study threshold is exceeded across the ensemble."
  },
  {
    src: "/marketing/portsmouth/overview/interval_width.webp",
    alt: "Spatial forecast interval-width map from the Portsmouth deployment",
    label: "Forecast interval width",
    insight: "Locate where plausible members remain widely separated."
  },
  {
    src: "/marketing/portsmouth/overview/probability.webp",
    alt: "Monitored calibrated exceedance-probability product from the Portsmouth deployment",
    label: "Monitored forecast evidence",
    insight: "Keep the input, forecast, calibration, and familiarity assessment connected."
  }
] as const;

export function DecisionRiskStory({ questions }: { questions: readonly NumberedContentItem[] }) {
  const { rootRef, enabled, activeStep } = useScrollScene({ stepCount: questions.length });

  return (
    <div
      ref={rootRef}
      className={`scroll-story decision-risk-story${enabled ? " is-scroll-enabled" : ""}`}
      data-scroll-scene="decision-risk"
      data-active-step={activeStep}
    >
      <div className="scroll-story-visual decision-risk-visual" aria-live="polite">
        <div className="decision-visual-stack">
          {VISUALS.map((visual, index) => (
            <Image
              key={visual.src + index}
              className={index === activeStep ? "active" : ""}
              src={caseStudyAsset(visual.src)}
              alt={index === activeStep ? visual.alt : ""}
              aria-hidden={index !== activeStep}
              width={1115}
              height={929}
              sizes="(max-width: 900px) 100vw, 62vw"
              priority={index === 0}
            />
          ))}
          <div className="decision-visual-caption">
            <span>{VISUALS[activeStep]?.label}</span>
            <strong>{VISUALS[activeStep]?.insight}</strong>
          </div>
          {activeStep === 4 ? (
            <div className="decision-monitoring-badge">
              <span>REFERENCE SCREENING</span>
              <strong>Forecast evidence remains traceable</strong>
            </div>
          ) : null}
        </div>
        <div className="scroll-scene-meter" aria-hidden="true"><span /></div>
      </div>

      <ol className="scroll-story-steps" aria-label="Decision questions supported by FloodUQ">
        {questions.map((question, index) => (
          <li key={question.number} className={index === activeStep ? "active" : ""} aria-current={index === activeStep ? "step" : undefined}>
            <span>{question.number}</span>
            <h3>{question.title}</h3>
            <p>{question.body}</p>
            <div className="scroll-step-visual">
              <Image
                src={caseStudyAsset(VISUALS[index]?.src ?? VISUALS[0].src)}
                alt={VISUALS[index]?.alt ?? VISUALS[0].alt}
                width={1115}
                height={929}
                sizes="100vw"
              />
              <strong>{VISUALS[index]?.insight}</strong>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
