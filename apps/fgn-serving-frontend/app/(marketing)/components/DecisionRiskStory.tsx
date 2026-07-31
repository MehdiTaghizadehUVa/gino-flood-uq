"use client";

import Image from "next/image";
import { caseStudyAsset } from "../caseStudyAsset";
import type { NumberedContentItem } from "../content";
import { useScrollScene } from "./useScrollScene";

const VISUALS = [
  {
    src: "/marketing/portsmouth/overview/mean_depth.webp",
    alt: "Expected coastal water-depth map from the Portsmouth deployment",
    label: "Expected water depth",
    insight: "See the average depth across the full group of plausible forecasts."
  },
  {
    src: "/marketing/portsmouth/overview/arrival_time.webp",
    alt: "Forecast arrival-time map from the Portsmouth deployment",
    label: "Arrival and persistence",
    insight: "See when water may arrive and how long flooding may persist."
  },
  {
    src: "/marketing/portsmouth/overview/probability.webp",
    alt: "Checked probability that coastal water depth passes a selected level in Portsmouth",
    label: "Chance depth passes the selected level",
    insight: "See what share of plausible forecasts pass a depth that matters to the study."
  },
  {
    src: "/marketing/portsmouth/overview/interval_width.webp",
    alt: "Spatial forecast interval-width map from the Portsmouth deployment",
    label: "Width of the 90% forecast range",
    insight: "Find where plausible outcomes are tightly grouped or far apart."
  },
  {
    src: "/marketing/portsmouth/overview/probability.webp",
    alt: "Flood probability product linked to input and model checks in Portsmouth",
    label: "Input and forecast checks",
    insight: "Keep the scenario, forecast, probability adjustment, and familiarity check connected."
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
              <span>SCENARIO FAMILIARITY CHECK</span>
              <strong>Inputs and results stay linked to their evidence</strong>
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
