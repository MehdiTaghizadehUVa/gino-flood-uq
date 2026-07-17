"use client";

import { Check, Database, Search, ShieldCheck } from "lucide-react";
import type { NumberedContentItem } from "../content";
import { useScrollScene } from "./useScrollScene";

export function MonitoringStory({ items }: { items: readonly NumberedContentItem[] }) {
  const { rootRef, enabled, activeStep } = useScrollScene({ stepCount: items.length });
  const activeItem = items[activeStep] ?? items[0];

  return (
    <div
      ref={rootRef}
      className={`scroll-story monitoring-story${enabled ? " is-scroll-enabled" : ""}`}
      data-scroll-scene="monitoring"
      data-active-step={activeStep}
    >
      <div className="scroll-story-visual monitoring-story-visual">
        <div className="monitoring-console-header">
          <span><ShieldCheck size={16} aria-hidden="true" /> Evidence governance</span>
          <strong>Expert review remains in control</strong>
        </div>
        <div className="monitoring-pipeline" aria-label="Active monitoring lifecycle step">
          {items.map((item, index) => (
            <div key={item.number} className={index < activeStep ? "complete" : index === activeStep ? "active" : ""}>
              <span>{index < activeStep ? <Check size={13} aria-hidden="true" /> : item.number}</span>
              <strong>{item.title}</strong>
            </div>
          ))}
        </div>
        <div className="monitoring-active-card" aria-live="polite">
          <span>{activeItem.number} / {String(items.length).padStart(2, "0")}</span>
          <div>{activeStep < 2 ? <Search size={24} aria-hidden="true" /> : <Database size={24} aria-hidden="true" />}</div>
          <h3>{activeItem.title}</h3>
          <p>{activeItem.body}</p>
        </div>
        <div className="scroll-scene-meter" aria-hidden="true"><span /></div>
      </div>
      <ol className="scroll-story-steps monitoring-story-steps">
        {items.map((item, index) => (
          <li key={item.number} className={index === activeStep ? "active" : ""} aria-current={index === activeStep ? "step" : undefined}>
            <span>{item.number}</span>
            <h3>{item.title}</h3>
            <p>{item.body}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
