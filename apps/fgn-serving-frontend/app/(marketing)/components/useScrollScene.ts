"use client";

import { useEffect, useRef, useState } from "react";
import { sceneProgressFromBounds, stepIndexFromProgress } from "./scrollSceneMath.mjs";

type ScrollSceneOptions = {
  stepCount: number;
  minWidth?: number;
  trackProgress?: boolean;
};

export function useScrollScene({ stepCount, minWidth = 900, trackProgress = false }: ScrollSceneOptions) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [enabled, setEnabled] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    const widthQuery = window.matchMedia(`(min-width: ${minWidth}px)`);
    let frameRequest = 0;
    let currentEnabled = false;
    let currentStep = 0;
    let currentProgress = 0;

    const measure = () => {
      frameRequest = 0;
      if (!currentEnabled) return;
      const bounds = root.getBoundingClientRect();
      const nextProgress = sceneProgressFromBounds({
        top: bounds.top,
        height: bounds.height,
        viewportHeight: window.innerHeight
      });
      root.style.setProperty("--scroll-scene-progress", nextProgress.toFixed(4));

      const nextStep = stepIndexFromProgress(nextProgress, stepCount);
      if (nextStep !== currentStep) {
        currentStep = nextStep;
        setActiveStep(nextStep);
      }
      if (trackProgress && Math.abs(nextProgress - currentProgress) >= 0.002) {
        currentProgress = nextProgress;
        setProgress(nextProgress);
      }
    };

    const scheduleMeasure = () => {
      if (!frameRequest) frameRequest = window.requestAnimationFrame(measure);
    };

    const syncMode = () => {
      currentEnabled = widthQuery.matches;
      setEnabled(currentEnabled);
      root.style.removeProperty("--scroll-scene-progress");
      if (!currentEnabled) {
        currentStep = 0;
        currentProgress = 0;
        setActiveStep(0);
        setProgress(0);
        return;
      }
      scheduleMeasure();
    };

    syncMode();
    window.addEventListener("scroll", scheduleMeasure, { passive: true });
    window.addEventListener("resize", scheduleMeasure);
    widthQuery.addEventListener("change", syncMode);

    return () => {
      window.cancelAnimationFrame(frameRequest);
      window.removeEventListener("scroll", scheduleMeasure);
      window.removeEventListener("resize", scheduleMeasure);
      widthQuery.removeEventListener("change", syncMode);
    };
  }, [minWidth, stepCount, trackProgress]);

  return { rootRef, enabled, activeStep, progress };
}
