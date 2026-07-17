const clamp01 = (value) => Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));

export function sceneProgressFromBounds({ top, height, viewportHeight }) {
  const scrollRange = Math.max(height - viewportHeight, 1);
  return clamp01(-top / scrollRange);
}

export function stepIndexFromProgress(progress, stepCount) {
  if (stepCount <= 1) return 0;
  return Math.min(stepCount - 1, Math.floor(clamp01(progress) * stepCount));
}

export function milestoneIndexFromFrame(frameIndex, milestoneFrames) {
  if (!milestoneFrames.length) return 0;
  let activeIndex = 0;
  for (let index = 1; index < milestoneFrames.length; index += 1) {
    if (frameIndex < milestoneFrames[index]) break;
    activeIndex = index;
  }
  return activeIndex;
}

export function depthStoryMilestoneFrames(lastIndex, peakMeanDepthIndex) {
  const safeLast = Math.max(0, Math.floor(Number.isFinite(lastIndex) ? lastIndex : 0));
  const safePeak = Math.min(
    safeLast,
    Math.max(0, Math.floor(Number.isFinite(peakMeanDepthIndex) ? peakMeanDepthIndex : 0))
  );
  const earlyResponse = Math.min(safeLast, Math.max(1, Math.round(safePeak / 3)));
  const inlandExpansion = Math.min(
    safeLast,
    Math.max(earlyResponse, Math.round((2 * safePeak) / 3))
  );
  return [0, earlyResponse, inlandExpansion, safePeak, safeLast];
}

export function calibrationLayerState(stage) {
  const normalizedStage = Math.min(2, Math.max(0, Math.floor(Number.isFinite(stage) ? stage : 0)));
  return {
    pointsVisible: true,
    curveVisible: normalizedStage >= 1
  };
}
