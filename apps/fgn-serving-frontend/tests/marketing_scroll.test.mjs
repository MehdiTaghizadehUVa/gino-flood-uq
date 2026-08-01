import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  calibrationLayerState,
  depthStoryMilestoneFrames,
  milestoneIndexFromFrame,
  sceneProgressFromBounds,
  stepIndexFromProgress
} from "../app/(marketing)/components/scrollSceneMath.mjs";

const scrollRevealSource = readFileSync(
  new URL("../app/(marketing)/components/ScrollReveal.tsx", import.meta.url),
  "utf8"
);
const marketingCss = readFileSync(
  new URL("../app/(marketing)/marketing.css", import.meta.url),
  "utf8"
);

test("scene progress spans the sticky scroll range and clamps outside it", () => {
  assert.equal(sceneProgressFromBounds({ top: 200, height: 2700, viewportHeight: 900 }), 0);
  assert.equal(sceneProgressFromBounds({ top: -900, height: 2700, viewportHeight: 900 }), 0.5);
  assert.equal(sceneProgressFromBounds({ top: -1800, height: 2700, viewportHeight: 900 }), 1);
  assert.equal(sceneProgressFromBounds({ top: -2400, height: 2700, viewportHeight: 900 }), 1);
});

test("scroll progress selects every narrative step including both endpoints", () => {
  assert.equal(stepIndexFromProgress(0, 5), 0);
  assert.equal(stepIndexFromProgress(0.2, 5), 1);
  assert.equal(stepIndexFromProgress(0.799, 5), 3);
  assert.equal(stepIndexFromProgress(1, 5), 4);
});

test("timed forecast playback selects evidence captions from physical frame milestones", () => {
  const milestones = [0, 22, 46, 66, 93];
  assert.equal(milestoneIndexFromFrame(0, milestones), 0);
  assert.equal(milestoneIndexFromFrame(21, milestones), 0);
  assert.equal(milestoneIndexFromFrame(22, milestones), 1);
  assert.equal(milestoneIndexFromFrame(65, milestones), 2);
  assert.equal(milestoneIndexFromFrame(93, milestones), 4);
  assert.equal(milestoneIndexFromFrame(10, []), 0);
});

test("mean-depth story advances through buildup, peak depth, and recession", () => {
  assert.deepEqual(depthStoryMilestoneFrames(93, 66), [0, 22, 44, 66, 93]);
  assert.deepEqual(depthStoryMilestoneFrames(7, 6), [0, 2, 4, 6, 7]);
});

test("calibration evidence points appear before the fitted mapping line", () => {
  assert.deepEqual(calibrationLayerState(0), { pointsVisible: true, curveVisible: false });
  assert.deepEqual(calibrationLayerState(1), { pointsVisible: true, curveVisible: true });
  assert.deepEqual(calibrationLayerState(2), { pointsVisible: true, curveVisible: true });
});

test("mobile narrative items reveal as they enter the viewport", () => {
  assert.match(scrollRevealSource, /data-mobile-reveal/);
  assert.match(scrollRevealSource, /max-width:\s*899px/);
  assert.match(scrollRevealSource, /MOBILE_ITEM_ROOT_MARGIN\s*=\s*"0px 0px -28% 0px"/);
  assert.match(marketingCss, /\[data-mobile-reveal\]/);
  assert.match(marketingCss, /\[data-mobile-reveal-visible="true"\]/);
  assert.match(marketingCss, /translate3d\(0, 44px, 0\) scale\(0\.985\)/);
});

test("variance equation adapts to its analysis column without overflowing", () => {
  assert.match(marketingCss, /container:\s*uncertainty-equation\s*\/\s*inline-size/);
  assert.match(marketingCss, /@container uncertainty-equation \(max-width:\s*720px\)/);
  assert.match(marketingCss, /\.variance-equation-compact\s*\{[\s\S]*?display:\s*none/);
  assert.match(marketingCss, /overflow:\s*hidden/);
});
