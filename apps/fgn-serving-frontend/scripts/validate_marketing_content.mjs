import assert from "node:assert/strict";
import { readFile, readdir, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const marketingRoot = path.join(frontendRoot, "app", "(marketing)");
const caseStudyManifestPath = path.join(frontendRoot, "public", "marketing", "portsmouth", "manifest.json");
const [contentSource, pageSource, evidenceSource, caseStudyManifestSource] = await Promise.all([
  readFile(path.join(marketingRoot, "content.ts"), "utf8"),
  readFile(path.join(marketingRoot, "page.tsx"), "utf8"),
  readFile(path.join(marketingRoot, "evidence.json"), "utf8"),
  readFile(caseStudyManifestPath, "utf8")
]);

const marketingFiles = await readdir(marketingRoot, { recursive: true, withFileTypes: true });
const marketingSources = await Promise.all(
  marketingFiles
    .filter((entry) => entry.isFile() && /\.(ts|tsx)$/.test(entry.name))
    .map((entry) => readFile(path.join(entry.parentPath, entry.name), "utf8"))
);
const allMarketingSource = marketingSources.join("\n");
const demoToneSource = (
  await Promise.all(
    [
      "app/components/AppShell.tsx",
      "app/components/ResearchNotice.tsx",
      "app/(demo)/layout.tsx",
      "app/(demo)/demo/page.tsx",
      "app/(demo)/demo/runs/[runId]/page.tsx",
      "app/(demo)/admin/page.tsx",
      "app/(demo)/admin/monitoring/page.tsx"
    ].map((relativePath) => readFile(path.join(frontendRoot, relativePath), "utf8"))
  )
).join("\n");
const visibleToneSource = `${allMarketingSource}\n${caseStudyManifestSource}\n${demoToneSource}`.toLowerCase();

for (const retiredPhrase of [
  "research only",
  "research status",
  "not for emergency",
  "operational flood forecast",
  "research service",
  "research demo",
  "research account",
  "research workspace",
  "research access",
  "research platform",
  "research workflow",
  "research safeguard",
  "research history",
  "research guardrail",
  "research server",
  "scientific review"
]) {
  assert.ok(!visibleToneSource.includes(retiredPhrase), `Retired research-only tone found: ${retiredPhrase}`);
}

const evidence = JSON.parse(evidenceSource);
assert.equal(evidence.caseStudy, "Portsmouth, Virginia", "Evidence must identify the Portsmouth case study.");
assert.ok(Array.isArray(evidence.claims) && evidence.claims.length >= 3, "Evidence must contain benchmark claims.");

for (const claim of evidence.claims) {
  for (const field of ["id", "value", "label", "scope", "sample", "hardware", "measurement", "sourceArtifact"]) {
    assert.ok(claim[field], `Evidence claim ${claim.id ?? "<unknown>"} is missing ${field}.`);
  }
  assert.equal(claim.measured, true, `Evidence claim ${claim.id} must use measured, not extrapolated, results.`);
}

const universalCopy = contentSource
  .replace(/export const portsmouthCaseStudy[\s\S]*?export const calibrationCurve/, "export const calibrationCurve")
  .concat(pageSource.replace(/<CaseStudyEvidence[\s\S]*?\/>/, ""));

for (const deploymentSpecific of ["5,904", "14.2 minutes", "60-member", "94-step", "RTX 4090"]) {
  assert.ok(!universalCopy.includes(deploymentSpecific), `${deploymentSpecific} must remain inside Portsmouth evidence.`);
}

for (const prohibitedClaim of [
  "works anywhere immediately",
  "improves with every storm",
  "perfectly calibrated",
  "when we say 30%, it means 30%",
  "operational flood guidance"
]) {
  assert.ok(!universalCopy.toLowerCase().includes(prohibitedClaim), `Prohibited claim found: ${prohibitedClaim}`);
}

for (const internalPhrase of [
  "one rendering contract",
  "production-configuration hindcasts",
  "input fingerprints",
  "arrive with its receipt",
  "the service knows when",
  "computational location",
  "scientifically appropriate units",
  "public page does not ship",
  "deployment contract"
]) {
  assert.ok(!visibleToneSource.includes(internalPhrase), `Internal implementation language found: ${internalPhrase}`);
}

assert.ok(pageSource.includes("Request a pilot"), "Request a pilot must be the primary conversion action.");
assert.ok(pageSource.includes("Explore the Portsmouth demo"), "The demo must be identified as the Portsmouth deployment.");
assert.ok(contentSource.includes("expert-reviewed"), "Monitoring copy must preserve human review.");
assert.ok(
  allMarketingSource.includes("Tested consistently across three held-out historical storms."),
  "Historical validation must lead with buyer-facing evidence."
);

const manifest = JSON.parse(caseStudyManifestSource);
assert.equal(manifest.schemaVersion, 1, "Unsupported Portsmouth case-study schema.");
assert.equal(manifest.provenance.ensemblePolicy, "3 checkpoints x 20 members");
assert.equal(manifest.provenance.dtSeconds, 900);
assert.equal(manifest.provenance.forecastSteps, 94);
assert.equal(manifest.flagship.eventId, "2011_IRENE");
assert.equal(manifest.flagship.thresholdM, 0.3);
assert.equal(manifest.flagship.products[0].id, "meanDepth", "Mean depth must be the first Portsmouth animation product.");
assert.ok(
  Number.isInteger(manifest.flagship.peakMeanDepthTimeIndex) && manifest.flagship.peakMeanDepthTimeIndex >= 0,
  "The depth story requires a physical peak area-weighted mean-depth lead."
);
assert.equal(manifest.flagship.hero.product, "Calibrated mean water depth");
assert.equal(manifest.flagship.hero.displayFloorM, 0.05);
assert.equal(manifest.flagship.hero.selection, "Peak expected footprint above 0.30 m");
assert.equal(manifest.flagship.hero.frameCount, 32, "Hero animation must use the scientific story frame set.");
assert.ok(
  manifest.flagship.hero.durationSeconds >= 6 && manifest.flagship.hero.durationSeconds <= 12,
  "Hero animation duration must remain calm enough to read and short enough to loop."
);
const heroSequence = JSON.parse(
  await readFile(path.join(frontendRoot, "public", manifest.flagship.hero.sequenceSrc), "utf8")
);
assert.equal(heroSequence.frameCount, manifest.flagship.hero.frameCount);
assert.equal(heroSequence.frameRate, manifest.flagship.hero.frameRate);
assert.equal(heroSequence.frames.length, manifest.flagship.hero.frameCount);
const heroTimeIndices = heroSequence.frames.map((frame) => frame.timeIndex);
const heroTimeIndexSet = new Set(heroTimeIndices);
assert.deepEqual(
  heroTimeIndices,
  manifest.flagship.products
    .find((product) => product.id === "meanDepth")
    .frames.filter((frame) => heroTimeIndexSet.has(frame.timeIndex))
    .map((frame) => frame.timeIndex),
  "Hero animation milestones must remain aligned with exact public mean-depth evidence frames."
);
assert.deepEqual(
  manifest.flagship.decomposition.maps.map((item) => item.label),
  ["Epistemic uncertainty", "Aleatoric uncertainty"],
  "Public decomposition maps must use the approved uncertainty terminology."
);
const peakMeanFrame = manifest.flagship.products
  .find((product) => product.id === "meanDepth")
  .frames.find((frame) => frame.timeIndex === manifest.flagship.peakAreaTimeIndex);
assert.equal(manifest.flagship.hero.leadHours, peakMeanFrame.leadHours);
assert.equal(manifest.historicalValidation.thresholdM, 0.1);
assert.deepEqual(
  manifest.historicalValidation.events.map((event) => event.eventId),
  ["2023_OPHELIA", "2003_ISABEL", "2011_IRENE"]
);
assert.equal(new Set(manifest.historicalValidation.events.map((event) => event.runId)).size, 3);
assert.deepEqual(
  Object.fromEntries(manifest.flagship.products.map((product) => [product.id, product.displayFloor])),
  { meanDepth: 0.05, probability: 0.1, intervalWidth: 0.08 }
);
assert.deepEqual(
  manifest.flagship.overviewMaps.map((item) => item.id),
  ["probability", "interval_width", "arrival_time", "mean_depth"],
  "All public overview maps must come from the canonical spatial renderer."
);
assert.equal(manifest.displayPolicy.terrainExtendsBeyondMesh, true);
assert.equal(manifest.displayPolicy.viewportPolicy, "mesh_bounds_plus_2p5_percent");
const [terrainLeft, terrainRight, terrainBottom, terrainTop] = manifest.displayPolicy.terrainViewport;
const [meshLeft, meshRight, meshBottom, meshTop] = manifest.displayPolicy.meshExtent;
assert.ok(
  terrainLeft < meshLeft && terrainRight > meshRight && terrainBottom < meshBottom && terrainTop > meshTop,
  "The external DEM viewport must extend beyond the computational mesh on every side."
);
const expectedXPad = (meshRight - meshLeft) * 0.025;
const expectedYPad = (meshTop - meshBottom) * 0.025;
assert.ok(Math.abs((meshLeft - terrainLeft) - expectedXPad) < 0.01);
assert.ok(Math.abs((terrainRight - meshRight) - expectedXPad) < 0.01);
assert.ok(Math.abs((meshBottom - terrainBottom) - expectedYPad) < 0.01);
assert.ok(Math.abs((terrainTop - meshTop) - expectedYPad) < 0.01);
const [sourceLeft, sourceRight, sourceBottom, sourceTop] = manifest.displayPolicy.terrainSourceExtent;
assert.ok(sourceLeft <= terrainLeft && sourceRight >= terrainRight && sourceBottom <= terrainBottom && sourceTop >= terrainTop);

for (const product of manifest.flagship.products) {
  assert.equal(product.frames.length, 94, `${product.id} must contain every physical forecast step.`);
  assert.equal(product.animation.frameCount, product.frames.length, `${product.id} video must cover every source frame.`);
  assert.equal(product.animation.sourceFrameRate, 6, `${product.id} source playback rate changed unexpectedly.`);
  assert.equal(product.animation.playbackFrameRate, 24, `${product.id} display video must be encoded at 24 fps.`);
  assert.ok(
    Math.abs(product.animation.durationSeconds - product.animation.frameCount / product.animation.sourceFrameRate) < 0.001,
    `${product.id} video duration must preserve the complete source horizon.`
  );
  assert.ok(
    product.animation.interpolation.includes("blended"),
    `${product.id} video must disclose presentation-only frame blending.`
  );
  assert.deepEqual(
    product.frames.map((frame) => frame.timeIndex),
    Array.from({ length: 94 }, (_, index) => index),
    `${product.id} must preserve the complete ordered rollout without temporal interpolation.`
  );
  assert.ok(
    product.frames.some((frame) => frame.timeIndex === manifest.flagship.peakAreaTimeIndex),
    `${product.id} must include the peak-footprint lead.`
  );
  assert.ok(
    product.frames.some((frame) => frame.timeIndex === manifest.flagship.peakDisagreementTimeIndex),
    `${product.id} must include the peak-disagreement lead.`
  );
}

assert.ok(!/100[- ]member/i.test(caseStudyManifestSource), "Public evidence must not mix archived 100-member results.");

const assetPaths = [
  manifest.flagship.posterSrc,
  manifest.flagship.hero.src,
  manifest.flagship.hero.posterSrc,
  manifest.flagship.hero.mp4Src,
  manifest.flagship.hero.webmSrc,
  manifest.flagship.hero.sequenceSrc,
  ...manifest.flagship.products.flatMap((product) => product.frames.map((frame) => frame.src)),
  ...manifest.flagship.products.flatMap((product) => [
    product.animation.mp4Src,
    product.animation.posterSrc
  ]),
  ...manifest.flagship.overviewMaps.map((item) => item.src),
  ...manifest.flagship.snapshot.map((item) => item.src),
  ...manifest.flagship.locations.flatMap((item) => [item.mapSrc, item.panelSrc]),
  ...manifest.flagship.decomposition.maps.map((item) => item.src),
  ...manifest.historicalValidation.events.flatMap((event) => [
    event.probabilitySrc,
    event.intervalWidthSrc,
    event.trajectorySrc
  ])
];

for (const assetPath of new Set(assetPaths)) {
  assert.ok(assetPath.startsWith("/marketing/portsmouth/"), `Unexpected public asset path: ${assetPath}`);
  await stat(path.join(frontendRoot, "public", assetPath));
}

const posterStats = await stat(path.join(frontendRoot, "public", manifest.flagship.posterSrc));
assert.ok(posterStats.size <= 180 * 1024, `Case-study poster is ${(posterStats.size / 1024).toFixed(1)} KB; limit is 180 KB.`);
const heroStats = await stat(path.join(frontendRoot, "public", manifest.flagship.hero.src));
assert.ok(heroStats.size <= 350 * 1024, `Hero map is ${(heroStats.size / 1024).toFixed(1)} KB; limit is 350 KB.`);
const heroPosterStats = await stat(path.join(frontendRoot, "public", manifest.flagship.hero.posterSrc));
assert.ok(
  heroPosterStats.size <= 180 * 1024,
  `Hero poster is ${(heroPosterStats.size / 1024).toFixed(1)} KB; limit is 180 KB.`
);
for (const videoSrc of [manifest.flagship.hero.mp4Src, manifest.flagship.hero.webmSrc]) {
  const videoStats = await stat(path.join(frontendRoot, "public", videoSrc));
  assert.ok(
    videoStats.size <= 4 * 1024 * 1024,
    `Hero video ${videoSrc} is ${(videoStats.size / 1024 / 1024).toFixed(2)} MB; limit is 4 MB.`
  );
}
for (const product of manifest.flagship.products) {
  const videoStats = await stat(path.join(frontendRoot, "public", product.animation.mp4Src));
  assert.ok(
    videoStats.size <= 8 * 1024 * 1024,
    `Product video ${product.animation.mp4Src} is ${(videoStats.size / 1024 / 1024).toFixed(2)} MB; limit is 8 MB.`
  );
}

assert.ok(!allMarketingSource.includes("/api/"), "The public marketing route must not call runtime APIs.");
assert.ok(pageSource.includes("<HeroFloodVideo"), "The marketing hero must use the managed video component.");
for (const legacyTerm of ["between-model", "within-model", "between model", "within model"]) {
  assert.ok(
    !allMarketingSource.toLowerCase().includes(legacyTerm),
    `Legacy public uncertainty term found: ${legacyTerm}`
  );
}
for (const maskingPhrase of [
  "values below",
  "values under",
  "are hidden",
  "are transparent",
  "display floor",
  "display cutoff"
]) {
  assert.ok(
    !allMarketingSource.toLowerCase().includes(maskingPhrase),
    `Public masking-language found: ${maskingPhrase}`
  );
}
assert.ok(pageSource.includes('<math display="block"'), "The variance decomposition must use semantic MathML.");
assert.ok(pageSource.includes("epistemic uncertainty"));
assert.ok(pageSource.includes("aleatoric uncertainty"));
assert.ok(!allMarketingSource.includes("MotionPreferenceToggle"), "Marketing motion must not require a visitor control.");
assert.ok(!allMarketingSource.includes("hero-motion-toggle"), "Hero playback must not expose a pause/play control.");
for (const legacyMapPath of [
  "/marketing/mean-depth.webp",
  "/marketing/exceedance-probability.webp",
  "/marketing/uncertainty-width.webp",
  "/marketing/arrival-time.webp"
]) {
  assert.ok(!allMarketingSource.includes(legacyMapPath), `Legacy spatial asset is still public: ${legacyMapPath}`);
}

console.log("Marketing content validation passed.");
