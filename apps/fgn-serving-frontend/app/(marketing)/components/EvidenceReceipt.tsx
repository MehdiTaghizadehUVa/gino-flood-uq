import type { EvidenceRegistry } from "../content";
import type { CaseStudyManifest } from "./caseStudyTypes";

export function EvidenceReceipt({
  manifest,
  evidence
}: {
  manifest: CaseStudyManifest;
  evidence: EvidenceRegistry;
}) {
  const receiptRows = [
    ["Model bundle", manifest.provenance.bundleId],
    ["Model code commit", manifest.provenance.bundleGitCommit],
    ["Ensemble policy", manifest.provenance.ensemblePolicy],
    ["Calibration", manifest.provenance.calibrationMode],
    ["Initial-condition library", manifest.provenance.initialConditionLibraryId],
    ["Reference scope", manifest.provenance.initialConditionReferenceScope],
    ["Seed", manifest.provenance.seed],
    ["Forecast timestep", `${Number(manifest.provenance.dtSeconds) / 60} min`],
    ["Terrain hash", manifest.provenance.terrainSha256],
    ["Terrain CRS path", `${manifest.provenance.terrainSourceCrs} to ${manifest.provenance.terrainTargetCrs}`]
  ] as const;
  return (
    <div className="case-study-receipt-layout">
      <div className="performance-receipt">
        <p className="micro-label">Measured comparison</p>
        <div className="evidence-claim-grid">
          {evidence.claims.map((claim) => (
            <article key={claim.id}>
              <strong>{claim.value}</strong>
              <h3>{claim.label}</h3>
              <p>{claim.scope}</p>
            </article>
          ))}
        </div>
        <div className="performance-scope-grid">
          <div>
            <strong>Full service workflow</strong>
            <span>{manifest.performance.workflow.hardware}</span>
            <p>{manifest.performance.workflow.scope}</p>
          </div>
          <div>
            <strong>Forward-only method comparison</strong>
            <span>{manifest.performance.comparison.hardware}</span>
            <p>{manifest.performance.comparison.sample}; {manifest.performance.comparison.ensembleBudget.toLowerCase()}.</p>
          </div>
        </div>
      </div>
      <details className="evidence-receipt">
        <summary>Open scientific provenance receipt</summary>
        <dl>
          {receiptRows.map(([label, value]) => (
            <div key={label}><dt>{label}</dt><dd>{String(value || "Not recorded")}</dd></div>
          ))}
        </dl>
        <p>{manifest.researchDisclaimer}</p>
      </details>
    </div>
  );
}
