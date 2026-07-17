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
    ["Model version", manifest.provenance.bundleId],
    ["Code version", manifest.provenance.bundleGitCommit],
    ["Ensemble design", manifest.provenance.ensemblePolicy],
    ["Calibration method", manifest.provenance.calibrationMode],
    ["Initial-state library", manifest.provenance.initialConditionLibraryId],
    ["Initial-state evidence", manifest.provenance.initialConditionReferenceScope],
    ["Seed", manifest.provenance.seed],
    ["Forecast timestep", `${Number(manifest.provenance.dtSeconds) / 60} min`],
    ["Terrain hash", manifest.provenance.terrainSha256],
    ["Terrain CRS path", `${manifest.provenance.terrainSourceCrs} to ${manifest.provenance.terrainTargetCrs}`]
  ] as const;
  return (
    <div className="case-study-receipt-layout">
      <div className="performance-receipt">
        <p className="micro-label">Measured speed and skill</p>
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
            <p>
              {manifest.performance.comparison.sample}; {manifest.performance.comparison.ensembleBudget.toLowerCase()}; {manifest.performance.comparison.timingScope.toLowerCase()}.
            </p>
          </div>
        </div>
      </div>
      <details className="evidence-receipt">
        <summary>View technical provenance</summary>
        <dl>
          {receiptRows.map(([label, value]) => (
            <div key={label}><dt>{label}</dt><dd>{String(value || "Not recorded")}</dd></div>
          ))}
        </dl>
      </details>
    </div>
  );
}
