import manifestJson from "../../../public/marketing/portsmouth/manifest.json";
import type { EvidenceRegistry } from "../content";
import { DecisionSnapshot } from "./DecisionSnapshot";
import { EvidenceReceipt } from "./EvidenceReceipt";
import { ForecastStoryPlayer } from "./ForecastStoryPlayer";
import { HistoricalValidation } from "./HistoricalValidation";
import { LocationEvidence } from "./LocationEvidence";
import { UncertaintyDecomposition } from "./UncertaintyDecomposition";
import type { CaseStudyManifest } from "./caseStudyTypes";

const manifest = manifestJson as unknown as CaseStudyManifest;

export function PortsmouthCaseStudy({ evidence }: { evidence: EvidenceRegistry }) {
  return (
    <div className="portsmouth-case-study">
      <div className="case-study-stats" aria-label="Irene 2011 evidence summary">
        {manifest.flagship.metrics.map((metric) => (
          <div key={metric.label}><strong>{metric.value}</strong><span>{metric.label}</span></div>
        ))}
      </div>

      <section className="case-study-chapter" aria-labelledby="forecast-story-title">
        <header><span>01 / EVENT STORY</span><h3 id="forecast-story-title">Follow the probability field through Irene 2011.</h3><p>A named historical forcing becomes a time-resolved view of probability, depth, and ensemble width.</p></header>
        <ForecastStoryPlayer eventLabel={manifest.flagship.label} products={manifest.flagship.products} posterSrc={manifest.flagship.posterSrc} />
      </section>

      <section className="case-study-chapter" aria-labelledby="snapshot-title">
        <header><span>02 / SAME LEAD, THREE QUESTIONS</span><h3 id="snapshot-title">Probability, central response, and width belong together.</h3><p>The maps share one event and lead time while retaining scientifically appropriate units and display scales.</p></header>
        <DecisionSnapshot products={manifest.flagship.snapshot} />
      </section>

      <section className="case-study-chapter" aria-labelledby="location-title">
        <header><span>03 / LOCATION EVIDENCE</span><h3 id="location-title">Move from a regional map to one computational location.</h3><p>Representative locations demonstrate how FloodUQ exposes the ensemble evidence behind a mapped result.</p></header>
        <LocationEvidence locations={manifest.flagship.locations} />
      </section>

      <section className="case-study-chapter" aria-labelledby="decomposition-title">
        <header><span>04 / UNCERTAINTY DECOMPOSITION</span><h3 id="decomposition-title">Separate epistemic uncertainty from aleatoric uncertainty.</h3><p>The nested production ensemble preserves both components instead of collapsing all variability into one spread number.</p></header>
        <UncertaintyDecomposition decomposition={manifest.flagship.decomposition} />
      </section>

      <section className="case-study-chapter" aria-labelledby="validation-title">
        <header><span>05 / HELD-OUT HISTORICAL EVIDENCE</span><h3 id="validation-title">One rendering contract across three named storms.</h3><p>Production-configuration hindcasts are evaluated against aligned HEC-RAS evidence using a separate historical threshold.</p></header>
        <HistoricalValidation historical={manifest.historicalValidation} />
      </section>

      <section className="case-study-chapter" aria-labelledby="receipt-title">
        <header><span>06 / PERFORMANCE AND PROVENANCE</span><h3 id="receipt-title">A measured result should arrive with its receipt.</h3><p>Full-workflow runtime and forward-only model comparison remain separate, scoped claims with auditable sources.</p></header>
        <EvidenceReceipt manifest={manifest} evidence={evidence} />
      </section>

      <div className="case-study-conversion">
        <div><span>PORTSMOUTH IS THE PROOF CASE</span><h3>Bring calibrated flood uncertainty to your coastal domain.</h3><p>Define the terrain, reference evidence, thresholds, and validation contract for a managed pilot deployment.</p></div>
        <div className="marketing-actions"><a className="marketing-button primary" href="mailto:jrj6wm@virginia.edu?subject=FloodUQ%20managed%20deployment%20pilot">Request a pilot</a><a className="marketing-button secondary" href="/demo">Explore the Portsmouth demo</a></div>
      </div>
    </div>
  );
}
