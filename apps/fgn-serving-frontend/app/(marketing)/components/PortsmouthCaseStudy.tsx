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
        <header><span>01 / EVENT STORY</span><h3 id="forecast-story-title">See how flooding develops, not just where it peaks.</h3><p>Follow Irene 2011 through time, then switch between expected depth, exceedance probability, and uncertainty width.</p></header>
        <ForecastStoryPlayer
          eventLabel={manifest.flagship.label}
          products={manifest.flagship.products}
          posterSrc={manifest.flagship.posterSrc}
          peakMeanDepthTimeIndex={manifest.flagship.peakMeanDepthTimeIndex}
        />
      </section>

      <section className="case-study-chapter" aria-labelledby="snapshot-title">
        <header><span>02 / ONE MOMENT, THREE QUESTIONS</span><h3 id="snapshot-title">Expected depth, probability, and uncertainty answer different questions.</h3><p>View all three at the same lead time to understand the expected outcome, its likelihood, and where forecasts diverge.</p></header>
        <DecisionSnapshot products={manifest.flagship.snapshot} />
      </section>

      <section className="case-study-chapter" aria-labelledby="location-title">
        <header><span>03 / LOCATION EVIDENCE</span><h3 id="location-title">Inspect the forecast at a specific location.</h3><p>Move from a regional pattern to the full range of depth, exceedance probability, and arrival timing at one selected point.</p></header>
        <LocationEvidence locations={manifest.flagship.locations} />
      </section>

      <section className="case-study-chapter" aria-labelledby="decomposition-title">
        <header><span>04 / UNCERTAINTY DECOMPOSITION</span><h3 id="decomposition-title">See which source of uncertainty drives the forecast.</h3><p>Separate epistemic uncertainty from aleatoric uncertainty so high disagreement can lead to a more focused review.</p></header>
        <UncertaintyDecomposition decomposition={manifest.flagship.decomposition} />
      </section>

      <section className="case-study-chapter" aria-labelledby="validation-title">
        <header><span>05 / HELD-OUT HISTORICAL EVIDENCE</span><h3 id="validation-title">Tested consistently across three held-out historical storms.</h3><p>Ophelia, Isabel, and Irene are evaluated with the same deployed model, map scales, and aligned HEC-RAS comparison method.</p></header>
        <HistoricalValidation historical={manifest.historicalValidation} />
      </section>

      <section className="case-study-chapter" aria-labelledby="receipt-title">
        <header><span>06 / PERFORMANCE AND PROVENANCE</span><h3 id="receipt-title">See the evidence behind every performance claim.</h3><p>Runtime, forecast skill, model version, calibration, hardware, and event provenance are presented together so each result can be judged in context.</p></header>
        <EvidenceReceipt manifest={manifest} evidence={evidence} />
      </section>

      <div className="case-study-conversion">
        <div><span>PORTSMOUTH IS THE PROOF CASE</span><h3>Bring calibrated flood uncertainty to your coastal domain.</h3><p>A pilot combines your terrain, forcing scenarios, reference simulations, decision thresholds, and validation needs into a practical deployment plan.</p></div>
        <div className="marketing-actions"><a className="marketing-button primary" href="mailto:jrj6wm@virginia.edu?subject=FloodUQ%20managed%20deployment%20pilot">Request a pilot</a><a className="marketing-button secondary" href="/demo">Explore the Portsmouth demo</a></div>
      </div>
    </div>
  );
}
