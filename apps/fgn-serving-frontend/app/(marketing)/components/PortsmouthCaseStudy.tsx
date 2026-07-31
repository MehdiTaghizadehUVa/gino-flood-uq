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
        <header><span>01 / EVENT STORY</span><h3 id="forecast-story-title">See how flooding develops, not just where it peaks.</h3><p>Follow Irene 2011 through time, then switch between expected depth, the chance of passing 0.30 m, and the width of the forecast range.</p></header>
        <ForecastStoryPlayer
          eventLabel={manifest.flagship.label}
          products={manifest.flagship.products}
          posterSrc={manifest.flagship.posterSrc}
          peakMeanDepthTimeIndex={manifest.flagship.peakMeanDepthTimeIndex}
        />
      </section>

      <section className="case-study-chapter" aria-labelledby="snapshot-title">
        <header><span>02 / ONE MOMENT, THREE QUESTIONS</span><h3 id="snapshot-title">Expected depth, probability, and forecast range answer different questions.</h3><p>View all three at the same moment to understand the average outcome, how often a selected depth is passed, and where plausible forecasts differ.</p></header>
        <DecisionSnapshot products={manifest.flagship.snapshot} />
      </section>

      <section className="case-study-chapter" aria-labelledby="location-title">
        <header><span>03 / LOCATION EVIDENCE</span><h3 id="location-title">Inspect the forecast at a specific location.</h3><p>Move from a regional map to the range of possible depths, the chance of passing 0.30 m, and arrival timing at one selected point.</p></header>
        <LocationEvidence locations={manifest.flagship.locations} />
      </section>

      <section className="case-study-chapter" aria-labelledby="decomposition-title">
        <header><span>04 / WHY FORECASTS DIFFER</span><h3 id="decomposition-title">Separate differences across trained models from the range retained by one model.</h3><p>Model uncertainty (epistemic) compares independently trained models. Outcome variability (aleatoric) describes the range retained by a single model. Seeing both makes high disagreement easier to review.</p></header>
        <UncertaintyDecomposition decomposition={manifest.flagship.decomposition} />
      </section>

      <section className="case-study-chapter" aria-labelledby="validation-title">
        <header><span>05 / HISTORICAL EVIDENCE</span><h3 id="validation-title">Tested on three historical storms not used for training.</h3><p>Ophelia, Isabel, and Irene are evaluated with the same deployed model and compared with HEC-RAS detailed hydraulic simulations.</p></header>
        <HistoricalValidation historical={manifest.historicalValidation} />
      </section>

      <section className="case-study-chapter" aria-labelledby="receipt-title">
        <header><span>06 / PERFORMANCE RECORD</span><h3 id="receipt-title">See the evidence behind every performance claim.</h3><p>Runtime, forecast skill, model version, probability adjustment, hardware, and event source are presented together. This provenance record shows how each result was produced.</p></header>
        <EvidenceReceipt manifest={manifest} evidence={evidence} />
      </section>

      <div className="case-study-conversion">
        <div><span>PORTSMOUTH IS THE PROOF CASE</span><h3>Bring checked flood probabilities to your coastal domain.</h3><p>A collaboration can connect your terrain, water-level and rainfall scenarios, detailed reference simulations, depth thresholds, and evaluation needs in a practical deployment plan.</p></div>
        <div className="marketing-actions"><a className="marketing-button primary" href="mailto:jrj6wm@virginia.edu?subject=FloodUQ%20research%20and%20deployment%20collaboration">Discuss a collaboration</a><a className="marketing-button secondary" href="/demo">Explore the Portsmouth demo</a></div>
      </div>
    </div>
  );
}
