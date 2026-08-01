import type { CaseStudyManifest } from "./caseStudyTypes";
import { caseStudyAsset } from "../caseStudyAsset";
import { EvidenceCaption } from "./EvidenceCaption";

type Historical = CaseStudyManifest["historicalValidation"];

export function HistoricalValidation({ historical }: { historical: Historical }) {
  return (
    <div className="historical-validation">
      <div className="historical-threshold-note">
        <strong>Comparison depth: water above {historical.thresholdM.toFixed(2)} m</strong>
        <p>
          These storms were not used to train the model. The maps use 0.10 m for comparison with HEC-RAS, a detailed
          hydraulic simulation; the interactive product view uses 0.30 m to answer a different depth question.
        </p>
      </div>
      {historical.events.map((event) => (
        <article key={event.eventId} className="historical-event-row">
          <header>
            <span>{event.eventId.replaceAll("_", " / ")}</span>
            <h4>{event.label}</h4>
          </header>
          <figure className="case-study-figure">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={caseStudyAsset(event.probabilitySrc)} alt={`${event.label} chance that maximum water depth passes 0.10 meters`} width={1400} height={1080} loading="lazy" />
            <EvidenceCaption
              title="Chance maximum depth passes 0.10 m"
              insight="Shows where the calibrated probability is elevated and how that footprint compares with the detailed hydraulic simulation."
              method="The calibrated probability combines the forecast-member exceedance rate with the versioned calibration curve for 0.10 m. The white line marks the HEC-RAS reference footprint."
            />
          </figure>
          <figure className="case-study-figure">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={caseStudyAsset(event.intervalWidthSrc)} alt={`${event.label} maximum 90 percent interval-width map`} width={1400} height={1080} loading="lazy" />
            <EvidenceCaption
              title="Width of the central 90% forecast range"
              insight="Highlights where plausible peak-depth forecasts remain farthest apart during the event."
              method="The width runs from the lower 5% boundary to the upper 95% boundary of plausible maximum depths. Every storm uses the same meter scale."
            />
          </figure>
          <figure className="case-study-figure historical-trajectory">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={caseStudyAsset(event.trajectorySrc)} alt={`${event.label} forecast and detailed-simulation flooded-area paths through time`} width={900} height={520} loading="lazy" />
            <EvidenceCaption
              title="Share of floodable area affected through time"
              insight="Shows whether the forecast range follows the timing and size of the detailed-simulation flood footprint."
              method="The shaded band contains the central 90% of plausible forecasts, the solid line is the middle forecast, and the dashed line is the aligned HEC-RAS reference."
            />
          </figure>
        </article>
      ))}
    </div>
  );
}
