import type { CaseStudyManifest } from "./caseStudyTypes";
import { EvidenceCaption } from "./EvidenceCaption";

type Historical = CaseStudyManifest["historicalValidation"];

export function HistoricalValidation({ historical }: { historical: Historical }) {
  return (
    <div className="historical-validation">
      <div className="historical-threshold-note">
        <strong>Validation threshold: WD &gt; {historical.thresholdM.toFixed(2)} m</strong>
        <p>{historical.note}</p>
      </div>
      {historical.events.map((event) => (
        <article key={event.eventId} className="historical-event-row">
          <header>
            <span>{event.eventId.replaceAll("_", " / ")}</span>
            <h4>{event.label}</h4>
          </header>
          <figure className="case-study-figure">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={event.probabilitySrc} alt={`${event.label} probability of maximum depth exceeding 0.10 meters`} width={1400} height={1080} loading="lazy" />
            <EvidenceCaption
              title="Maximum-event exceedance probability"
              insight="Shows how consistently production ensemble members place meaningful flooding during the historical event."
              method="Probability is the fraction of calibrated depth members whose cellwise maximum exceeds 0.10 m. Values below 0.10 probability are transparent. The white, dark-haloed line is the HEC-RAS reference contour."
            />
          </figure>
          <figure className="case-study-figure">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={event.intervalWidthSrc} alt={`${event.label} maximum 90 percent interval-width map`} width={1400} height={1080} loading="lazy" />
            <EvidenceCaption
              title="Maximum-event 90% interval width"
              insight="Locates where the memberwise maximum response remains more dispersed across the event."
              method="Width is p95 minus p05 of memberwise maximum depth. Values below 0.08 m are transparent, and every storm uses the same color scale."
            />
          </figure>
          <figure className="case-study-figure historical-trajectory">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={event.trajectorySrc} alt={`${event.label} predicted and HEC-RAS wettable-domain inundated-area trajectories`} width={900} height={520} loading="lazy" />
            <EvidenceCaption
              title="Wettable-domain extent through time"
              insight="Compares the production ensemble's extent range with the HEC-RAS reference trajectory rather than only one peak value."
              method="The band is the 5th-95th percentile across production members; the solid line is the median and the dashed line is the HEC-RAS reference. This is held-out research evidence, not an operational guarantee."
            />
          </figure>
        </article>
      ))}
    </div>
  );
}
