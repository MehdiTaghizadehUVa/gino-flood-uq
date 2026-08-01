import type { CaseStudyManifest } from "./caseStudyTypes";
import { caseStudyAsset } from "../caseStudyAsset";
import { EvidenceCaption } from "./EvidenceCaption";

type Decomposition = CaseStudyManifest["flagship"]["decomposition"];

export function UncertaintyDecomposition({ decomposition }: { decomposition: Decomposition }) {
  const betweenPercent = Math.round(decomposition.betweenVarianceShare * 100);
  return (
    <div className="decomposition-evidence">
      <div className="decomposition-summary">
        <div>
          <span>Share attributed to epistemic uncertainty</span>
          <strong>{betweenPercent}%</strong>
        </div>
        <div className="variance-share" aria-label={`${betweenPercent}% model uncertainty and ${100 - betweenPercent}% outcome variability`}>
          <span style={{ width: `${betweenPercent}%` }} />
        </div>
        <p>
          At +{decomposition.leadHours.toFixed(2)} h, epistemic uncertainty accounts for {betweenPercent}% of the
          measured forecast variation. Here, epistemic uncertainty means disagreement between independently trained
          models. This share changes through time and does not identify a physical cause.
        </p>
      </div>
      <div className="decomposition-grid">
        {decomposition.maps.map((map) => (
          <figure key={map.id} className="case-study-figure">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={caseStudyAsset(map.src)}
              alt={`${map.id === "betweenModel" ? "Epistemic uncertainty" : "Aleatoric uncertainty"} over the Portsmouth terrain`}
              width={1400}
              height={1080}
              loading="lazy"
            />
            <EvidenceCaption
              title={map.id === "betweenModel" ? "Epistemic uncertainty" : "Aleatoric uncertainty"}
              insight={
                map.id === "betweenModel"
                  ? "Model uncertainty (epistemic) is larger where independently trained models give different central forecasts."
                  : "Outcome variability (aleatoric) is larger where one trained model retains a wider range of plausible forecasts."
              }
              method="Both maps use the same meter scale. Standard deviation measures spread: larger values mean the forecasts are farther apart. The left map compares the central forecast from each trained model; the right map measures variation among plausible forecasts from the same model."
            />
          </figure>
        ))}
      </div>
    </div>
  );
}
