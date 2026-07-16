import type { CaseStudyManifest } from "./caseStudyTypes";
import { EvidenceCaption } from "./EvidenceCaption";

type Decomposition = CaseStudyManifest["flagship"]["decomposition"];

export function UncertaintyDecomposition({ decomposition }: { decomposition: Decomposition }) {
  const betweenPercent = Math.round(decomposition.betweenVarianceShare * 100);
  return (
    <div className="decomposition-evidence">
      <div className="decomposition-summary">
        <div>
          <span>Epistemic share at selected lead</span>
          <strong>{betweenPercent}%</strong>
        </div>
        <div className="variance-share" aria-label={`${betweenPercent}% epistemic and ${100 - betweenPercent}% aleatoric variance`}>
          <span style={{ width: `${betweenPercent}%` }} />
        </div>
        <p>
          The selected lead, +{decomposition.leadHours.toFixed(2)} h, maximizes the area-weighted epistemic
          uncertainty for this event. The share changes through time and should not be read as a causal attribution.
        </p>
      </div>
      <div className="decomposition-grid">
        {decomposition.maps.map((map) => (
          <figure key={map.id} className="case-study-figure">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={map.src} alt={`${map.label} over the Portsmouth terrain`} width={1400} height={1080} loading="lazy" />
            <EvidenceCaption
              title={map.label}
              insight={
                map.id === "betweenModel"
                  ? "Highlights epistemic uncertainty where independently trained model checkpoints produce different central responses."
                  : "Highlights aleatoric uncertainty where latent members from the same checkpoint retain a wider range."
              }
              method="Both maps show standard deviation in meters on one shared scale. Epistemic uncertainty is estimated across checkpoint means; aleatoric uncertainty is estimated among latent members within each checkpoint. These components describe the deployed ensemble structure and do not prove model error."
            />
          </figure>
        ))}
      </div>
    </div>
  );
}
