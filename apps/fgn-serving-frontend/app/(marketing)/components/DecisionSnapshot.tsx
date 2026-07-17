"use client";

import { useState } from "react";
import { caseStudyAsset } from "../caseStudyAsset";
import { EvidenceCaption } from "./EvidenceCaption";

export function DecisionSnapshot({ products }: { products: { id: string; label: string; src: string }[] }) {
  const [selected, setSelected] = useState(products[0]?.id ?? "");
  return (
    <div className="decision-snapshot">
      <div className="case-study-segmented snapshot-mobile-tabs" aria-label="Decision snapshot product">
        {products.map((product) => (
          <button key={product.id} type="button" className={selected === product.id ? "active" : ""} onClick={() => setSelected(product.id)}>
            {product.label}
          </button>
        ))}
      </div>
      <div className="decision-snapshot-grid">
        {products.map((product) => (
          <figure key={product.id} className={`case-study-figure ${selected === product.id ? "selected" : ""}`}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={caseStudyAsset(product.src)} alt={`${product.label} at the peak expected 0.30 meter footprint`} width={1400} height={1080} loading="lazy" />
            <EvidenceCaption
              title={product.label}
              insight={
                product.id === "probability"
                  ? "Shows where the ensemble assigns meaningful probability to the selected study threshold."
                  : product.id === "meanDepth"
                    ? "Shows the central depth response at the same lead time as the probability view."
                    : "Shows where plausible members retain a wider range at the same lead time."
              }
              method="All three maps use the same event and lead time but have different units and display scales; they should be interpreted together, not numerically compared color-for-color."
            />
          </figure>
        ))}
      </div>
    </div>
  );
}
