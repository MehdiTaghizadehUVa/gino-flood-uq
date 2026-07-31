"use client";

import { useState } from "react";
import { caseStudyAsset } from "../caseStudyAsset";
import { EvidenceCaption } from "./EvidenceCaption";

function displayLabel(productId: string, fallback: string) {
  if (productId === "probability") return "Chance depth passes 0.30 m";
  if (productId === "meanDepth") return "Expected water depth";
  if (productId === "intervalWidth") return "Width of the 90% forecast range";
  return fallback;
}

export function DecisionSnapshot({ products }: { products: { id: string; label: string; src: string }[] }) {
  const [selected, setSelected] = useState(products[0]?.id ?? "");
  return (
    <div className="decision-snapshot">
      <div className="case-study-segmented snapshot-mobile-tabs" aria-label="Decision snapshot product">
        {products.map((product) => (
          <button key={product.id} type="button" className={selected === product.id ? "active" : ""} onClick={() => setSelected(product.id)}>
            {displayLabel(product.id, product.label)}
          </button>
        ))}
      </div>
      <div className="decision-snapshot-grid">
        {products.map((product) => (
          <figure key={product.id} className={`case-study-figure ${selected === product.id ? "selected" : ""}`}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={caseStudyAsset(product.src)} alt={`${displayLabel(product.id, product.label)} at the selected Portsmouth lead time`} width={1400} height={1080} loading="lazy" />
            <EvidenceCaption
              title={displayLabel(product.id, product.label)}
              insight={
                product.id === "probability"
                  ? "Shows the share of plausible forecasts that pass the selected depth at each location."
                  : product.id === "meanDepth"
                    ? "Shows the average water depth across the full group of plausible forecasts."
                    : "Shows where plausible depths stay close together or remain widely separated."
              }
              method="All three maps show the same event and moment. Because they answer different questions and use different units, interpret their patterns together rather than comparing colors directly."
            />
          </figure>
        ))}
      </div>
    </div>
  );
}
