"use client";

import { useState } from "react";
import type { CaseStudyManifest } from "./caseStudyTypes";
import { EvidenceCaption } from "./EvidenceCaption";

type Location = CaseStudyManifest["flagship"]["locations"][number];

export function LocationEvidence({ locations }: { locations: Location[] }) {
  const [selectedId, setSelectedId] = useState(locations[0]?.id ?? "");
  const selected = locations.find((location) => location.id === selectedId) ?? locations[0];
  if (!selected) return null;
  return (
    <div className="location-evidence">
      <div className="location-selector" role="tablist" aria-label="Representative locations">
        {locations.map((location) => (
          <button
            key={location.id}
            type="button"
            role="tab"
            aria-selected={selected.id === location.id}
            className={selected.id === location.id ? "active" : ""}
            onClick={() => setSelectedId(location.id)}
          >
            <strong>{location.label}</strong>
            <span>{location.interpretation}</span>
          </button>
        ))}
      </div>
      <div className="location-evidence-layout">
        <figure className="case-study-figure location-map">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={selected.mapSrc} alt={`${selected.label} selected on the Irene probability map`} width={1400} height={1080} loading="lazy" />
          <EvidenceCaption
            title={`${selected.label} / ${selected.interpretation}`}
            insight="The marker connects a regional flood pattern to a representative local forecast."
            method={`Location ${selected.cellIndex.toLocaleString()} is selected reproducibly from forecast behavior at UTM easting ${selected.coordinates.easting.toLocaleString()} m and northing ${selected.coordinates.northing.toLocaleString()} m; it is not a named asset.`}
          />
        </figure>
        <figure className="case-study-figure location-chart">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={selected.panelSrc} alt={`${selected.label} depth ensemble, exceedance probability, and arrival-time distribution`} width={1700} height={520} loading="lazy" />
          <EvidenceCaption
            title="A local forecast with uncertainty in view"
            insight="Depth range, calibrated exceedance probability, and arrival timing reveal what a central map cannot show at one location."
            method="The figure summarizes all 60 production members so the central response and forecast range remain visible together."
          />
        </figure>
      </div>
    </div>
  );
}
