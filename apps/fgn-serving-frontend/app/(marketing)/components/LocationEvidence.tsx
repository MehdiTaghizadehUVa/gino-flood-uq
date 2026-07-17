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
            insight={`The marker identifies computational cell ${selected.cellIndex.toLocaleString()} at a representative wettable-domain response.`}
            method={`Coordinates are UTM easting ${selected.coordinates.easting.toLocaleString()} m and northing ${selected.coordinates.northing.toLocaleString()} m. Locations are selected deterministically from forecast behavior and are not named assets.`}
          />
        </figure>
        <figure className="case-study-figure location-chart">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={selected.panelSrc} alt={`${selected.label} depth ensemble, exceedance probability, and arrival-time distribution`} width={1700} height={520} loading="lazy" />
          <EvidenceCaption
            title="One cell, the full ensemble story"
            insight="The depth fan, calibrated exceedance trace, and arrival distribution provide context that a central map alone cannot convey."
            method="The chart is rendered from the 60 production members. Member traces are presented as a static evidence figure; the public page does not ship HDF5 files or numerical member arrays."
          />
        </figure>
      </div>
    </div>
  );
}
