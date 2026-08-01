"use client";

import { useState } from "react";
import { caseStudyAsset } from "../caseStudyAsset";
import type { CaseStudyManifest } from "./caseStudyTypes";
import { EvidenceCaption } from "./EvidenceCaption";

type Location = CaseStudyManifest["flagship"]["locations"][number];

function displayInterpretation(location: Location) {
  if (location.interpretation === "Early onset") return "Earlier arrival";
  if (location.interpretation === "Elevated member spread") return "Wider forecast range";
  if (location.interpretation === "Later response") return "Later arrival";
  return location.interpretation;
}

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
            <span>{displayInterpretation(location)}</span>
          </button>
        ))}
      </div>
      <div className="location-evidence-layout">
        <figure className="case-study-figure location-map">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={caseStudyAsset(selected.mapSrc)} alt={`${selected.label} selected on the Irene probability map`} width={1400} height={1080} loading="lazy" />
          <EvidenceCaption
            title={`${selected.label} / ${displayInterpretation(selected)}`}
            insight="The marker connects a regional flood pattern to a representative local forecast."
            method={`Location ${selected.cellIndex.toLocaleString()} is selected reproducibly from forecast behavior at UTM easting ${selected.coordinates.easting.toLocaleString()} m and northing ${selected.coordinates.northing.toLocaleString()} m; it is not a named asset.`}
          />
        </figure>
        <figure className="case-study-figure location-chart">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={caseStudyAsset(selected.panelSrc)} alt={`${selected.label} possible depths, chance of passing 0.30 meters, and arrival-time range`} width={1700} height={520} loading="lazy" />
          <EvidenceCaption
            title="Possible depths, chance of passing 0.30 m, and arrival timing"
            insight="The depth range, calibrated probability, and arrival-time distribution show what an average map cannot reveal at one location."
            method="The figure summarizes all 60 production members so the central response and forecast range remain visible together."
          />
        </figure>
      </div>
    </div>
  );
}
