export type CaseStudyFrame = {
  timeIndex: number;
  leadHours: number;
  src: string;
};

export type CaseStudyProduct = {
  id: "probability" | "meanDepth" | "intervalWidth";
  label: string;
  displayFloor: number;
  vmin: number;
  vmax: number;
  animation: {
    mp4Src: string;
    posterSrc: string;
    frameCount: number;
    sourceFrameRate: number;
    playbackFrameRate: number;
    durationSeconds: number;
    interpolation: string;
  };
  frames: CaseStudyFrame[];
};

export type CaseStudyManifest = {
  schemaVersion: number;
  caseStudyId: string;
  generatedAt: string;
  title: string;
  eyebrow: string;
  intro: string;
  flagship: {
    eventId: string;
    label: string;
    thresholdM: number;
    metrics: { value: string; label: string }[];
    peakAreaTimeIndex: number;
    peakDisagreementTimeIndex: number;
    peakMeanDepthTimeIndex: number;
    posterSrc: string;
    hero: {
      src: string;
      posterSrc: string;
      mp4Src: string;
      webmSrc: string;
      sequenceSrc: string;
      frameCount: number;
      frameRate: number;
      durationSeconds: number;
      product: string;
      leadHours: number;
      displayFloorM: number;
      selection: string;
    };
    products: CaseStudyProduct[];
    snapshot: { id: string; label: string; src: string }[];
    locations: {
      id: string;
      label: string;
      interpretation: string;
      cellIndex: number;
      coordinates: { easting: number; northing: number };
      mapSrc: string;
      panelSrc: string;
    }[];
    decomposition: {
      leadHours: number;
      betweenVarianceShare: number;
      displayFloorM: number;
      sharedVmaxM: number;
      maps: { id: string; label: string; src: string }[];
    };
  };
  historicalValidation: {
    thresholdM: number;
    note: string;
    events: {
      eventId: string;
      label: string;
      thresholdM: number;
      probabilitySrc: string;
      intervalWidthSrc: string;
      trajectorySrc: string;
      runId: string;
    }[];
  };
  performance: {
    workflow: { hardware: string; scope: string; event: string };
    comparison: {
      sample: string;
      ensembleBudget: string;
      hardware: string;
      timingScope: string;
      fairCrpsUnit: string;
      brierThresholdM: number;
      sourceArtifact: string;
    };
  };
  provenance: Record<string, string | number>;
  displayPolicy: Record<string, string | number | number[]>;
  researchDisclaimer: string;
};
