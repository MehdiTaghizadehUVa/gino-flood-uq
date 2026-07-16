import evidence from "./evidence.json";

export const RESEARCH_DISCLAIMER =
  "Research only; not for emergency or operational decision use.";

export const PILOT_MAILTO =
  "mailto:jrj6wm@virginia.edu?subject=FloodUQ%20managed%20deployment%20pilot";

export type NumberedContentItem = {
  number: string;
  title: string;
  body: string;
};

export type ComparisonMethod = {
  id: "physics" | "diffusion" | "rawEnsemble" | "flooduq";
  label: string;
  qualifier: string;
};

export type ComparisonRow = {
  dimension: string;
  physics: string;
  diffusion: string;
  rawEnsemble: string;
  flooduq: string;
};

export type EvidenceClaim = {
  id: string;
  value: string;
  label: string;
  scope: string;
  sample: string;
  hardware: string;
  measurement: string;
  sourceArtifact: string;
  measured: true;
};

export type EvidenceRegistry = {
  caseStudy: string;
  benchmark: {
    label: string;
    sample: string;
    ensembleBudget: string;
    hardware: string;
    timingScope: string;
    skillMetrics: string;
    figure: string;
  };
  claims: EvidenceClaim[];
};

export const evidenceRegistry = evidence as EvidenceRegistry;

export const marketingNav = [
  { href: "#capabilities", label: "Capabilities" },
  { href: "#comparison", label: "Why FloodUQ" },
  { href: "#evidence", label: "Evidence" },
  { href: "#deployment", label: "Deployment" }
] as const;

export const servicePillars = [
  {
    title: "Rapid scenario iteration",
    body: "Explore changing coastal-stage and precipitation assumptions without rebuilding a physics ensemble for every question."
  },
  {
    title: "Calibrated probabilities",
    body: "Turn raw ensemble frequencies into threshold-specific probabilities evaluated against held-out reference simulations."
  },
  {
    title: "Uncertainty source separation",
    body: "Separate epistemic uncertainty from aleatoric uncertainty instead of collapsing both into one spread map."
  },
  {
    title: "Novelty monitoring",
    body: "Identify scenarios that are unfamiliar, disagreement-heavy, or suitable for additional high-fidelity evidence."
  }
] as const;

export const decisionQuestions: readonly NumberedContentItem[] = [
  { number: "01", title: "Where?", body: "Map expected depth, exceedance footprints, and affected-area extent across the deployed domain." },
  { number: "02", title: "When?", body: "Inspect arrival, duration, peak timing, and the forcing-response trajectory on one lead-time axis." },
  { number: "03", title: "How likely?", body: "Evaluate calibrated probabilities for the depth thresholds that matter to the study." },
  { number: "04", title: "Why uncertain?", body: "Separate epistemic uncertainty from aleatoric uncertainty within the nested forecast ensemble." },
  { number: "05", title: "Is it familiar?", body: "Screen the forcing and completed forecast against monitored reference behavior." }
] as const;

export const serviceOutcomes = [
  {
    src: "/marketing/exceedance-probability.webp",
    alt: "Calibrated coastal water-depth exceedance probability map from the Portsmouth deployment",
    width: 1400,
    height: 1180,
    label: "Probability and extent",
    body: "See where a study threshold may be exceeded and how the expected affected area evolves through time."
  },
  {
    src: "/marketing/uncertainty-width.webp",
    alt: "Spatial uncertainty-width map from the Portsmouth deployment",
    width: 1400,
    height: 1176,
    label: "Confidence and disagreement",
    body: "Locate where forecast members agree, where spread concentrates, and which uncertainty source dominates."
  },
  {
    src: "/marketing/arrival-time.webp",
    alt: "Forecast arrival-time map from the Portsmouth deployment",
    width: 1400,
    height: 1180,
    label: "Timing and persistence",
    body: "Translate spatial forecasts into arrival, peak, and wet-duration products for scenario review."
  }
] as const;

export const speedPrinciples: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Direct ensemble inference",
    body: "FGNO produces spatial forecast members through direct neural-operator inference rather than an iterative denoising chain for every member."
  },
  {
    number: "02",
    title: "More scenarios per study",
    body: "Faster ensemble generation makes it practical to compare forcing alternatives, sensitivity cases, and updated assumptions."
  },
  {
    number: "03",
    title: "Measured, scoped evidence",
    body: "Every performance statement is tied to a named deployment, ensemble budget, hardware environment, and held-out benchmark."
  }
] as const;

export const calibrationNotes: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Raw evidence remains visible",
    body: "Calibration does not erase the original ensemble signal; raw and corrected products remain available together."
  },
  {
    number: "02",
    title: "Threshold-specific correction",
    body: "Each exceedance threshold uses its own held-out calibration mapping rather than one global adjustment."
  },
  {
    number: "03",
    title: "Versioned and reviewable",
    body: "The deployed model, calibration bundle, forcing fingerprint, seed policy, and result artifacts remain connected."
  }
] as const;

export const uncertaintySources: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Epistemic uncertainty",
    body: "Variation across independently trained model means represents uncertainty in learned structure and parameters, helping prioritize model review or new reference evidence."
  },
  {
    number: "02",
    title: "Aleatoric uncertainty",
    body: "Variation among latent forecast members from the same model represents the retained event variability within that learned forecast representation."
  }
] as const;

export const comparisonMethods: readonly ComparisonMethod[] = [
  { id: "physics", label: "High-fidelity physics", qualifier: "Reference authority" },
  { id: "diffusion", label: "Diffusion surrogate", qualifier: "Iterative probabilistic sampling" },
  { id: "rawEnsemble", label: "Raw ML ensemble", qualifier: "Fast member aggregation" },
  { id: "flooduq", label: "FloodUQ", qualifier: "Managed calibrated UQ service" }
] as const;

export const comparisonRows: readonly ComparisonRow[] = [
  {
    dimension: "Best role",
    physics: "High-fidelity simulation and reference-data generation.",
    diffusion: "Stochastic surrogate generation where iterative sampling cost is acceptable.",
    rawEnsemble: "Rapid spread estimates from multiple model outputs.",
    flooduq: "Rapid scenario studies with calibrated, decomposed, and monitored uncertainty."
  },
  {
    dimension: "Ensemble generation",
    physics: "Repeated numerical simulations can make large ensembles costly.",
    diffusion: "Each member requires an iterative denoising trajectory.",
    rawEnsemble: "Direct inference, depending on ensemble construction.",
    flooduq: "Direct FGNO inference across nested model and member structure."
  },
  {
    dimension: "Probability calibration",
    physics: "Not inherently a calibrated surrogate probability product.",
    diffusion: "Requires a separate calibration design and evaluation.",
    rawEnsemble: "Raw frequency may not match held-out event frequency.",
    flooduq: "Threshold-specific mappings evaluated on held-out reference simulations."
  },
  {
    dimension: "Uncertainty interpretation",
    physics: "Depends on the forcing and parameter ensemble design.",
    diffusion: "Generated spread is available, but source separation is not automatic.",
    rawEnsemble: "Combined spread can obscure why models disagree.",
    flooduq: "Epistemic and aleatoric uncertainty components are reported separately."
  },
  {
    dimension: "Model familiarity",
    physics: "Applicability is governed through the physics setup and assumptions.",
    diffusion: "Monitoring must be added around the surrogate.",
    rawEnsemble: "Monitoring is usually external to the prediction workflow.",
    flooduq: "Pre-run screening, post-run diagnostics, and candidate capture are integrated."
  }
] as const;

export const monitoringLoop: readonly NumberedContentItem[] = [
  { number: "01", title: "Screen", body: "Compare uploaded forcing against the deployed domain's monitored reference behavior before GPU work." },
  { number: "02", title: "Detect", body: "Evaluate novelty, affected-area uncertainty, calibration shift, and model disagreement after inference." },
  { number: "03", title: "Review", body: "Place strong signals into an expert-reviewed evidence queue instead of silently treating every forecast as equally familiar." },
  { number: "04", title: "Simulate", body: "Select high-value candidates for optional HEC-RAS simulation and error analysis." },
  { number: "05", title: "Stage an update", body: "Use reviewed evidence to prepare future calibration or model updates; activation remains versioned and operator-controlled." }
] as const;

export const deploymentSteps: readonly NumberedContentItem[] = [
  { number: "01", title: "Define the domain", body: "Agree on terrain, mesh, boundary forcings, study thresholds, forecast horizon, and intended research use." },
  { number: "02", title: "Prepare reference evidence", body: "Assemble historical or designed events and aligned high-fidelity simulations for training and evaluation." },
  { number: "03", title: "Train the ensemble", body: "Fit the domain-specific neural-operator ensemble and a forcing-conditioned initial-state library." },
  { number: "04", title: "Calibrate and validate", body: "Evaluate held-out skill, fit threshold-specific calibration, and document deployment limits." },
  { number: "05", title: "Deploy the service", body: "Provide a gated console or API with reproducible runs, artifacts, provenance, and GPU queue controls." },
  { number: "06", title: "Monitor and review", body: "Track unfamiliar scenarios and high-disagreement outputs for future high-fidelity evidence collection." }
] as const;

export const useCases: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Coastal resilience studies",
    body: "Compare planning scenarios through calibrated depth, probability, timing, extent, and uncertainty products."
  },
  {
    number: "02",
    title: "Infrastructure scenario review",
    body: "Inspect site- and cell-level trajectories around ports, transportation corridors, utilities, and other exposed assets."
  },
  {
    number: "03",
    title: "Engineering and risk partnerships",
    body: "Add fast, calibrated surrogate ensembles and model-governance evidence to established coastal modeling workflows."
  }
] as const;

export const portsmouthCaseStudy = {
  label: "Portsmouth, Virginia deployment proof",
  title: "From historical forcing to calibrated flood probabilities.",
  intro: "A named historical event shows how a domain-specific FloodUQ deployment turns coastal forcing into probability, timing, extent, and uncertainty evidence. Portsmouth is the proof case, while the managed workflow is designed to be repeated for other coastal domains.",
  deploymentStats: [
    { value: "3 x 20", label: "model and latent member structure" },
    { value: "5,904", label: "coastal mesh cells" },
    { value: "94", label: "quarter-hour forecast steps" },
    { value: "14.2 min", label: "full RTX 4090 workflow with artifacts" }
  ],
  runtimeBars: [
    { label: "FGNO rollout", value: 7.7, display: "7.7 min", tone: "blue" },
    { label: "Scrubbable map frames", value: 4.4, display: "4.4 min", tone: "cyan" },
    { label: "Maps and animations", value: 1.9, display: "1.9 min", tone: "green" },
    { label: "Calibration and summaries", value: 0.2, display: "< 0.2 min", tone: "amber" }
  ],
  fullWorkflowNote: "Measured July 14, 2026 for one 60-member, 94-step run on the lab RTX 4090. Includes HDF5, summaries, maps, animations, and scrubbable frames; runtime varies with hardware, queue state, and requested products.",
  historicalNote: "Held-out historical cases include Ophelia 2023, Isabel 2003, and Irene 2011. Maps show exceedance probability and interval width; trajectories include HEC-RAS reference behavior."
} as const;

export const calibrationCurve = [
  [0.025, 0.000015], [0.075, 0.002536], [0.125, 0.012229], [0.175, 0.031054],
  [0.225, 0.055722], [0.275, 0.098521], [0.325, 0.174692], [0.375, 0.231909],
  [0.425, 0.299162], [0.475, 0.396797], [0.525, 0.487939], [0.575, 0.649683],
  [0.625, 0.70145], [0.675, 0.792875], [0.725, 0.870756], [0.775, 0.903805],
  [0.825, 0.950184], [0.875, 0.97902], [0.925, 0.99414], [0.975, 0.99989]
] as const;
