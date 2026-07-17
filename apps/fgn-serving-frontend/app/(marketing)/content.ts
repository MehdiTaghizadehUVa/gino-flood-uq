import evidence from "./evidence.json";

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
    title: "Faster scenario exploration",
    body: "Compare coastal-stage and precipitation scenarios without running a new high-fidelity ensemble for every alternative."
  },
  {
    title: "Calibrated probabilities",
    body: "Evaluate depth-threshold probabilities that are calibrated and tested against held-out reference simulations."
  },
  {
    title: "Diagnosable uncertainty",
    body: "See whether uncertainty is driven by model disagreement or variability among plausible outcomes."
  },
  {
    title: "Evidence-aware monitoring",
    body: "Flag unfamiliar or high-disagreement scenarios for expert review and targeted high-fidelity simulation."
  }
] as const;

export const decisionQuestions: readonly NumberedContentItem[] = [
  { number: "01", title: "Where?", body: "Identify where water may reach, how deep it may become, and how much of the study area may be affected." },
  { number: "02", title: "When?", body: "Track onset, peak timing, recession, and duration across the forecast horizon." },
  { number: "03", title: "How likely?", body: "Evaluate calibrated probabilities for the depth thresholds that matter to the study." },
  { number: "04", title: "Why uncertain?", body: "Distinguish model disagreement from variability among plausible forecast outcomes." },
  { number: "05", title: "When is more evidence needed?", body: "Identify scenarios that fall outside familiar forcing or forecast behavior before they are treated as routine." }
] as const;

export const serviceOutcomes = [
  {
    src: "/marketing/portsmouth/overview/probability.webp",
    alt: "Calibrated coastal water-depth exceedance probability map from the Portsmouth deployment",
    width: 1115,
    height: 929,
    label: "Probability and extent",
    body: "See where a study threshold may be exceeded and how the expected flood footprint changes through time."
  },
  {
    src: "/marketing/portsmouth/overview/interval_width.webp",
    alt: "Spatial uncertainty-width map from the Portsmouth deployment",
    width: 1115,
    height: 929,
    label: "Uncertainty and disagreement",
    body: "Find locations where forecasts diverge and determine which source of uncertainty deserves attention."
  },
  {
    src: "/marketing/portsmouth/overview/arrival_time.webp",
    alt: "Forecast arrival-time map from the Portsmouth deployment",
    width: 1115,
    height: 922,
    label: "Timing and persistence",
    body: "See when flooding begins, reaches its peak, and persists at locations across the study area."
  }
] as const;

export const speedPrinciples: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Direct ensemble inference",
    body: "FGNO generates spatial forecast members directly, avoiding the iterative sampling required by diffusion-based surrogates."
  },
  {
    number: "02",
    title: "More alternatives in one study cycle",
    body: "Rapid ensemble generation makes it practical to compare forcing alternatives, sensitivity cases, and updated assumptions."
  },
  {
    number: "03",
    title: "Performance backed by evidence",
    body: "Speed and skill claims are tied to a named deployment, equal ensemble budgets, measured hardware, and held-out events."
  }
] as const;

export const calibrationNotes: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Compare before and after calibration",
    body: "Review raw ensemble frequencies beside calibrated probabilities to see exactly how the evidence changes."
  },
  {
    number: "02",
    title: "Calibrated for the threshold in question",
    body: "Each depth threshold uses its own mapping fitted and evaluated on held-out reference simulations."
  },
  {
    number: "03",
    title: "Traceable from input to result",
    body: "Each result retains the forcing, model version, calibration version, ensemble settings, and monitoring record used to create it."
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
  { number: "01", title: "Screen", body: "Assess whether a new forcing scenario resembles the evidence used to build the deployed model." },
  { number: "02", title: "Detect", body: "Identify unusual inputs, elevated forecast disagreement, and calibration behavior that warrants attention." },
  { number: "03", title: "Review", body: "Route strong signals into an expert-reviewed queue so attention stays focused on the scenarios with the greatest evidence gap." },
  { number: "04", title: "Simulate", body: "Prioritize selected scenarios for HEC-RAS simulation and direct error analysis." },
  { number: "05", title: "Stage an update", body: "Turn reviewed evidence into a controlled, versioned candidate for future model or calibration updates." }
] as const;

export const deploymentSteps: readonly NumberedContentItem[] = [
  { number: "01", title: "Define the decision need", body: "Align the coastal domain, forcing scenarios, depth thresholds, forecast horizon, and decisions the service must support." },
  { number: "02", title: "Build the reference evidence", body: "Assemble terrain, historical or designed events, and aligned high-fidelity simulations for training and evaluation." },
  { number: "03", title: "Train the domain model", body: "Fit the neural-operator ensemble and forcing-conditioned initial-state library to the target coastline." },
  { number: "04", title: "Calibrate and validate", body: "Measure held-out performance, calibrate threshold probabilities, and document the validated operating range." },
  { number: "05", title: "Launch the service", body: "Deliver a secure console or API for repeatable scenario runs, comparison, reporting, and review." },
  { number: "06", title: "Monitor the evidence gap", body: "Track unfamiliar and high-disagreement scenarios so new high-fidelity simulations are directed where they add the most value." }
] as const;

export const useCases: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Coastal resilience studies",
    body: "Compare adaptation and planning scenarios through calibrated depth, probability, timing, extent, and uncertainty products."
  },
  {
    number: "02",
    title: "Infrastructure scenario review",
    body: "Inspect local flood trajectories around ports, transportation corridors, utilities, and other exposed assets."
  },
  {
    number: "03",
    title: "Engineering and risk partnerships",
    body: "Extend established coastal modeling programs with rapid probabilistic scenarios, calibrated uncertainty, and traceable evidence."
  }
] as const;

export const portsmouthCaseStudy = {
  label: "Portsmouth, Virginia deployment proof",
  title: "From historical forcing to calibrated flood probabilities.",
  intro: "Portsmouth demonstrates how FloodUQ turns coastal forcing into calibrated probability, timing, extent, and uncertainty products. The evidence is local; the deployment process is designed to be repeated with each coastline's own terrain and reference simulations.",
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
  fullWorkflowNote: "Measured July 14, 2026 for one 60-member, 94-step run on the lab RTX 4090, from validated forcing through calibrated summaries, maps, animations, and location inspection. Runtime varies with hardware, queue state, and requested products.",
  historicalNote: "Held-out historical cases include Ophelia 2023, Isabel 2003, and Irene 2011. Maps show exceedance probability and interval width; trajectories include HEC-RAS reference behavior."
} as const;

export const calibrationCurve = [
  [0.025, 0.000015], [0.075, 0.002536], [0.125, 0.012229], [0.175, 0.031054],
  [0.225, 0.055722], [0.275, 0.098521], [0.325, 0.174692], [0.375, 0.231909],
  [0.425, 0.299162], [0.475, 0.396797], [0.525, 0.487939], [0.575, 0.649683],
  [0.625, 0.70145], [0.675, 0.792875], [0.725, 0.870756], [0.775, 0.903805],
  [0.825, 0.950184], [0.875, 0.97902], [0.925, 0.99414], [0.975, 0.99989]
] as const;
