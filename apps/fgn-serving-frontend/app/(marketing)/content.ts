import evidence from "./evidence.json";

export const COLLABORATION_MAILTO =
  "mailto:jrj6wm@virginia.edu?subject=FloodUQ%20research%20and%20deployment%20collaboration";

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
    title: "Test more scenarios, sooner",
    body: "Compare changes in coastal water level and rainfall without rebuilding a detailed physics simulation for every alternative."
  },
  {
    title: "Calibrated probabilities",
    body: "See the chance that water passes a chosen depth after calibration against reference simulations not used for training."
  },
  {
    title: "See why forecasts differ",
    body: "Separate disagreement between trained models from variation among plausible outcomes produced by the same model."
  },
  {
    title: "Know when conditions are unfamiliar",
    body: "Flag unusual or high-disagreement scenarios for expert review and targeted detailed simulation."
  }
] as const;

export const plainLanguageGuide: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Scenario inputs",
    body: "The coastal water level and rainfall supplied through time. Modelers often call these inputs forcing."
  },
  {
    number: "02",
    title: "Forecast group",
    body: "A collection of plausible model outcomes, often called an ensemble, used to show a range instead of one answer."
  },
  {
    number: "03",
    title: "Calibrated probability",
    body: "The chance estimated by the forecast group, then calibrated with separate reference simulations so it better reflects how often the selected depth was exceeded in those cases."
  },
  {
    number: "04",
    title: "Uncertainty",
    body: "How widely plausible forecasts differ. A narrow range means stronger agreement; a wide range means more review may be useful."
  }
] as const;

export const decisionQuestions: readonly NumberedContentItem[] = [
  { number: "01", title: "Where?", body: "Identify where water may reach, how deep it may become, and how much of the study area may be affected." },
  { number: "02", title: "When?", body: "See when water is expected to first cross a selected depth at locations across the study area." },
  { number: "03", title: "How likely?", body: "See the calibrated probability that water depth rises above a level chosen for the study." },
  { number: "04", title: "Why do forecasts differ?", body: "Separate differences between trained models from variation among plausible outcomes from the same model." }
] as const;

export const serviceOutcomes = [
  {
    src: "/marketing/portsmouth/overview/probability.webp",
    alt: "Map showing the calibrated probability that coastal water depth passes the selected level in Portsmouth",
    width: 1115,
    height: 929,
    label: "Chance and affected area",
    body: "See the calibrated chance that water passes a chosen depth and how the expected flood footprint changes through time."
  },
  {
    src: "/marketing/portsmouth/overview/interval_width.webp",
    alt: "Map showing where plausible Portsmouth flood forecasts differ most",
    width: 1115,
    height: 929,
    label: "Forecast agreement",
    body: "Find locations where plausible forecasts stay close together or spread apart, then see what causes the difference."
  },
  {
    src: "/marketing/portsmouth/overview/arrival_time.webp",
    alt: "Forecast arrival-time map from the Portsmouth deployment",
    width: 1115,
    height: 922,
    label: "Arrival timing",
    body: "See when water is expected to first cross the selected depth at locations across the study area."
  }
] as const;

export const speedPrinciples: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Generate plausible forecasts directly",
    body: "FloodUQ creates each member of a forecast group directly. Diffusion-based AI methods instead refine each member through many repeated steps."
  },
  {
    number: "02",
    title: "More alternatives in one study cycle",
    body: "Rapid forecast-group generation makes it practical to compare water-level and rainfall alternatives, sensitivity cases, and updated assumptions."
  },
  {
    number: "03",
    title: "Performance backed by evidence",
    body: "Speed and accuracy claims are tied to a named deployment, equal numbers of plausible forecasts, measured hardware, and storms not used for training."
  }
] as const;

export const calibrationNotes: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Start with the raw model probability",
    body: "The share of plausible forecasts above a chosen depth provides the raw probability before calibration."
  },
  {
    number: "02",
    title: "Calibrate each depth threshold",
    body: "Each chosen depth uses its own calibration curve, fitted with detailed simulations not used to train the forecast model and evaluated on held-out reference cases."
  },
  {
    number: "03",
    title: "Keep the raw and calibrated values",
    body: "Each result retains the scenario inputs, raw model probability, calibrated probability, calibration version, model version, and review record."
  }
] as const;

export const uncertaintySources: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Model uncertainty (epistemic)",
    body: "Independently trained models can give different central forecasts. Larger differences suggest that model choices or additional reference evidence deserve review."
  },
  {
    number: "02",
    title: "Outcome variability (aleatoric)",
    body: "One trained model can still produce a range of plausible outcomes. This range represents event variability retained within that model's forecast."
  }
] as const;

export const comparisonMethods: readonly ComparisonMethod[] = [
  { id: "physics", label: "Detailed physics model", qualifier: "High-detail reference simulation" },
  { id: "diffusion", label: "Diffusion AI model", qualifier: "Repeatedly refines each forecast" },
  { id: "rawEnsemble", label: "Uncalibrated AI forecast group", qualifier: "Combines several model outputs" },
  { id: "flooduq", label: "FloodUQ", qualifier: "Managed probability and uncertainty service" }
] as const;

export const comparisonRows: readonly ComparisonRow[] = [
  {
    dimension: "Best role",
    physics: "Detailed simulation and trusted reference-data generation.",
    diffusion: "AI-based forecast generation when repeated refinement time is acceptable.",
    rawEnsemble: "Rapid estimates of forecast range from several model outputs.",
    flooduq: "Rapid scenario studies with calibrated probabilities, separated uncertainty sources, and unfamiliar-condition monitoring."
  },
  {
    dimension: "Creating a range of forecasts",
    physics: "Repeated detailed simulations can make large forecast groups costly.",
    diffusion: "Each plausible forecast is refined through many repeated steps.",
    rawEnsemble: "Usually creates each model output directly.",
    flooduq: "Creates plausible spatial forecasts directly using several trained models and members per model."
  },
  {
    dimension: "Probability calibration",
    physics: "Provides the reference outcomes needed to fit or evaluate calibration.",
    diffusion: "Requires a separate calibration workflow for forecast probabilities.",
    rawEnsemble: "Raw model probabilities may not match how often depth thresholds are exceeded in separate test cases.",
    flooduq: "Calibrates each selected depth probability against held-out reference simulations."
  },
  {
    dimension: "Explaining why forecasts differ",
    physics: "Depends on how input and parameter alternatives are designed.",
    diffusion: "Shows a range, but does not automatically separate its sources.",
    rawEnsemble: "A combined range can hide why model outputs differ.",
    flooduq: "Reports model uncertainty (epistemic) separately from outcome variability (aleatoric)."
  },
  {
    dimension: "Recognizing unfamiliar conditions",
    physics: "Applicability is governed through the physics setup and assumptions.",
    diffusion: "Checks for unfamiliar conditions must be added separately.",
    rawEnsemble: "Checks for unfamiliar conditions are usually separate from prediction.",
    flooduq: "Checks scenario inputs before a run, reviews forecast behavior afterward, and preserves priority cases for experts."
  }
] as const;

export const monitoringLoop: readonly NumberedContentItem[] = [
  { number: "01", title: "Screen", body: "Check whether new water-level and rainfall inputs resemble the evidence used to build the deployed model." },
  { number: "02", title: "Detect", body: "Identify unusual inputs, forecasts that disagree strongly, and large calibration shifts that warrant attention." },
  { number: "03", title: "Review", body: "Route strong signals into an expert-reviewed queue so attention stays focused on the scenarios with the greatest evidence gap." },
  { number: "04", title: "Simulate", body: "Prioritize selected scenarios for HEC-RAS, a detailed hydraulic simulation used as reference evidence, and directly measure error." },
  { number: "05", title: "Stage an update", body: "Turn reviewed evidence into a controlled, versioned candidate for future model or calibration updates." }
] as const;

export const deploymentSteps: readonly NumberedContentItem[] = [
  { number: "01", title: "Define the decision need", body: "Align the coastline, water-level and rainfall scenarios, selected depths, time horizon, and decisions the service must support." },
  { number: "02", title: "Build the reference evidence", body: "Assemble terrain, historical or designed events, and aligned high-fidelity simulations for training and evaluation." },
  { number: "03", title: "Train the coastline model", body: "Train the fast spatial forecast models and their starting-water conditions for the target coastline." },
  { number: "04", title: "Calibrate probabilities and validate skill", body: "Test performance on events excluded from training, fit selected-depth calibration curves, and document where the model has been evaluated." },
  { number: "05", title: "Launch the service", body: "Deliver a secure console or API for repeatable scenario runs, comparison, reporting, and review." },
  { number: "06", title: "Monitor the evidence gap", body: "Track unfamiliar and high-disagreement scenarios so new high-fidelity simulations are directed where they add the most value." }
] as const;

export const useCases: readonly NumberedContentItem[] = [
  {
    number: "01",
    title: "Coastal resilience studies",
    body: "Compare adaptation and planning scenarios through expected depth, calibrated probability, timing, affected area, and forecast-agreement products."
  },
  {
    number: "02",
    title: "Infrastructure scenario review",
    body: "Inspect local flood trajectories around ports, transportation corridors, utilities, and other exposed assets."
  },
  {
    number: "03",
    title: "Engineering and risk partnerships",
    body: "Extend established coastal modeling programs with rapid probability-based scenarios, visible forecast agreement, and traceable evidence."
  }
] as const;

export const portsmouthCaseStudy = {
  label: "Portsmouth, Virginia deployment proof",
  title: "From historical water levels and rainfall to calibrated flood probabilities.",
  intro: "Portsmouth demonstrates how FloodUQ turns coastal water-level and rainfall inputs into expected depth, calibrated probability, timing, affected-area, and forecast-agreement products. The evidence is local; the deployment process is designed to be repeated with each coastline's own terrain and reference simulations.",
  deploymentStats: [
    { value: "3 x 20", label: "trained models and plausible outcomes per model" },
    { value: "5,904", label: "coastal mesh cells" },
    { value: "94", label: "quarter-hour forecast steps" },
    { value: "14.2 min", label: "full RTX 4090 workflow with artifacts" }
  ],
  runtimeBars: [
    { label: "FloodUQ forecast generation", value: 7.7, display: "7.7 min", tone: "blue" },
    { label: "Scrubbable map frames", value: 4.4, display: "4.4 min", tone: "cyan" },
    { label: "Maps and animations", value: 1.9, display: "1.9 min", tone: "green" },
    { label: "Probability calibration and summaries", value: 0.2, display: "< 0.2 min", tone: "amber" }
  ],
  fullWorkflowNote: "Measured July 14, 2026 for one run with 60 plausible forecasts and 94 time steps on the lab RTX 4090, from validated water-level and rainfall inputs through calibrated probabilities, maps, animations, and location inspection. Runtime varies with hardware, queue state, and requested products.",
  historicalNote: "Historical test cases not used for training include Ophelia 2023, Isabel 2003, and Irene 2011. Maps show the chance of passing a selected depth and the width of the forecast range; time plots include HEC-RAS detailed-simulation reference behavior."
} as const;

export const calibrationCurve = [
  [0.025, 0.000015], [0.075, 0.002536], [0.125, 0.012229], [0.175, 0.031054],
  [0.225, 0.055722], [0.275, 0.098521], [0.325, 0.174692], [0.375, 0.231909],
  [0.425, 0.299162], [0.475, 0.396797], [0.525, 0.487939], [0.575, 0.649683],
  [0.625, 0.70145], [0.675, 0.792875], [0.725, 0.870756], [0.775, 0.903805],
  [0.825, 0.950184], [0.875, 0.97902], [0.925, 0.99414], [0.975, 0.99989]
] as const;
