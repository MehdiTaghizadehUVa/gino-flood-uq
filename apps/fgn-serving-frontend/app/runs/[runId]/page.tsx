"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { ArrowLeft, Pin, Trash2, XCircle } from "lucide-react";
import { AppShell } from "../../components/AppShell";
import { ArtifactDrawer } from "../../components/ArtifactDrawer";
import { RunProgress } from "../../components/RunProgress";
import { CellInspector } from "../../components/run-detail/CellInspector";
import { RunHeader } from "../../components/run-detail/RunHeader";
import {
  CompareTab,
  DriversTab,
  HazardTab,
  RunDetailTabs,
  UncertaintyTab
} from "../../components/run-detail/RunTabs";
import { TimePlayer } from "../../components/run-detail/TimePlayer";

// A run created in the SPA is visible to /api/runs/{id} only after the
// orchestrator finishes its initial repository write. The first one or two
// detail polls can race that write and 404. Surface the error only after the
// fetch has consistently failed so users don't see a flash of red on every
// freshly submitted run.
const TRANSIENT_404_TOLERATED = 3;
const DEFAULT_FORCING_DT_SECONDS = 900;

type RunTiming = {
  started_at?: string | null;
  completed_at?: string | null;
  elapsed_seconds?: number | null;
  runtime_seconds?: number | null;
  estimated_total_seconds?: number | null;
  estimated_remaining_seconds?: number | null;
  estimated_basis?: string | null;
  average_full_rollout_seconds?: number | null;
  average_full_rollout_sample_size?: number;
  average_full_rollout_basis?: string;
  current_run_work_fraction_of_full_rollout?: number | null;
};

type RunData = {
  run_id: string;
  label?: string | null;
  status: string;
  progress?: number;
  progress_label?: string | null;
  spec: Record<string, unknown>;
  failure_reason: string | null;
  pinned: boolean;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  runtime_seconds?: number | null;
  timing?: RunTiming;
  cache?: {
    mode?: string;
    materialized_from_cache?: boolean;
    waiting_for_cached_result?: boolean;
    cache_key_prefix?: string | null;
  };
};

type CurrentUser = {
  email: string;
  is_admin: boolean;
  disclaimer_acknowledged: boolean;
};

type Artifact = {
  artifact_id: string;
  content_type: string;
  size_bytes: number;
};

type MonitoringFlag = {
  code: string;
  message: string;
  severity?: string;
  descriptor?: string | null;
  value?: number | null;
  details?: Record<string, unknown> | null;
};

type MonitoringReport = {
  phase: "PRE_RUN" | "POST_RUN";
  monitoring_bundle_id: string;
  scores?: Record<string, number | null>;
  flags?: MonitoringFlag[];
  candidate_recommended?: boolean;
  candidate_reason?: string | null;
  created_at?: string;
};

type MonitoringPayload = {
  available: boolean;
  monitoring_bundle_id?: string | null;
  pre_run?: MonitoringReport | null;
  post_run?: MonitoringReport | null;
  candidate?: {
    candidate_id: string;
    candidate_score: number;
    reason: string;
    status: string;
    updated_at?: string;
  } | null;
};

type InitialConditionSelection = {
  mode?: string;
  library_id?: string | null;
  reference_scope?: string | null;
  selected_reference_ids?: string[];
  weights?: number[];
  distances?: number[];
  low_confidence?: boolean;
  confidence_label?: string;
};

type Summary = {
  label: string;
  n_members: number;
  n_time: number;
  n_cells: number;
  lead_time_hours: number[];
  mean_wd_overall_m: number;
  max_mean_wd_m: number;
  mean_spread_wd_m: number;
  mean_q05_wd_m: number;
  mean_q50_wd_m: number;
  mean_q95_wd_m: number;
  peak_mean_wd_by_time_m: number[];
  "inundated_cells_by_time_gt_0.05m": number[];
  max_inundated_cells_gt_0_05m?: number;
  "max_inundated_cells_gt_0.05m"?: number;
  mean_arrival_time_hours_gt_0_05m?: number | null;
  "mean_arrival_time_hours_gt_0.05m"?: number | null;
  wettable_area_km2?: number;
  "peak_expected_flooded_area_km2_gt_0.05m"?: number;
  "peak_expected_flooded_area_fraction_wettable_gt_0.05m"?: number;
  "peak_expected_flooded_area_lead_hours_gt_0.05m"?: number | null;
  "onset_lead_hours_expected_flooded_area_fraction_gt_1pct_gt_0.05m"?: number | null;
  peak_area_weighted_iqr_wd_m?: number;
  peak_area_weighted_central_90_wd_m?: number;
  peak_area_weighted_between_checkpoint_spread_wd_m?: number | null;
  peak_area_weighted_between_checkpoint_variance_share?: number | null;
  peak_high_checkpoint_disagreement_area_fraction_wettable?: number | null;
  uncertainty_to_signal_ratio?: number | null;
  isotonic_calibration_applied: boolean;
  [key: string]: unknown;
};

type ProductKey = "p_gt_0p30m" | "iqr" | "p95" | "mean" | "spread";

const PRODUCT_LABELS: Record<ProductKey, string> = {
  p_gt_0p30m: "P(WD > 0.30 m)",
  iqr: "WD IQR (m)",
  p95: "WD p95 (m)",
  mean: "Mean WD (m)",
  spread: "Ensemble spread (m)",
};

const PRODUCT_CAPTIONS: Record<ProductKey, { caption: string; insight: string }> = {
  p_gt_0p30m: {
    caption: "Calibrated exceedance-probability map at the selected lead time.",
    insight: "Values are probabilities for WD above 0.30 m, so this map foregrounds risk of crossing a meaningful depth threshold rather than expected depth.",
  },
  iqr: {
    caption: "Interquartile depth spread at the selected lead time.",
    insight: "Large IQR means the middle 50% of ensemble members disagree locally, which is a robust uncertainty-width diagnostic.",
  },
  p95: {
    caption: "Upper-tail calibrated depth at the selected lead time.",
    insight: "p95 highlights plausible high-end response under the ensemble, not a deterministic worst case.",
  },
  mean: {
    caption: "Calibrated ensemble-mean water depth at the selected lead time.",
    insight: "Mean depth is useful for expected response, but should be read with spread and probability maps for UQ context.",
  },
  spread: {
    caption: "Ensemble standard-deviation map at the selected lead time.",
    insight: "Spread shows where members disagree; high spread over low mean depth can be as important as high spread over deep regions.",
  },
};

const TERMINAL_STATUSES = new Set(["COMPLETED", "FAILED", "CANCELED", "EXPIRED", "DELETED"]);

function formatNumber(value: unknown, digits = 3): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function formatPercent(value: unknown, digits = 1): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function formatDuration(seconds: unknown): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "Not enough history";
  const rounded = Math.max(0, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (hours > 0) return `${hours}h ${minutes.toString().padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m ${secs.toString().padStart(2, "0")}s`;
  return `${secs}s`;
}

function formatRunLabel(run: RunData | null | undefined): string {
  if (!run) return "Select run";
  return `${run.label || run.run_id.slice(0, 10)} · ${run.status}`;
}

function InfoTip({ text }: { text: string }) {
  return (
    <span
      aria-label={text}
      title={text}
      style={{
        display: "inline-grid",
        placeItems: "center",
        width: 17,
        height: 17,
        marginLeft: 6,
        borderRadius: 999,
        border: "1px solid #99b8b5",
        color: "#0f766e",
        fontSize: 11,
        fontWeight: 800,
        lineHeight: 1,
        cursor: "help",
        background: "#f0fdfa",
        verticalAlign: "text-bottom",
      }}
    >
      i
    </span>
  );
}

function FigureCaption({
  title,
  caption,
  insight,
}: {
  title: string;
  caption: string;
  insight: string;
}) {
  return (
    <figcaption
      style={{
        display: "grid",
        gap: 4,
        padding: "10px 14px 12px",
        color: "#475569",
        fontSize: 12,
        lineHeight: 1.45,
      }}
    >
      <strong style={{ color: "#0f172a", fontSize: 13, fontWeight: 700 }}>{title}</strong>
      <span>
        {caption}
        <InfoTip text={insight} />
      </span>
    </figcaption>
  );
}

function FigureNote({ caption, insight }: { caption: string; insight: string }) {
  return (
    <p
      style={{
        margin: "10px 0 0",
        color: "#475569",
        fontSize: 12.5,
        lineHeight: 1.5,
      }}
    >
      {caption}
      <InfoTip text={insight} />
    </p>
  );
}

const REFERENCE_DIAGNOSTIC_CODES = new Set([
  "below_candidate_reference",
  "above_candidate_reference",
  "below_reference_warning",
  "above_reference_warning",
]);

const DECISION_ELIGIBLE_CODES = new Set([
  "multivariate_outlier",
  "high_uncertainty_to_signal",
  "high_impact_high_uncertainty",
  "large_calibration_shift",
  "population_reinforced_candidate",
  "deterministic_control_sample",
]);

function isDecisionEligibleFlag(flag: MonitoringFlag): boolean {
  if (flag.details?.decision_eligible === true) return true;
  if (flag.details?.decision_eligible === false) return false;
  return DECISION_ELIGIBLE_CODES.has(flag.code) && flag.severity === "candidate";
}

function isReferenceDiagnosticFlag(flag: MonitoringFlag): boolean {
  return flag.details?.decision_eligible === false || REFERENCE_DIAGNOSTIC_CODES.has(flag.code);
}

const DESCRIPTOR_LABELS: Record<string, string> = {
  calibrated_max_mean_wd_m: "Peak calibrated water depth",
  raw_max_mean_wd_m: "Peak raw water depth",
  max_mean_wd_m: "Peak mean water depth",
  stage_max: "Peak coastal stage",
  stage_min: "Minimum coastal stage",
  stage_range: "Coastal-stage range",
  stage_mean: "Mean coastal stage",
  stage_peak_lead_hours: "Lead time to peak coastal stage",
  stage_max_rise_rate_per_hour: "Fastest coastal-stage rise",
  stage_max_fall_rate_per_hour: "Fastest coastal-stage fall",
  precipitation_total: "Total precipitation",
  precipitation_mean: "Mean precipitation",
  precipitation_max: "Peak precipitation",
  precipitation_active_hours: "Precipitation duration",
  precipitation_peak_lead_hours: "Lead time to peak precipitation",
  peak_precip_to_peak_stage_lag_hours: "Rainfall-to-stage timing offset",
  "peak_expected_flooded_area_fraction_wettable_gt_0.05m": "Peak expected flooded fraction above 0.05 m",
  "peak_expected_flooded_area_km2_gt_0.05m": "Peak expected flooded area above 0.05 m",
  "peak_expected_flooded_area_lead_hours_gt_0.05m": "Lead time to peak expected flooded area",
  peak_area_weighted_iqr_wd_m: "Area-weighted uncertainty width",
  peak_area_weighted_central_90_wd_m: "Area-weighted central 90% width",
  peak_area_weighted_total_ensemble_spread_wd_m: "Peak total ensemble spread",
  peak_area_weighted_between_checkpoint_spread_wd_m: "Peak checkpoint disagreement",
  peak_area_weighted_between_checkpoint_variance_share: "Checkpoint-disagreement share",
  peak_high_checkpoint_disagreement_area_fraction_wettable: "High-disagreement footprint",
  peak_between_checkpoint_disagreement_lead_hours: "Lead time to peak checkpoint disagreement",
  uncertainty_to_signal_ratio: "Uncertainty-to-signal ratio",
  "calibration_peak_area_shift_percentage_points_gt_0.05m": "Calibration shift in flooded area",
  max_abs_calibration_shift_percentage_points_by_threshold: "Largest calibration shift",
};

const CANDIDATE_REASON_LABELS: Record<string, string> = {
  multivariate_outlier: "Joint input or forecast pattern is outside the reference population",
  high_uncertainty_to_signal: "Uncertainty is high relative to predicted signal",
  high_impact_high_uncertainty: "Broad affected area coincides with elevated uncertainty",
  high_checkpoint_disagreement_spread: "Trained checkpoints disagree more than expected",
  high_checkpoint_disagreement_share: "Checkpoint disagreement dominates ensemble spread",
  broad_checkpoint_disagreement_footprint: "Checkpoint disagreement covers a broad area",
  large_calibration_shift: "Calibration substantially changed the exceedance footprint",
  population_reinforced_candidate: "Individual signal aligns with persistent population drift",
  deterministic_control_sample: "Selected by deterministic sampling for review-set balance",
  below_candidate_reference: "Historical reference-envelope-only selection",
  above_candidate_reference: "Historical reference-envelope-only selection",
  monitoring_recommended: "Monitoring recommended retraining review",
};

const CANDIDATE_STATUS_LABELS: Record<string, string> = {
  NEW: "New review item",
  REVIEWED: "Reviewed",
  SELECTED_FOR_HECRAS: "Selected for HEC-RAS simulation",
  REJECTED: "Rejected after review",
  SIMULATED: "HEC-RAS simulated",
};

function formatDescriptorLabel(descriptor?: string | null): string {
  if (!descriptor) return "Joint monitored pattern";
  if (DESCRIPTOR_LABELS[descriptor]) return DESCRIPTOR_LABELS[descriptor];
  const threshold = descriptor.match(/_gt_([0-9.]+)m/);
  const thresholdText = threshold ? ` above ${threshold[1]} m` : "";
  return descriptor
    .replace(/_gt_[0-9.]+m/g, "")
    .replace(/wd/g, "water depth")
    .replace(/iqr/g, "IQR")
    .replace(/p95/g, "95th percentile")
    .replace(/p05/g, "5th percentile")
    .replace(/km2/g, "km²")
    .replace(/m2/g, "m²")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .concat(thresholdText);
}

function formatCandidateReason(reason?: string | null): string {
  if (!reason) return "Monitoring recommended retraining review";
  return CANDIDATE_REASON_LABELS[reason] ?? formatDescriptorLabel(reason);
}

function formatCandidateStatus(status?: string | null): string {
  if (!status) return "Not selected";
  return CANDIDATE_STATUS_LABELS[status] ?? status.replace(/_/g, " ").toLowerCase();
}

function formatInitialConditionMode(mode?: string | null): string {
  if (mode === "forcing_conditioned_baseline") return "Forcing-conditioned baseline history";
  if (mode === "dry") return "Dry diagnostic history";
  return "Initial water-depth history";
}

function formatMonitoringFlag(flag: MonitoringFlag): string {
  const label = formatDescriptorLabel(flag.descriptor);
  switch (flag.code) {
    case "above_candidate_reference":
      return `${label} is above the historical reference envelope.`;
    case "below_candidate_reference":
      return `${label} is below the historical reference envelope.`;
    case "above_reference_warning":
      return `${label} is near the upper edge of the reference envelope.`;
    case "below_reference_warning":
      return `${label} is near the lower edge of the reference envelope.`;
    case "multivariate_outlier":
      return "The combined scenario pattern is unusual relative to the reference population.";
    case "high_uncertainty_to_signal":
      return "Ensemble uncertainty is high relative to the predicted flood signal.";
    case "high_impact_high_uncertainty":
      return "A broad affected area coincides with elevated ensemble uncertainty.";
    case "high_checkpoint_disagreement_spread":
      return "Trained checkpoints disagree more strongly than expected.";
    case "high_checkpoint_disagreement_share":
      return "Checkpoint-to-checkpoint disagreement explains an unusually large share of ensemble spread.";
    case "broad_checkpoint_disagreement_footprint":
      return "Checkpoint-driven disagreement covers an unusually large part of the wettable domain.";
    case "large_calibration_shift":
      return "Calibration materially changed the exceedance footprint.";
    case "population_reinforced_candidate":
      return "This run aligns with a persistent monitoring signal in recent submissions.";
    case "deterministic_control_sample":
      return "Selected by deterministic sampling to keep the review set balanced.";
    default:
      return flag.message
        .replace(/calibrated_max_mean_wd_m/g, "peak calibrated water depth")
        .replace(/raw_max_mean_wd_m/g, "peak raw water depth")
        .replace(/peak_area_weighted_iqr_wd_m/g, "area-weighted uncertainty width")
        .replace(/peak_area_weighted_between_checkpoint_spread_wd_m/g, "peak checkpoint disagreement")
        .replace(/peak_area_weighted_between_checkpoint_variance_share/g, "checkpoint-disagreement share")
        .replace(/peak_high_checkpoint_disagreement_area_fraction_wettable/g, "high-disagreement footprint")
        .replace(/max_abs_calibration_shift_percentage_points_by_threshold/g, "largest calibration shift")
        .replace(/_/g, " ");
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function artifactUrlFor(runId: string, artifactId: string) {
  return `/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}`;
}

function numericArray(summary: Summary | null | undefined, key: string): number[] {
  const raw = summary?.[key];
  if (!Array.isArray(raw)) return [];
  return raw.map((value) => Number(value)).filter((value) => Number.isFinite(value));
}

function exceedanceEntries(summary: Summary | null): { threshold: string; value: number }[] {
  if (!summary) return [];
  const entries: { threshold: string; value: number }[] = [];
  for (const [key, raw] of Object.entries(summary)) {
    const match = key.match(/^p_wd_gt_([0-9eE+\-.]+)m_mean$/);
    if (!match) continue;
    const value = typeof raw === "number" ? raw : Number(raw);
    if (!Number.isFinite(value)) continue;
    entries.push({ threshold: match[1], value });
  }
  return entries.sort((a, b) => Number(a.threshold) - Number(b.threshold));
}

function exceedanceAreaEntries(summary: Summary | null): { threshold: number; peakFraction: number; peakKm2: number }[] {
  const raw = summary?.exceedance_by_threshold_m;
  if (!raw || typeof raw !== "object") return [];
  return Object.entries(raw as Record<string, Record<string, unknown>>)
    .flatMap(([key, value]) => {
      const threshold = Number(value.threshold_m ?? key);
      const peakFraction = typeof value.peak_expected_area_fraction_wettable === "number"
        ? value.peak_expected_area_fraction_wettable
        : Number(value.peak_expected_area_fraction_wettable);
      if (!Number.isFinite(threshold) || !Number.isFinite(peakFraction)) return [];
      const peakKm2 = typeof value.peak_expected_area_km2 === "number"
        ? value.peak_expected_area_km2
        : Number(value.peak_expected_area_km2 ?? 0);
      return [{ threshold, peakFraction, peakKm2: Number.isFinite(peakKm2) ? peakKm2 : 0 }];
    })
    .sort((a, b) => a.threshold - b.threshold);
}

function parseHydrograph(csvText: string): { t_h: number[]; stage: number[]; precip: number[] } {
  const lines = csvText.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) return { t_h: [], stage: [], precip: [] };
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const stageIdx = header.findIndex((h) => h.includes("stage"));
  const precipIdx = header.findIndex((h) => h.includes("precip"));
  const timeIdx = header.findIndex((h) => h.startsWith("time") || h === "t" || h === "hour" || h === "hours");
  if (stageIdx < 0 || precipIdx < 0) return { t_h: [], stage: [], precip: [] };
  const t_h: number[] = [];
  const stage: number[] = [];
  const precip: number[] = [];
  lines.slice(1).forEach((line, idx) => {
    const cols = line.split(",");
    const stageVal = Number(cols[stageIdx]);
    const precipVal = Number(cols[precipIdx]);
    if (!Number.isFinite(stageVal) || !Number.isFinite(precipVal)) return;
    let t = idx * (DEFAULT_FORCING_DT_SECONDS / 3600);
    if (timeIdx >= 0) {
      const rawT = Number(cols[timeIdx]);
      if (Number.isFinite(rawT)) {
        const lower = header[timeIdx];
        t = lower.includes("hour") ? rawT : rawT / 3600;
      }
    }
    t_h.push(t);
    stage.push(stageVal);
    precip.push(precipVal);
  });
  return { t_h, stage, precip };
}

function LineSparkline({
  series,
  color,
  height = 80,
  yLabel,
}: {
  series: { t: number; y: number }[];
  color: string;
  height?: number;
  yLabel: string;
}) {
  if (series.length < 2) {
    return <p style={{ color: "#64748b", fontStyle: "italic" }}>Not enough samples to plot.</p>;
  }
  const padding = { left: 38, right: 12, top: 8, bottom: 22 };
  const innerW = 480;
  const innerH = height;
  const totalW = innerW + padding.left + padding.right;
  const totalH = innerH + padding.top + padding.bottom;
  const tMin = series[0].t;
  const tMax = series[series.length - 1].t;
  const yMin = Math.min(...series.map((s) => s.y));
  const yMax = Math.max(...series.map((s) => s.y));
  const yRange = yMax - yMin || 1;
  const tRange = tMax - tMin || 1;
  const points = series
    .map((s) => {
      const x = padding.left + ((s.t - tMin) / tRange) * innerW;
      const y = padding.top + (1 - (s.y - yMin) / yRange) * innerH;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg width={totalW} height={totalH} role="img" aria-label={yLabel} style={{ maxWidth: "100%" }}>
      <rect x={padding.left} y={padding.top} width={innerW} height={innerH} fill="#f1f5f9" stroke="#cbd5e1" />
      <polyline fill="none" stroke={color} strokeWidth={2} points={points} />
      <text x={padding.left} y={padding.top - 2} fontSize={10} fill="#475569">
        {yLabel}: {yMin.toFixed(3)} – {yMax.toFixed(3)}
      </text>
      <text x={padding.left} y={totalH - 6} fontSize={10} fill="#475569">
        {tMin.toFixed(2)} h
      </text>
      <text x={padding.left + innerW - 38} y={totalH - 6} fontSize={10} fill="#475569">
        {tMax.toFixed(2)} h
      </text>
    </svg>
  );
}

function MultiLineChart({
  series,
  yLabel,
  height = 180,
}: {
  series: { label: string; color: string; points: { t: number; y: number }[] }[];
  yLabel: string;
  height?: number;
}) {
  const valid = series
    .map((item) => ({ ...item, points: item.points.filter((p) => Number.isFinite(p.t) && Number.isFinite(p.y)) }))
    .filter((item) => item.points.length >= 2);
  if (valid.length === 0) return <p className="muted">Not enough samples to plot.</p>;
  const all = valid.flatMap((item) => item.points);
  const pad = { left: 54, right: 18, top: 20, bottom: 34 };
  const width = 760;
  const innerW = width - pad.left - pad.right;
  const innerH = height;
  const totalH = innerH + pad.top + pad.bottom;
  const tMin = Math.min(...all.map((p) => p.t));
  const tMax = Math.max(...all.map((p) => p.t));
  const yMin = Math.min(0, ...all.map((p) => p.y));
  const yMax = Math.max(...all.map((p) => p.y), 1e-6);
  const xFor = (t: number) => pad.left + ((t - tMin) / (tMax - tMin || 1)) * innerW;
  const yFor = (y: number) => pad.top + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;
  return (
    <svg viewBox={`0 0 ${width} ${totalH}`} role="img" aria-label={yLabel} className="poi-chart">
      <rect x={pad.left} y={pad.top} width={innerW} height={innerH} fill="#ffffff" stroke="#dbe3ea" />
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <line
          key={tick}
          x1={pad.left}
          x2={pad.left + innerW}
          y1={pad.top + tick * innerH}
          y2={pad.top + tick * innerH}
          stroke="#eef2f7"
        />
      ))}
      {valid.map((item) => (
        <polyline
          key={item.label}
          fill="none"
          stroke={item.color}
          strokeWidth={2.4}
          strokeLinecap="round"
          strokeLinejoin="round"
          points={item.points.map((p) => `${xFor(p.t).toFixed(2)},${yFor(p.y).toFixed(2)}`).join(" ")}
        />
      ))}
      <text x={pad.left} y={14} fontSize={11} fontWeight={700} fill="#0f172a">{yLabel}</text>
      <text x={pad.left} y={totalH - 10} fontSize={10} fill="#64748b">{tMin.toFixed(1)} h</text>
      <text x={pad.left + innerW} y={totalH - 10} textAnchor="end" fontSize={10} fill="#64748b">{tMax.toFixed(1)} h</text>
      <text x={pad.left - 8} y={yFor(yMax) + 4} textAnchor="end" fontSize={10} fill="#64748b">{yMax.toFixed(2)}</text>
      <text x={pad.left - 8} y={yFor(yMin) + 4} textAnchor="end" fontSize={10} fill="#64748b">{yMin.toFixed(2)}</text>
      {valid.map((item, idx) => (
        <g key={`legend-${item.label}`} transform={`translate(${pad.left + 8 + idx * 150}, ${pad.top + 16})`}>
          <line x1={0} x2={18} y1={0} y2={0} stroke={item.color} strokeWidth={3} strokeLinecap="round" />
          <text x={24} y={4} fontSize={11} fontWeight={700} fill="#0f172a">{item.label}</text>
        </g>
      ))}
    </svg>
  );
}

function BarSeriesChart({
  points,
  color,
  yLabel,
  height = 120,
}: {
  points: { t: number; y: number }[];
  color: string;
  yLabel: string;
  height?: number;
}) {
  const valid = points.filter((p) => Number.isFinite(p.t) && Number.isFinite(p.y));
  if (valid.length < 1) return <p className="muted">Not enough samples to plot.</p>;
  const pad = { left: 54, right: 18, top: 20, bottom: 34 };
  const width = 760;
  const innerW = width - pad.left - pad.right;
  const innerH = height;
  const totalH = innerH + pad.top + pad.bottom;
  const tMin = valid[0].t;
  const tMax = valid[valid.length - 1].t;
  const yMax = Math.max(...valid.map((p) => p.y), 1e-6);
  const barW = Math.max(1, innerW / valid.length - 1);
  return (
    <svg viewBox={`0 0 ${width} ${totalH}`} role="img" aria-label={yLabel} className="poi-chart">
      <rect x={pad.left} y={pad.top} width={innerW} height={innerH} fill="#ffffff" stroke="#dbe3ea" />
      {valid.map((p, idx) => {
        const h = (Math.max(0, p.y) / yMax) * innerH;
        return (
          <rect
            key={idx}
            x={pad.left + idx * (innerW / valid.length)}
            y={pad.top + innerH - h}
            width={barW}
            height={h}
            fill={color}
            opacity={0.86}
          />
        );
      })}
      <text x={pad.left} y={14} fontSize={11} fontWeight={700} fill="#0f172a">{yLabel}</text>
      <text x={pad.left} y={totalH - 10} fontSize={10} fill="#64748b">{tMin.toFixed(1)} h</text>
      <text x={pad.left + innerW} y={totalH - 10} textAnchor="end" fontSize={10} fill="#64748b">{tMax.toFixed(1)} h</text>
      <text x={pad.left - 8} y={pad.top + 4} textAnchor="end" fontSize={10} fill="#64748b">{yMax.toFixed(2)}</text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Per-cell inspector, click hit-testing, and chart primitives.
// ---------------------------------------------------------------------------

type SpreadDecompositionSummary = {
  n_groups: number;
  groups: string[];
  between_share: number;
  within_share: number;
};

type ReliabilityCurve = {
  threshold_m: number;
  raw_bin_centers: number[];
  calibrated_means_per_bin: (number | null)[];
  raw_distribution_counts: number[];
  calibrated_distribution_counts: number[];
  raw_mean: number;
  calibrated_mean: number;
};

type ReliabilityCurves = {
  applied: boolean;
  n_wettable_cells: number;
  peak_time_idx: number;
  peak_lead_hours: number;
  thresholds_m: number[];
  bin_edges: number[];
  curves: Record<string, ReliabilityCurve>;
};

type SpreadToPeakHistogram = {
  n_wettable_cells: number;
  n_signal_cells: number;
  mean_ratio: number | null;
  median_ratio: number | null;
  p95_ratio: number | null;
  bin_edges: number[];
  counts: number[];
};

type CellContributionRow = {
  cell_index: number;
  x?: number;
  y?: number;
  peak_expected_area_m2: number;
  contribution_share: number;
  peak_probability: number;
  peak_mean_wd_m: number;
  peak_lead_hours: number | null;
  cell_area_m2: number;
};

type CellContributionLeaderboard = {
  threshold_m: number;
  total_peak_expected_area_km2: number;
  rows: CellContributionRow[];
};

type ComparePayload = {
  run_a: RunData;
  run_b: RunData;
  artifacts_a: Artifact[];
  artifacts_b: Artifact[];
  summary_a: Summary;
  summary_b: Summary;
  summary_delta: {
    key: string;
    label: string;
    unit: string;
    a: number;
    b: number;
    delta: number;
    percent_delta: number | null;
  }[];
  common_lead_time_hours: number[];
  delta_frames: Record<string, string[]>;
};

function ReliabilityCurveCard({ data }: { data: ReliabilityCurves }) {
  // Two-curve reliability diagram + per-threshold histogram strip.
  // X-axis = raw calibrated-ensemble exceedance probability per cell, binned;
  // Y-axis = mean post-isotonic probability for cells in that raw bin.
  // The 45-degree reference line means "calibration was a no-op". Deviations
  // above/below the line tell the user where the model over- or
  // under-confidently predicted exceedance for this run.
  const width = 720;
  const padL = 56;
  const padR = 18;
  const padT = 18;
  const histH = 28;
  const padB = 36 + histH;
  const innerW = width - padL - padR;
  const innerH = 260;
  const totalH = innerH + padT + padB;
  const xFor = (p: number) => padL + Math.max(0, Math.min(1, p)) * innerW;
  const yFor = (p: number) => padT + (1 - Math.max(0, Math.min(1, p))) * innerH;
  const palette: Record<string, string> = { "0.05": "#0e7490", "0.3": "#b91c1c" };
  const entries = Object.entries(data.curves).sort((a, b) => Number(a[0]) - Number(b[0]));
  return (
    <svg viewBox={`0 0 ${width} ${totalH}`} role="img" aria-label="Reliability diagram" className="poi-chart">
      <rect x={padL} y={padT} width={innerW} height={innerH} fill="#ffffff" stroke="#e2e8f0" />
      {/* Reference grid + diagonal */}
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={padL} x2={padL + innerW} y1={yFor(g)} y2={yFor(g)} stroke="#f1f5f9" />
          <text x={padL - 6} y={yFor(g) + 3.5} fontSize={10} textAnchor="end" fill="#64748b">
            {(g * 100).toFixed(0)}%
          </text>
          <line x1={xFor(g)} x2={xFor(g)} y1={padT} y2={padT + innerH} stroke="#f8fafc" />
        </g>
      ))}
      <line
        x1={xFor(0)} y1={yFor(0)} x2={xFor(1)} y2={yFor(1)}
        stroke="#cbd5e1" strokeWidth={1.2} strokeDasharray="4 3"
      />
      <text x={xFor(1) - 4} y={yFor(1) + 14} fontSize={9.5} textAnchor="end" fill="#94a3b8">
        identity (no calibration change)
      </text>

      {/* Per-threshold curve */}
      {entries.map(([thrKey, curve]) => {
        const stroke = palette[thrKey] ?? "#475569";
        const pts: string[] = [];
        for (let i = 0; i < curve.raw_bin_centers.length; i += 1) {
          const y = curve.calibrated_means_per_bin[i];
          if (y === null || !Number.isFinite(y)) continue;
          pts.push(`${xFor(curve.raw_bin_centers[i]).toFixed(2)},${yFor(y).toFixed(2)}`);
        }
        if (pts.length < 2) return null;
        return (
          <polyline
            key={`curve-${thrKey}`}
            points={pts.join(" ")}
            fill="none"
            stroke={stroke}
            strokeWidth={2.4}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        );
      })}

      {/* Per-threshold "current value" markers (raw_mean, calibrated_mean) */}
      {entries.map(([thrKey, curve]) => {
        const stroke = palette[thrKey] ?? "#475569";
        return (
          <g key={`marker-${thrKey}`}>
            <circle cx={xFor(curve.raw_mean)} cy={yFor(curve.calibrated_mean)} r={4} fill={stroke} stroke="#ffffff" strokeWidth={1.4} />
          </g>
        );
      })}

      {/* Axis labels */}
      <text x={padL} y={padT - 4} fontSize={11} fontWeight={700} fill="#0f172a">
        Calibrated P(WD &gt; threshold) vs raw P
      </text>
      <text x={padL + innerW / 2} y={padT + innerH + 22} fontSize={11} textAnchor="middle" fill="#475569">
        Raw ensemble probability (wettable cells, peak time)
      </text>

      {/* Histogram strip showing raw-probability density per threshold */}
      {entries.map(([thrKey, curve], idx) => {
        const stroke = palette[thrKey] ?? "#475569";
        const total = Math.max(1, curve.raw_distribution_counts.reduce((a, b) => a + b, 0));
        const yBase = padT + innerH + 30 + idx * (histH / 2 + 2);
        const barW = innerW / curve.raw_distribution_counts.length;
        return (
          <g key={`hist-${thrKey}`}>
            <text x={padL - 6} y={yBase + 9} fontSize={9.5} textAnchor="end" fill={stroke} fontWeight={700}>
              &gt; {curve.threshold_m.toFixed(2)} m
            </text>
            {curve.raw_distribution_counts.map((count, i) => {
              const h = (count / total) * (histH / 2);
              return (
                <rect
                  key={i}
                  x={padL + i * barW + 0.5}
                  y={yBase + (histH / 2 - h)}
                  width={Math.max(0.5, barW - 1)}
                  height={Math.max(0.5, h)}
                  fill={stroke}
                  opacity={0.55}
                />
              );
            })}
          </g>
        );
      })}

      {/* Legend */}
      {entries.map(([thrKey], idx) => {
        const stroke = palette[thrKey] ?? "#475569";
        return (
          <g key={`legend-${thrKey}`} transform={`translate(${padL + 8 + idx * 130}, ${padT + 12})`}>
            <line x1={0} x2={18} y1={0} y2={0} stroke={stroke} strokeWidth={3} strokeLinecap="round" />
            <text x={24} y={3.5} fontSize={11} fontWeight={700} fill="#0f172a">
              &gt; {Number(thrKey).toFixed(2)} m
            </text>
          </g>
        );
      })}
    </svg>
  );
}

type GeometryMeta = {
  n_cells: number;
  bounds: { x_min: number; x_max: number; y_min: number; y_max: number };
  x: number[];
  y: number[];
  wettable: boolean[];
  image_data_viewport?: { left: number; right: number; top: number; bottom: number };
  image_data_bounds?: { x_min: number; x_max: number; y_min: number; y_max: number };
  elevation_m?: (number | null)[];
  slope?: (number | null)[];
  flow_accumulation?: (number | null)[];
};

type CellSeries = {
  cell_index: number;
  n_members: number;
  n_time: number;
  lead_time_hours: number[];
  raw_members_wd: number[][];
  calibrated_members_wd: number[][];
  calibrated_mean_wd: number[];
  calibrated_q05_wd: number[];
  calibrated_q50_wd: number[];
  calibrated_q95_wd: number[];
  calibrated_exceedance_prob: Record<string, number[]>;
  calibrated_member_arrival_hours: Record<string, (number | null)[]>;
  peak_calibrated_mean_wd_m: number;
  peak_calibrated_lead_hours: number;
  wettable: boolean;
  elevation_m?: number | null;
  slope?: number | null;
  flow_accumulation?: number | null;
};

// Click → UTM transform + nearest-cell lookup. New runs include
// `image_data_viewport`, measured server-side from the exact Matplotlib
// Rivanna-style layout. Older runs fall back to the full image rectangle.
function geometryViewport(geometry: GeometryMeta) {
  return geometry.image_data_viewport ?? { left: 0, right: 1, top: 0, bottom: 1 };
}

function geometryDataBounds(geometry: GeometryMeta) {
  return geometry.image_data_bounds ?? geometry.bounds;
}

function nearestCellAtClick(
  click: { offsetX: number; offsetY: number; width: number; height: number },
  geometry: GeometryMeta,
): number | null {
  if (!geometry || geometry.n_cells === 0) return null;
  const imageX = click.offsetX / Math.max(click.width, 1);
  const imageY = click.offsetY / Math.max(click.height, 1);
  const viewport = geometryViewport(geometry);
  const viewportWidth = Math.max(viewport.right - viewport.left, 1e-9);
  const viewportHeight = Math.max(viewport.bottom - viewport.top, 1e-9);
  if (
    imageX < viewport.left ||
    imageX > viewport.right ||
    imageY < viewport.top ||
    imageY > viewport.bottom
  ) {
    return null;
  }
  const relX = (imageX - viewport.left) / viewportWidth;
  const relY = (imageY - viewport.top) / viewportHeight;
  const { x_min, x_max, y_min, y_max } = geometryDataBounds(geometry);
  // Pixel Y grows downward; UTM Y grows upward.
  const targetX = x_min + relX * (x_max - x_min);
  const targetY = y_max - relY * (y_max - y_min);
  let best = -1;
  let bestDist = Infinity;
  for (let i = 0; i < geometry.n_cells; i += 1) {
    if (!geometry.wettable[i]) continue;
    const dx = geometry.x[i] - targetX;
    const dy = geometry.y[i] - targetY;
    const d = dx * dx + dy * dy;
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best >= 0 ? best : null;
}

function cellImageFraction(cellIndex: number, geometry: GeometryMeta): { fx: number; fy: number } | null {
  if (
    cellIndex < 0 ||
    cellIndex >= geometry.n_cells ||
    !Number.isFinite(geometry.x[cellIndex]) ||
    !Number.isFinite(geometry.y[cellIndex])
  ) {
    return null;
  }
  const viewport = geometryViewport(geometry);
  const { x_min, x_max, y_min, y_max } = geometryDataBounds(geometry);
  const xRange = x_max - x_min;
  const yRange = y_max - y_min;
  if (!Number.isFinite(xRange) || !Number.isFinite(yRange) || xRange <= 0 || yRange <= 0) {
    return null;
  }
  const relX = (geometry.x[cellIndex] - x_min) / xRange;
  const relY = (y_max - geometry.y[cellIndex]) / yRange;
  if (!Number.isFinite(relX) || !Number.isFinite(relY)) {
    return null;
  }
  return {
    fx: viewport.left + Math.min(1, Math.max(0, relX)) * (viewport.right - viewport.left),
    fy: viewport.top + Math.min(1, Math.max(0, relY)) * (viewport.bottom - viewport.top),
  };
}

function PoiHydrograph({ series }: { series: CellSeries }) {
  // Fan chart: 60 member traces drawn faintly behind a p05-p95 ribbon and a
  // bold median line. The ribbon communicates ensemble spread at a glance;
  // the spaghetti behind it lets researchers see member structure (clusters,
  // outliers). All in one SVG, no chart library.
  const width = 760;
  const height = 260;
  const padL = 56;
  const padR = 18;
  const padT = 18;
  const padB = 36;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const t = series.lead_time_hours;
  if (t.length < 2) return <p className="muted">Not enough timesteps to plot.</p>;
  const tMin = t[0];
  const tMax = t[t.length - 1];
  const tSpan = Math.max(tMax - tMin, 1e-9);
  const yMax = Math.max(
    series.peak_calibrated_mean_wd_m,
    ...series.calibrated_q95_wd,
    1e-3,
  );
  const ySpan = Math.max(yMax, 1e-9);
  const xFor = (tt: number) => padL + ((tt - tMin) / tSpan) * innerW;
  const yFor = (yy: number) => padT + (1 - yy / ySpan) * innerH;
  const memberPaths = series.calibrated_members_wd.map((trace, m) => {
    const d = trace
      .map((v, i) => `${i === 0 ? "M" : "L"} ${xFor(t[i]).toFixed(2)} ${yFor(v).toFixed(2)}`)
      .join(" ");
    return <path key={m} d={d} fill="none" stroke="rgba(14, 116, 144, 0.18)" strokeWidth={0.7} />;
  });
  const ribbon = (() => {
    const top = t.map((tt, i) => `${xFor(tt).toFixed(2)},${yFor(series.calibrated_q95_wd[i]).toFixed(2)}`);
    const bottom = t.map((tt, i) => `${xFor(tt).toFixed(2)},${yFor(series.calibrated_q05_wd[i]).toFixed(2)}`).reverse();
    return `M ${top.concat(bottom).join(" L ")} Z`;
  })();
  const medianPath = t
    .map((tt, i) => `${i === 0 ? "M" : "L"} ${xFor(tt).toFixed(2)} ${yFor(series.calibrated_q50_wd[i]).toFixed(2)}`)
    .join(" ");
  const peakX = xFor(series.peak_calibrated_lead_hours);
  const yTicks = [0, yMax / 3, (2 * yMax) / 3, yMax];
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Ensemble depth fan at cell" className="poi-chart">
      <rect x={padL} y={padT} width={innerW} height={innerH} fill="#ffffff" stroke="#e2e8f0" />
      {yTicks.map((yy) => (
        <g key={yy}>
          <line x1={padL} x2={padL + innerW} y1={yFor(yy)} y2={yFor(yy)} stroke="#f1f5f9" />
          <text x={padL - 6} y={yFor(yy) + 3.5} fontSize={10} textAnchor="end" fill="#64748b">
            {yy.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={padL} y={padT - 4} fontSize={11} fontWeight={700} fill="#0f172a">
        Calibrated depth at cell {series.cell_index}
      </text>
      <text x={padL + innerW} y={padT - 4} fontSize={10} textAnchor="end" fill="#64748b">
        m
      </text>
      {memberPaths}
      <path d={ribbon} fill="rgba(14, 116, 144, 0.20)" stroke="none" />
      <path d={medianPath} fill="none" stroke="#0e7490" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" />
      <line x1={peakX} x2={peakX} y1={padT} y2={padT + innerH} stroke="#dc2626" strokeWidth={1.4} strokeDasharray="3 3" />
      <text x={peakX + 4} y={padT + 12} fontSize={10} fontWeight={700} fill="#b91c1c">
        peak {series.peak_calibrated_lead_hours.toFixed(2)} h
      </text>
      <text x={padL} y={height - 8} fontSize={10} fill="#64748b">
        {tMin.toFixed(1)} h
      </text>
      <text x={padL + innerW} y={height - 8} fontSize={10} textAnchor="end" fill="#64748b">
        {tMax.toFixed(1)} h
      </text>
    </svg>
  );
}

function PoiExceedanceTrace({ series }: { series: CellSeries }) {
  // P(WD > threshold) over time at this cell. Two curves (one per threshold)
  // share an axis bounded by [0, 1]. Communicates risk evolution clearly.
  const width = 760;
  const height = 180;
  const padL = 56;
  const padR = 18;
  const padT = 14;
  const padB = 32;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const t = series.lead_time_hours;
  if (t.length < 2) return <p className="muted">Not enough timesteps.</p>;
  const tMin = t[0];
  const tMax = t[t.length - 1];
  const tSpan = Math.max(tMax - tMin, 1e-9);
  const xFor = (tt: number) => padL + ((tt - tMin) / tSpan) * innerW;
  const yFor = (p: number) => padT + (1 - Math.max(0, Math.min(1, p))) * innerH;
  const palette: Record<string, string> = { "0.05": "#0e7490", "0.3": "#b91c1c" };
  const entries = Object.entries(series.calibrated_exceedance_prob).sort((a, b) => Number(a[0]) - Number(b[0]));
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Calibrated exceedance probability at cell" className="poi-chart">
      <rect x={padL} y={padT} width={innerW} height={innerH} fill="#ffffff" stroke="#e2e8f0" />
      {[0, 0.25, 0.5, 0.75, 1].map((y) => (
        <g key={y}>
          <line x1={padL} x2={padL + innerW} y1={yFor(y)} y2={yFor(y)} stroke="#f1f5f9" />
          <text x={padL - 6} y={yFor(y) + 3.5} fontSize={10} textAnchor="end" fill="#64748b">
            {(y * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      <text x={padL} y={padT - 4} fontSize={11} fontWeight={700} fill="#0f172a">
        Calibrated P(WD &gt; threshold)
      </text>
      {entries.map(([thr, probs]) => {
        const stroke = palette[thr] ?? "#475569";
        const d = probs
          .map((p, i) => `${i === 0 ? "M" : "L"} ${xFor(t[i]).toFixed(2)} ${yFor(p).toFixed(2)}`)
          .join(" ");
        return <path key={thr} d={d} fill="none" stroke={stroke} strokeWidth={2.2} strokeLinecap="round" />;
      })}
      {entries.map(([thr], idx) => {
        const stroke = palette[thr] ?? "#475569";
        return (
          <g key={`legend-${thr}`} transform={`translate(${padL + 8 + idx * 110}, ${height - 10})`}>
            <line x1={0} x2={18} y1={0} y2={0} stroke={stroke} strokeWidth={3} strokeLinecap="round" />
            <text x={24} y={3.5} fontSize={11} fontWeight={700} fill="#0f172a">
              &gt; {Number(thr).toFixed(2)} m
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function PoiArrivalHistogram({ series }: { series: CellSeries }) {
  // Per-member arrival distribution at the cell — answers "when does this
  // place get wet, across the ensemble's plausible futures?"
  const arrivalEntries = Object.entries(series.calibrated_member_arrival_hours)
    .filter(([, values]) => Array.isArray(values))
    .sort((a, b) => Number(a[0]) - Number(b[0]));
  const selectedArrival =
    arrivalEntries.find(([thr]) => Math.abs(Number(thr) - 0.05) < 1e-6) ??
    arrivalEntries.find(([thr]) => Math.abs(Number(thr) - 0.30) < 1e-6) ??
    arrivalEntries[0];
  if (!selectedArrival) {
    return <p className="muted">No arrival-threshold trace is available for this cell.</p>;
  }
  const [arrivalThreshold, arrivals] = selectedArrival;
  const thresholdLabel = Number(arrivalThreshold).toFixed(2);
  const validArrivals = arrivals.filter((a): a is number => a !== null && Number.isFinite(a));
  if (validArrivals.length === 0) {
    return <p className="muted">No ensemble member crosses {thresholdLabel} m at this cell.</p>;
  }
  const width = 760;
  const height = 160;
  const padL = 56;
  const padR = 18;
  const padT = 18;
  const padB = 28;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const tMin = series.lead_time_hours[0];
  const tMax = series.lead_time_hours[series.lead_time_hours.length - 1];
  const nBins = 14;
  const binW = (tMax - tMin) / nBins || 1;
  const bins = Array.from({ length: nBins }, () => 0);
  for (const a of validArrivals) {
    const k = Math.min(nBins - 1, Math.max(0, Math.floor((a - tMin) / binW)));
    bins[k] += 1;
  }
  const yMax = Math.max(...bins, 1);
  const xFor = (bIdx: number) => padL + (bIdx / nBins) * innerW;
  const yFor = (count: number) => padT + (1 - count / yMax) * innerH;
  const neverWet = arrivals.length - validArrivals.length;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Per-member arrival time distribution at cell" className="poi-chart">
      <rect x={padL} y={padT} width={innerW} height={innerH} fill="#ffffff" stroke="#e2e8f0" />
      <text x={padL} y={padT - 4} fontSize={11} fontWeight={700} fill="#0f172a">
        Per-member arrival time at WD &gt; {thresholdLabel} m
      </text>
      <text x={padL + innerW} y={padT - 4} fontSize={10} textAnchor="end" fill="#64748b">
        {validArrivals.length} / {arrivals.length} members{neverWet ? ` · ${neverWet} never wet` : ""}
      </text>
      {bins.map((count, i) => {
        if (count === 0) return null;
        const x = xFor(i) + 1;
        const y = yFor(count);
        const w = Math.max(1, innerW / nBins - 2);
        const h = padT + innerH - y;
        return <rect key={i} x={x} y={y} width={w} height={Math.max(h, 1)} fill="#1e3a8a" rx={2} />;
      })}
      <text x={padL} y={height - 8} fontSize={10} fill="#64748b">
        {tMin.toFixed(1)} h
      </text>
      <text x={padL + innerW} y={height - 8} fontSize={10} textAnchor="end" fill="#64748b">
        {tMax.toFixed(1)} h
      </text>
    </svg>
  );
}

function MapWithCellPicker({
  src,
  alt,
  geometry,
  onPick,
  selectedCellIndex,
}: {
  src: string;
  alt: string;
  geometry: GeometryMeta | null;
  onPick: (cellIndex: number) => void;
  selectedCellIndex: number | null;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [markerPos, setMarkerPos] = useState<{ x: number; y: number } | null>(null);

  const recomputePixels = useCallback(() => {
    const img = imgRef.current;
    if (!geometry || !img || selectedCellIndex === null) {
      setMarkerPos(null);
      return;
    }
    const frac = cellImageFraction(selectedCellIndex, geometry);
    if (!frac) {
      setMarkerPos(null);
      return;
    }
    setMarkerPos({ x: frac.fx * img.offsetWidth, y: frac.fy * img.offsetHeight });
  }, [geometry, selectedCellIndex]);

  useEffect(() => {
    recomputePixels();
    const img = imgRef.current;
    if (!img) return;
    const ro = new ResizeObserver(() => recomputePixels());
    ro.observe(img);
    return () => ro.disconnect();
  }, [recomputePixels]);

  useEffect(() => {
    if (selectedCellIndex === null) {
      setMarkerPos(null);
    }
  }, [selectedCellIndex]);

  if (!geometry) {
    return <img src={src} alt={alt} className="map-image" />;
  }
  const handleClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const offsetY = e.clientY - rect.top;
    const idx = nearestCellAtClick(
      { offsetX, offsetY, width: rect.width, height: rect.height },
      geometry,
    );
    if (idx !== null) {
      onPick(idx);
    }
  };

  return (
    <div style={{ position: "relative", textAlign: "center", overflow: "visible" }}>
      <div
        style={{
          position: "relative",
          display: "inline-block",
          maxWidth: "100%",
          margin: "16px auto 8px",
          lineHeight: 0,
          overflow: "visible",
          verticalAlign: "top",
        }}
      >
        <img
          ref={imgRef}
          src={src}
          alt={alt}
          className="map-image"
          onClick={handleClick}
          onLoad={recomputePixels}
          title="Click any cell to inspect its ensemble"
          style={{ cursor: "crosshair", margin: 0, display: "block", maxWidth: "100%", borderRadius: 12, boxShadow: "0 1px 4px rgba(15,23,42,0.08)" }}
        />
        {markerPos && (
          <div
            aria-hidden="true"
            style={{
              position: "absolute",
              left: `${markerPos.x}px`,
              top: `${markerPos.y}px`,
              width: 44,
              height: 44,
              transform: "translate(-50%, -50%)",
              pointerEvents: "none",
              zIndex: 20,
              filter: "drop-shadow(0 2px 5px rgba(0,0,0,0.42))",
              animation: "cell-marker-enter 280ms cubic-bezier(0.34,1.56,0.64,1) both",
            }}
          >
            <svg width="44" height="44" viewBox="0 0 44 44" fill="none" style={{ display: "block", overflow: "visible" }}>
              <circle cx="22" cy="22" r="14" stroke="#ffffff" strokeWidth="5" fill="rgba(255,255,255,0.10)" />
              <circle cx="22" cy="22" r="14" stroke="#0f172a" strokeWidth="2.25" fill="none" />
              <circle cx="22" cy="22" r="14" stroke="#f97316" strokeWidth="1.7" fill="none" style={{ animation: "cell-marker-pulse 2s ease-in-out infinite", transformOrigin: "22px 22px" }} />
              <line x1="22" y1="3" x2="22" y2="13" stroke="#ffffff" strokeWidth="5" strokeLinecap="round" />
              <line x1="22" y1="31" x2="22" y2="41" stroke="#ffffff" strokeWidth="5" strokeLinecap="round" />
              <line x1="3" y1="22" x2="13" y2="22" stroke="#ffffff" strokeWidth="5" strokeLinecap="round" />
              <line x1="31" y1="22" x2="41" y2="22" stroke="#ffffff" strokeWidth="5" strokeLinecap="round" />
              <line x1="22" y1="3" x2="22" y2="13" stroke="#0f172a" strokeWidth="2.25" strokeLinecap="round" />
              <line x1="22" y1="31" x2="22" y2="41" stroke="#0f172a" strokeWidth="2.25" strokeLinecap="round" />
              <line x1="3" y1="22" x2="13" y2="22" stroke="#0f172a" strokeWidth="2.25" strokeLinecap="round" />
              <line x1="31" y1="22" x2="41" y2="22" stroke="#0f172a" strokeWidth="2.25" strokeLinecap="round" />
              <circle cx="22" cy="22" r="4.6" fill="#f97316" stroke="#ffffff" strokeWidth="2.2" />
            </svg>
          </div>
        )}
      </div>
      {selectedCellIndex !== null && (
        <span
          style={{
            position: "absolute",
            left: 10,
            bottom: 10,
            padding: "4px 10px",
            background: "rgba(15,23,42,0.78)",
            color: "#f8fafc",
            borderRadius: 999,
            fontSize: "11.5px",
            fontWeight: 600,
            letterSpacing: "0.01em",
            backdropFilter: "blur(6px)",
          }}
        >
          Inspecting cell {selectedCellIndex} · click another point to switch
        </span>
      )}
      <style>{`
        @keyframes cell-marker-enter {
          from { opacity: 0; transform: translate(-50%,-50%) scale(0.3); }
          to   { opacity: 1; transform: translate(-50%,-50%) scale(1); }
        }
        @keyframes cell-marker-pulse {
          0%, 100% { opacity: 1; stroke-width: 1.7; }
          50%      { opacity: 0.5; stroke-width: 3; }
        }
      `}</style>
    </div>
  );
}

type ProductFrame = { artifactId: string; tIndex: number; isScrub: boolean };

function findMatchingPng(
  artifacts: Artifact[],
  view: "calibrated" | "raw",
  product: ProductKey,
): ProductFrame[] {
  // Two artifact families resolve to the same Forecast-maps slider:
  //   1. Dense per-step scrub frames: ``calibrated_{product}_scrub_t{NNN}.png``
  //      — currently only emitted for the calibrated view.
  //   2. Sparse 3-slot gallery frames: ``{view}_{product}_t{NNN}.png``.
  // Prefer (1) when it's available so the slider behaves like a real
  // forecast-step scrubber instead of a 3-slot toggle. Fall back to (2)
  // for the raw view (no scrub frames produced) and for older runs that
  // pre-date the scrub-frame contract for this product.
  if (view === "calibrated") {
    const scrubRe = new RegExp(`^calibrated_${product}_scrub_t(\\d{3})\\.png$`);
    const scrubMatches: ProductFrame[] = [];
    for (const artifact of artifacts) {
      const m = artifact.artifact_id.match(scrubRe);
      if (!m) continue;
      const tIndex = parseInt(m[1], 10);
      if (Number.isFinite(tIndex)) {
        scrubMatches.push({ artifactId: artifact.artifact_id, tIndex, isScrub: true });
      }
    }
    if (scrubMatches.length > 0) {
      return scrubMatches.sort((a, b) => a.tIndex - b.tIndex);
    }
  }
  // Slot fallback. Anchor the regex so the prefix doesn't accidentally
  // match ``{view}_{product}_scrub_t…`` (it won't because of the literal
  // 't' immediately after _, but explicit beats clever).
  const slotRe = new RegExp(`^${view}_${product}_t(\\d{3})\\.png$`);
  const slotMatches: ProductFrame[] = [];
  for (const artifact of artifacts) {
    const m = artifact.artifact_id.match(slotRe);
    if (!m) continue;
    const tIndex = parseInt(m[1], 10);
    if (Number.isFinite(tIndex)) {
      slotMatches.push({ artifactId: artifact.artifact_id, tIndex, isScrub: false });
    }
  }
  return slotMatches.sort((a, b) => a.tIndex - b.tIndex);
}

export default function RunDetails() {
  const params = useParams<{ runId: string }>();
  const runId = String(params.runId);
  const [run, setRun] = useState<RunData | null>(null);
  const [me, setMe] = useState<CurrentUser | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [error, setError] = useState("");
  const [actionBusy, setActionBusy] = useState(false);
  const [calibratedSummary, setCalibratedSummary] = useState<Summary | null>(null);
  const [rawSummary, setRawSummary] = useState<Summary | null>(null);
  const [forcing, setForcing] = useState<{ t_h: number[]; stage: number[]; precip: number[] } | null>(null);
  const [view, setView] = useState<"calibrated" | "raw">("calibrated");
  const [product, setProduct] = useState<ProductKey>("mean");
  const [timeSlot, setTimeSlot] = useState(0);
  type TabKey = "hazard" | "uncertainty" | "drivers" | "compare";
  const [tab, setTab] = useState<TabKey>("hazard");

  // Per-cell inspector state. Geometry is fetched once per run; cell series is
  // fetched on every click. Both are silent on failure so an older run without
  // these artifacts does not break the Hazard tab.
  const [geometryMeta, setGeometryMeta] = useState<GeometryMeta | null>(null);
  const [selectedCell, setSelectedCell] = useState<number | null>(null);
  const [cellSeries, setCellSeries] = useState<CellSeries | null>(null);
  const [cellSeriesLoading, setCellSeriesLoading] = useState(false);
  const [cellSeriesError, setCellSeriesError] = useState<string | null>(null);

  // Uncertainty tab state. Fetched once per run when the artifacts land; both
  // endpoints are silent on failure so the rest of the page works for older
  // completed runs.
  const [spreadSummary, setSpreadSummary] = useState<SpreadDecompositionSummary | null>(null);
  const [reliability, setReliability] = useState<ReliabilityCurves | null>(null);
  const [spreadHistogram, setSpreadHistogram] = useState<SpreadToPeakHistogram | null>(null);
  const [leaderboard, setLeaderboard] = useState<CellContributionLeaderboard | null>(null);
  const [availableRuns, setAvailableRuns] = useState<RunData[]>([]);
  const [compareRunId, setCompareRunId] = useState("");
  const [comparePayload, setComparePayload] = useState<ComparePayload | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareTimeSlot, setCompareTimeSlot] = useState(0);
  const [monitoring, setMonitoring] = useState<MonitoringPayload | null>(null);
  const [initialConditionSelection, setInitialConditionSelection] = useState<InitialConditionSelection | null>(null);

  const transient404Ref = useRef(0);

  const artifactUrl = useCallback(
    (artifactId: string) =>
      `/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}`,
    [runId],
  );

  const loadJson = useCallback(
    async (artifactId: string) => {
      const res = await fetch(artifactUrl(artifactId), { cache: "no-store" });
      if (!res.ok) return null;
      try {
        return (await res.json()) as Summary;
      } catch {
        return null;
      }
    },
    [artifactUrl],
  );

  const refresh = useCallback(async () => {
    const runRes = await fetch(`/api/runs/${encodeURIComponent(runId)}`, { cache: "no-store" });
    if (runRes.status === 404) {
      transient404Ref.current += 1;
      if (transient404Ref.current >= TRANSIENT_404_TOLERATED) {
        setError("Run not found. It may have been deleted or the URL is wrong.");
      }
      return;
    }
    if (!runRes.ok) {
      const detail = await runRes.json().catch(() => ({}));
      setError(detail.detail ?? "Could not load run.");
      return;
    }
    transient404Ref.current = 0;
    setError("");
    setRun(await runRes.json());
    const artifactRes = await fetch(`/api/runs/${encodeURIComponent(runId)}/artifacts`, {
      cache: "no-store",
    });
    if (artifactRes.ok) {
      const list = (await artifactRes.json()) as Artifact[];
      setArtifacts(list);
      const ids = new Set(list.map((a) => a.artifact_id));
      if (ids.has("calibrated_summary.json")) {
        loadJson("calibrated_summary.json").then(setCalibratedSummary).catch(() => undefined);
      }
      if (ids.has("raw_summary.json")) {
        loadJson("raw_summary.json").then(setRawSummary).catch(() => undefined);
      }
      if (ids.has("initial_condition_selection.json")) {
        loadJson("initial_condition_selection.json")
          .then((payload) => setInitialConditionSelection(payload as InitialConditionSelection | null))
          .catch(() => undefined);
      }
      if (ids.has("forcing.csv") && forcing === null) {
        fetch(artifactUrl("forcing.csv"), { cache: "no-store" })
          .then((res) => (res.ok ? res.text() : null))
          .then((text) => {
            if (text) setForcing(parseHydrograph(text));
          })
          .catch(() => undefined);
      }
    }
    fetch(`/api/runs/${encodeURIComponent(runId)}/monitoring`, { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => setMonitoring(payload as MonitoringPayload | null))
      .catch(() => setMonitoring(null));
  }, [runId, loadJson, forcing, artifactUrl]);

  const cancelRun = useCallback(async () => {
    if (!run || actionBusy) return;
    setActionBusy(true);
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(run.run_id)}/cancel`, { method: "POST" });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail ?? "Could not cancel run.");
      }
      await refresh();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setActionBusy(false);
    }
  }, [actionBusy, refresh, run]);

  const deleteRun = useCallback(async () => {
    if (!run || actionBusy) return;
    setActionBusy(true);
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(run.run_id)}`, { method: "DELETE" });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail ?? "Could not delete run.");
      }
      window.location.href = "/";
    } catch (exc) {
      setError(String(exc));
      setActionBusy(false);
    }
  }, [actionBusy, run]);

  const togglePin = useCallback(async () => {
    if (!run || !me?.is_admin || actionBusy) return;
    setActionBusy(true);
    try {
      const action = run.pinned ? "unpin" : "pin";
      const res = await fetch(`/api/admin/runs/${encodeURIComponent(run.run_id)}/${action}`, { method: "POST" });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail ?? `Could not ${action} run.`);
      }
      await refresh();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setActionBusy(false);
    }
  }, [actionBusy, me?.is_admin, refresh, run]);

  useEffect(() => {
    refresh().catch((exc) => setError(String(exc)));
    const timer = setInterval(() => refresh().catch(() => undefined), 10000);
    return () => clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    fetch("/api/me", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => setMe(payload as CurrentUser | null))
      .catch(() => setMe(null));
  }, []);

  // Fetch geometry sidecar once per run. Older runs without it leave the
  // inspector disabled but the rest of the Hazard tab works.
  useEffect(() => {
    if (geometryMeta) return;
    const hasArtifact = artifacts.some((a) => a.artifact_id === "geometry_meta.json");
    if (!hasArtifact) return;
    fetch(artifactUrl("geometry_meta.json"), { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setGeometryMeta(data as GeometryMeta);
      })
      .catch(() => undefined);
  }, [artifacts, geometryMeta, artifactUrl]);

  // Fetch small JSON diagnostics the first time each artifact appears.
  useEffect(() => {
    if (!spreadSummary) {
      if (artifacts.some((a) => a.artifact_id === "spread_decomposition_summary.json")) {
        fetch(artifactUrl("spread_decomposition_summary.json"), { cache: "no-store" })
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (data) setSpreadSummary(data as SpreadDecompositionSummary);
          })
          .catch(() => undefined);
      }
    }
    if (!reliability) {
      if (artifacts.some((a) => a.artifact_id === "reliability_curves.json")) {
        fetch(artifactUrl("reliability_curves.json"), { cache: "no-store" })
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (data) setReliability(data as ReliabilityCurves);
          })
          .catch(() => undefined);
      }
    }
    if (!spreadHistogram) {
      if (artifacts.some((a) => a.artifact_id === "spread_to_peak_histogram.json")) {
        fetch(artifactUrl("spread_to_peak_histogram.json"), { cache: "no-store" })
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (data) setSpreadHistogram(data as SpreadToPeakHistogram);
          })
          .catch(() => undefined);
      }
    }
    if (!leaderboard) {
      if (artifacts.some((a) => a.artifact_id === "cell_contribution_leaderboard.json")) {
        fetch(artifactUrl("cell_contribution_leaderboard.json"), { cache: "no-store" })
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (data) setLeaderboard(data as CellContributionLeaderboard);
          })
          .catch(() => undefined);
      }
    }
  }, [artifacts, spreadSummary, reliability, spreadHistogram, leaderboard, artifactUrl]);

  useEffect(() => {
    fetch("/api/runs", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        if (Array.isArray(data)) setAvailableRuns(data as RunData[]);
      })
      .catch(() => undefined);
  }, []);

  // Fetch the picked cell's full ensemble trace + derived stats.
  const fetchCellSeries = useCallback(
    async (cellIndex: number) => {
      setCellSeriesLoading(true);
      setCellSeriesError(null);
      try {
        const res = await fetch(
          `/api/runs/${encodeURIComponent(runId)}/cell/${encodeURIComponent(String(cellIndex))}/timeseries`,
          { cache: "no-store" },
        );
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail ?? `Cell endpoint returned HTTP ${res.status}`);
        }
        const payload = (await res.json()) as CellSeries;
        setCellSeries(payload);
      } catch (exc) {
        setCellSeries(null);
        setCellSeriesError(exc instanceof Error ? exc.message : String(exc));
      } finally {
        setCellSeriesLoading(false);
      }
    },
    [runId],
  );

  useEffect(() => {
    if (selectedCell === null) {
      setCellSeries(null);
      return;
    }
    fetchCellSeries(selectedCell).catch(() => undefined);
  }, [selectedCell, fetchCellSeries]);

  useEffect(() => {
    if (!compareRunId) {
      setComparePayload(null);
      setCompareError(null);
      return;
    }
    setCompareError(null);
    fetch(`/api/runs/${encodeURIComponent(runId)}/compare/${encodeURIComponent(compareRunId)}`, {
      cache: "no-store",
    })
      .then(async (res) => {
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.detail ?? `Compare endpoint returned HTTP ${res.status}`);
        setComparePayload(body as ComparePayload);
        setCompareTimeSlot(0);
      })
      .catch((exc) => {
        setComparePayload(null);
        setCompareError(exc instanceof Error ? exc.message : String(exc));
      });
  }, [runId, compareRunId]);

  const summary = view === "calibrated" ? calibratedSummary : rawSummary;
  const otherSummary = view === "calibrated" ? rawSummary : calibratedSummary;

  const pngSlots = useMemo(() => findMatchingPng(artifacts, view, product), [artifacts, view, product]);
  const safeTimeSlot = pngSlots.length === 0 ? 0 : Math.min(timeSlot, pngSlots.length - 1);
  const currentPng = pngSlots[safeTimeSlot];
  const currentLeadHours = useMemo(() => {
    if (!summary?.lead_time_hours || !currentPng) return null;
    const idx = currentPng.tIndex - 1;
    return summary.lead_time_hours[idx] ?? null;
  }, [summary, currentPng]);
  const completedCompareRuns = availableRuns.filter(
    (candidate) => candidate.run_id !== runId && candidate.status === "COMPLETED",
  );
  const leadHours = calibratedSummary?.lead_time_hours ?? rawSummary?.lead_time_hours ?? [];
  const responseDepthSeries = leadHours.map((t, idx) => ({
    t,
    y: Number(calibratedSummary?.peak_mean_wd_by_time_m?.[idx] ?? 0),
  }));
  const floodedAreaByTime = numericArray(calibratedSummary, "expected_flooded_area_fraction_wettable_by_time_gt_0.05m");
  const floodedAreaSeries = leadHours.map((t, idx) => ({
    t,
    y: 100 * Number(floodedAreaByTime[idx] ?? 0),
  }));
  // Trim the leading "warm-up" portion of forcing where stage is constant
  // and rainfall is zero. Coastal scenarios pad their CSV with a steady-state
  // window before any hydrodynamic event, and showing that flat lead-in
  // wastes plot real estate and visually buries the actual driver dynamics.
  // Detection: walk forward from the first row, and start plotting once
  // either stage moves materially off its baseline or precipitation rises
  // above noise. One pre-event row is kept as visual context for the
  // baseline level. If no dynamic event is detected the original series is
  // preserved (better than blanking the chart).
  const trimmedForcing = (() => {
    if (!forcing || forcing.t_h.length < 2) return forcing;
    const stageBaseline = forcing.stage[0];
    // 1 cm or 0.5% of baseline magnitude, whichever is larger; keeps the
    // detector robust to floating-point noise in CSVs while still firing
    // on real-world tide rises of a few cm.
    const stageThreshold = Math.max(0.01, Math.abs(stageBaseline) * 0.005);
    const precipThreshold = 1e-3; // ~zero rain in mm
    let firstDynamic = forcing.t_h.length;
    for (let i = 0; i < forcing.t_h.length; i += 1) {
      const stageMoved = Math.abs(forcing.stage[i] - stageBaseline) > stageThreshold;
      const rainStarted = forcing.precip[i] > precipThreshold;
      if (stageMoved || rainStarted) {
        firstDynamic = i;
        break;
      }
    }
    if (firstDynamic >= forcing.t_h.length) return forcing; // never moved → keep all
    const cut = Math.max(0, firstDynamic - 1); // include one baseline point
    return {
      t_h: forcing.t_h.slice(cut),
      stage: forcing.stage.slice(cut),
      precip: forcing.precip.slice(cut),
    };
  })();
  const stageSeries = trimmedForcing?.t_h.map((t, idx) => ({ t, y: trimmedForcing.stage[idx] })) ?? [];
  const rainSeries = trimmedForcing?.t_h.map((t, idx) => ({ t, y: trimmedForcing.precip[idx] })) ?? [];
  const cumulative = (points: { t: number; y: number }[]) => {
    let total = 0;
    return points.map((point, idx) => {
      const dt = idx === 0 ? 0 : Math.max(0, point.t - points[idx - 1].t);
      total += point.y * dt;
      return { t: point.t, y: total };
    });
  };
  const compareProduct: "mean" | "spread" | "p_gt_0p30m" =
    product === "spread" ? "spread" : product === "p_gt_0p30m" ? "p_gt_0p30m" : "mean";
  const compareAFrames = comparePayload
    ? findMatchingPng(comparePayload.artifacts_a, "calibrated", compareProduct)
    : [];
  const compareBFrames = comparePayload
    ? findMatchingPng(comparePayload.artifacts_b, "calibrated", compareProduct)
    : [];
  const compareDeltaFrames = comparePayload?.delta_frames?.[compareProduct] ?? [];
  const compareMaxSlot = Math.max(0, Math.min(compareAFrames.length, compareBFrames.length, compareDeltaFrames.length) - 1);
  const compareSafeSlot = Math.min(compareTimeSlot, compareMaxSlot);

  const exceedance = exceedanceEntries(summary);
  const otherExceedance = exceedanceEntries(otherSummary);
  const exceedanceArea = exceedanceAreaEntries(summary);
  const primaryExceedance = exceedanceArea.find((entry) => Math.abs(entry.threshold - 0.30) < 1e-6) ?? exceedanceArea[exceedanceArea.length - 1] ?? null;
  const hasAnimation = artifacts.some((a) => a.artifact_id.endsWith(".gif"));
  const hasHdf5 = artifacts.some((a) => a.artifact_id === "forecast_members.h5");
  const figureArtifacts = [
    // forcing_hydrograph.svg deliberately omitted — superseded by the
    // Drivers tab's stage + rainfall + response stacked chart.
    {
      id: "uq_extent_by_time.svg",
      title: "Expected Flooded-Area Fraction",
      caption: "Expected flooded-area fraction through lead time for each configured threshold.",
      insight: "Use this to separate timing-driven risk from depth-driven risk: curves that peak late or stay broad indicate persistent spatial exposure.",
    },
    {
      id: "uq_exceedance_bars.svg",
      title: "Peak Exceedance Footprint",
      caption: "Peak expected wettable-domain footprint at each water-depth threshold.",
      insight: "A steep drop between thresholds points to broad shallow flooding; persistent footprint at high thresholds points to deeper inundation.",
    },
    {
      id: "uq_uncertainty_width.svg",
      title: "Uncertainty Width",
      caption: "Area-weighted ensemble interval width over time.",
      insight: "Peaks mark lead times where the 60-member ensemble disagrees most, independent of any ground-truth observation.",
    },
    {
      id: "calibration_effect.svg",
      title: "Calibration Shift",
      caption: "Raw-to-calibrated changes in exceedance footprint and uncertainty summaries.",
      insight: "Large shifts show where calibration materially changed the UQ story; they are adjustment diagnostics, not validation accuracy.",
    },
  ].filter((figure) => artifacts.some((artifact) => artifact.artifact_id === figure.id));

  // Envelope tiles for the Hazard tab. Each entry is a temporal-summary map
  // written by the worker (collapses time into one frame); we just have to
  // show it if the artifact lands. The ``0p05`` / ``0p3`` suffixes come from
  // ForecastProductBuilder.write_envelope_maps thresholds.
  const envelopeTiles = [
    {
      id: "peak_depth_map.png",
      title: "Peak depth",
      caption: "Per-cell maximum of the ensemble-mean depth.",
      insight: "This map collapses time to each cell's maximum response, so it is best for locating peak hazard rather than judging event timing.",
    },
    {
      id: "arrival_time_map_gt_0p05m.png",
      title: "Arrival time @ 0.05 m",
      caption: "Lead hour at which each cell first exceeds 5 cm.",
      insight: "Use arrival time to see propagation of the wetting front. Cells that never exceed the threshold are masked.",
    },
    {
      id: "duration_map_gt_0p05m.png",
      title: "Wet duration @ 0.05 m",
      caption: "Total hours each cell stays above 5 cm.",
      insight: "Duration highlights persistence, which can differ from peak-depth hot spots when water arrives early or drains slowly.",
    },
    {
      id: "quantile_envelope_at_peak.png",
      title: "Spread at peak",
      caption: "p95 − p05 envelope width at each cell's peak time.",
      insight: "This is a spatial uncertainty-width map: large values indicate cells where ensemble members disagree on depth near peak response.",
    },
  ].filter((tile) => artifacts.some((artifact) => artifact.artifact_id === tile.id));

  // Uncertainty tab tiles. Each maps to a PNG written alongside diagnostic
  // .npy/.npz arrays. Filter on artifact presence so older runs gracefully show
  // only what they have.
  const uncertaintyTiles = [
    {
      id: "empirical_crps_map.png",
      title: "Empirical CRPS (peak time)",
      caption: "Mean pairwise absolute difference across the 60 members. No ground truth needed — a self-consistent measure of where the ensemble disagrees most.",
      insight: "Treat this as a ground-truth-free spread proxy. It identifies model disagreement, not measured forecast error.",
    },
    {
      id: "between_var_map.png",
      title: "Between-checkpoint variance",
      caption: "Variance across the three checkpoint means. Reflects structural / epistemic disagreement between models.",
      insight: "High values mean the three trained checkpoints disagree with each other, which is the model-structure part of the ensemble uncertainty.",
    },
    {
      id: "within_var_map.png",
      title: "Within-checkpoint variance",
      caption: "Mean of per-checkpoint variances. Reflects latent / aleatoric variability inside each model.",
      insight: "High values mean latent samples inside the same checkpoint spread out, even when checkpoint means are similar.",
    },
  ].filter((tile) => artifacts.some((artifact) => artifact.artifact_id === tile.id));

  return (
    <AppShell active="runs" userEmail={me?.email}>
    <div className="shell run-console">
      <a className="back" href="/">
        <ArrowLeft size={15} aria-hidden="true" /> Back to workspace
      </a>
      <RunHeader
        title={`Run ${run?.label || runId.slice(0, 10)}`}
        status={run?.status}
        createdAt={run?.created_at}
        pinned={run?.pinned}
        cache={run?.cache}
        actions={
          run ? (
            <div className="hero-actions" aria-label="Run actions">
              {!TERMINAL_STATUSES.has(run.status) && (
                <button type="button" className="action-button danger" onClick={cancelRun} disabled={actionBusy}>
                  <XCircle size={15} aria-hidden="true" /> Cancel
                </button>
              )}
              {TERMINAL_STATUSES.has(run.status) && (
                <button type="button" className="action-button danger" onClick={deleteRun} disabled={actionBusy}>
                  <Trash2 size={15} aria-hidden="true" /> Delete from history
                </button>
              )}
              {me?.is_admin && (
                <button type="button" className="action-button" onClick={togglePin} disabled={actionBusy}>
                  <Pin size={15} aria-hidden="true" /> {run.pinned ? "Unpin" : "Pin"}
                </button>
              )}
            </div>
          ) : null
        }
      />
      {run && (() => {
        const progressPct = Math.round(Math.max(0, Math.min(1, run.progress ?? 0)) * 100);
        return (
          <section className="panel run-progress-panel" aria-label="Run progress and runtime">
            <RunProgress status={run.status} progress={progressPct} detail={run.progress_label || run.status} />
            <div className="runtime-strip">
              <div>
                <span>Elapsed</span>
                <strong>{formatDuration(run.timing?.elapsed_seconds)}</strong>
              </div>
              <div>
                <span>{TERMINAL_STATUSES.has(run.status) ? "Runtime" : "ETA"}</span>
                <strong>
                  {TERMINAL_STATUSES.has(run.status)
                    ? formatDuration(run.timing?.runtime_seconds ?? run.runtime_seconds)
                    : formatDuration(run.timing?.estimated_remaining_seconds)}
                </strong>
              </div>
              <div>
                <span>Average full rollout</span>
                <strong>{formatDuration(run.timing?.average_full_rollout_seconds)}</strong>
                <small>
                  {run.timing?.average_full_rollout_sample_size
                    ? `${run.timing.average_full_rollout_sample_size} completed full rollout(s)`
                    : "Recorded after one full 60-member rollout completes"}
                </small>
              </div>
            </div>
          </section>
        );
      })()}
      {initialConditionSelection && (
        <section className="panel" aria-label="Initial water-depth history">
          <header className="envelope-head">
            <div>
              <p className="eyebrow">Initial condition</p>
              <h2>{formatInitialConditionMode(initialConditionSelection.mode)}</h2>
            </div>
            <span className="envelope-sub">
              {initialConditionSelection.reference_scope || "diagnostic"}
            </span>
          </header>
          <p className="tab-intro" style={{ margin: 0 }}>
            {initialConditionSelection.mode === "forcing_conditioned_baseline"
              ? `Initial water-depth history selected from ${initialConditionSelection.selected_reference_ids?.length ?? 0} similar train/calibration reference event(s).`
              : "Dry zero water-depth history was used for this diagnostic or compatibility run."}
            {initialConditionSelection.low_confidence
              ? " The uploaded forcing is far from the reference library, so the selection is marked low confidence."
              : ""}
          </p>
          {initialConditionSelection.selected_reference_ids?.length ? (
            <p className="muted" style={{ marginBottom: 0 }}>
              References: {initialConditionSelection.selected_reference_ids.join(", ")}
            </p>
          ) : null}
        </section>
      )}
      {monitoring?.available && (
        <section className="panel monitoring-panel" aria-label="Model monitoring">
          <header className="envelope-head">
            <div>
              <p className="eyebrow">Model monitoring</p>
              <h2>Reference screening and retraining review</h2>
            </div>
            <span className="envelope-sub">{monitoring.monitoring_bundle_id}</span>
          </header>
          <div className="monitoring-grid">
            {([monitoring.pre_run, monitoring.post_run] as (MonitoringReport | null | undefined)[]).map((report) => {
              if (!report) return null;
              const referenceScore =
                report.phase === "PRE_RUN"
                  ? report.scores?.input_novelty_score
                  : report.scores?.forecast_monitoring_score;
              const decisionScore = report.scores?.candidate_decision_score ?? report.scores?.candidate_score ?? 0;
              const decisionFlags = (report.flags ?? []).filter(isDecisionEligibleFlag);
              const diagnosticFlags = (report.flags ?? []).filter(isReferenceDiagnosticFlag);
              return (
                <article key={report.phase} className="monitoring-card">
                  <span>{report.phase === "PRE_RUN" ? "Input scenario" : "Completed forecast"}</span>
                  <strong>{typeof decisionScore === "number" ? decisionScore.toFixed(2) : "—"}</strong>
                  <small>
                    {report.candidate_recommended
                      ? `Selected for review: ${formatCandidateReason(report.candidate_reason)}`
                      : "Not selected for retraining review"}
                  </small>
                  <small>
                    Reference diagnostic score {typeof referenceScore === "number" ? referenceScore.toFixed(2) : "—"}
                  </small>
                  {decisionFlags.slice(0, 2).map((flag) => (
                    <small key={`${report.phase}-decision-${flag.code}`} style={{ color: "#0f766e", fontWeight: 700 }}>
                      {formatMonitoringFlag(flag)}
                    </small>
                  ))}
                  {diagnosticFlags.length > 0 && (
                    <small style={{ color: "#64748b", fontWeight: 700 }}>
                      Reference diagnostics
                    </small>
                  )}
                  {diagnosticFlags.slice(0, 3).map((flag) => (
                    <small key={`${report.phase}-diagnostic-${flag.code}-${flag.descriptor ?? "unknown"}`}>
                      {formatMonitoringFlag(flag)}
                    </small>
                  ))}
                </article>
              );
            })}
            <article className="monitoring-card">
              <span>Retraining review status</span>
              <strong>{formatCandidateStatus(monitoring.candidate?.status)}</strong>
              <small>
                {monitoring.candidate
                  ? `Selection score ${monitoring.candidate.candidate_score.toFixed(2)} · ${formatCandidateReason(monitoring.candidate.reason)}`
                  : "This run has not been preserved for retraining review."}
              </small>
            </article>
          </div>
          <p className="muted">
            Monitoring is a research guardrail for future HEC-RAS labeling and retraining review. It is not proof of model error or decision guidance.
          </p>
        </section>
      )}
      {error && <p className="error">{error}</p>}
      {run?.failure_reason && (
        <section className="panel error-panel">
          <h2>Run failed</h2>
          <p>{run.failure_reason}</p>
        </section>
      )}

      <RunDetailTabs active={tab} onChange={setTab} />
      <p className="tab-intro">
        Results are organized into three questions — <strong>Hazard</strong> (what / where / when),
        {" "}<strong>Uncertainty</strong> (how confident, and where uncertainty concentrates), and
        {" "}<strong>Drivers</strong> (forcing → response). Click any cell on the map to read its
        ensemble depth fan, calibrated probability of exceedance, and arrival-time distribution
        at that point.
      </p>

      {tab === "uncertainty" && (
        <UncertaintyTab>
          <section className="panel" aria-label="Uncertainty headline">
            <header className="envelope-head">
              <h2>Uncertainty</h2>
              <span className="envelope-sub">
                Calibration {reliability?.applied ? "applied" : "no-op"}
                {spreadSummary && ` · ${spreadSummary.n_groups} checkpoints`}
              </span>
            </header>
            <p className="tab-intro" style={{ margin: 0 }}>
              How confident is this forecast, and <em>where</em> is the uncertainty
              concentrated? Maps below show the intra-ensemble spread metric and how
              total variance splits between checkpoint disagreement and within-
              checkpoint (latent) variability. The reliability card shows what
              isotonic calibration actually did to exceedance probabilities on this
              run.
            </p>
            {spreadSummary && (
              <div className="share-strip" aria-label="Between- vs within-checkpoint share">
                <div className="share-bar">
                  <span
                    className="share-bar-between"
                    style={{ width: `${(spreadSummary.between_share * 100).toFixed(1)}%` }}
                  />
                  <span
                    className="share-bar-within"
                    style={{ width: `${(spreadSummary.within_share * 100).toFixed(1)}%` }}
                  />
                </div>
                <div className="share-legend">
                  <span><i className="dot dot-between" /> between checkpoints {(spreadSummary.between_share * 100).toFixed(0)}%</span>
                  <span><i className="dot dot-within" /> within checkpoints {(spreadSummary.within_share * 100).toFixed(0)}%</span>
                </div>
              </div>
            )}
          </section>

          {uncertaintyTiles.length > 0 && (
            <section className="panel envelope-panel" aria-label="Uncertainty maps">
              <header className="envelope-head">
                <h2>Uncertainty maps</h2>
                <span className="envelope-sub">{uncertaintyTiles.length} maps · evaluated at peak time</span>
              </header>
              <div className="envelope-grid">
                {uncertaintyTiles.map((tile) => (
                  <figure key={tile.id} className="envelope-tile">
                    <img src={artifactUrl(tile.id)} alt={`${tile.title} — ${tile.caption}`} />
                    <FigureCaption title={tile.title} caption={tile.caption} insight={tile.insight} />
                  </figure>
                ))}
              </div>
            </section>
          )}

          {reliability && (
            <section className="panel" aria-label="Reliability diagram">
              <header className="envelope-head">
                <h2>Reliability</h2>
                <span className="envelope-sub">
                  raw vs isotonic-calibrated P(WD &gt; threshold), {reliability.n_wettable_cells} wettable cells at peak
                </span>
              </header>
              <ReliabilityCurveCard data={reliability} />
              <FigureNote
                caption="Isotonic reliability curves compare raw ensemble probabilities with calibrated probabilities over wettable cells at peak time."
                insight="Use the curve shape to understand probability adjustment. A curve above identity increases exceedance probabilities; below identity decreases them."
              />
              {!reliability.applied && (
                <p className="muted" style={{ marginTop: 8 }}>
                  Calibration was a no-op for this run (no isotonic curves). The
                  curve collapses to the identity reference line.
                </p>
              )}
              <div className="reliability-summary">
                {Object.values(reliability.curves)
                  .sort((a, b) => a.threshold_m - b.threshold_m)
                  .map((curve) => {
                    const delta = curve.calibrated_mean - curve.raw_mean;
                    return (
                      <div key={curve.threshold_m} className="reliability-row">
                        <strong>&gt; {curve.threshold_m.toFixed(2)} m</strong>
                        <span>raw {(curve.raw_mean * 100).toFixed(1)}%</span>
                        <span>cal {(curve.calibrated_mean * 100).toFixed(1)}%</span>
                        <span className={delta > 0.005 ? "delta-up" : delta < -0.005 ? "delta-down" : "delta-flat"}>
                          {delta >= 0 ? "+" : ""}{(delta * 100).toFixed(1)} pp
                        </span>
                      </div>
                    );
                  })}
              </div>
            </section>
          )}

          {(spreadHistogram || artifacts.some((a) => a.artifact_id === "spread_to_peak_histogram.svg")) && (
            <section className="panel" aria-label="Spread-to-peak histogram">
              <header className="envelope-head">
                <h2>Spread-to-peak distribution</h2>
                <span className="envelope-sub">
                  relative ensemble disagreement across wettable cells
                </span>
              </header>
              {artifacts.some((a) => a.artifact_id === "spread_to_peak_histogram.svg") && (
                <img
                  className="figure-wide"
                  src={artifactUrl("spread_to_peak_histogram.svg")}
                  alt="Histogram of spread divided by local peak depth"
                />
              )}
              {spreadHistogram && (
                <div className="reliability-summary">
                  <div className="reliability-row">
                    <strong>signal cells</strong>
                    <span>{spreadHistogram.n_signal_cells} / {spreadHistogram.n_wettable_cells}</span>
                  </div>
                  <div className="reliability-row">
                    <strong>median ratio</strong>
                    <span>{formatNumber(spreadHistogram.median_ratio, 3)}</span>
                  </div>
                  <div className="reliability-row">
                    <strong>p95 ratio</strong>
                    <span>{formatNumber(spreadHistogram.p95_ratio, 3)}</span>
                  </div>
                </div>
              )}
            </section>
          )}

          {uncertaintyTiles.length === 0 && !reliability && (
            <section className="panel tab-stub">
              <h2>Uncertainty</h2>
              <p>
                Diagnostics for this tab haven&apos;t been written yet (run is still
                postprocessing, or this run was created before these diagnostics were added). Submit a fresh
                scenario to see the empirical-CRPS map, the between/within
                variance split, and the calibration reliability diagram.
              </p>
            </section>
          )}
        </UncertaintyTab>
      )}

      {tab === "drivers" && (
        <DriversTab>
          <section className="panel">
            <header className="envelope-head">
              <h2>Drivers</h2>
              <span className="envelope-sub">forcing and response on one lead-time axis</span>
            </header>
            <div className="driver-stack">
              <figure className="chart-figure">
                <MultiLineChart series={[{ label: "Stage", color: "#0f766e", points: stageSeries }]} yLabel="Coastal stage" />
                <FigureCaption
                  title="Coastal stage"
                  caption="Uploaded boundary stage after removing the flat warm-up lead-in."
                  insight="This is the dominant coastal forcing signal; compare peaks and rise rates against the response panels below."
                />
              </figure>
              <figure className="chart-figure">
                <BarSeriesChart points={rainSeries} color="#1d4ed8" yLabel="Precipitation" />
                <FigureCaption
                  title="Precipitation"
                  caption="Uploaded rainfall intensity by lead time."
                  insight="Bars show timing and concentration of rainfall forcing. Compact high bars imply intense pulses; broad bars imply sustained rainfall."
                />
              </figure>
              <figure className="chart-figure">
                <MultiLineChart series={[{ label: "Peak depth", color: "#b45309", points: responseDepthSeries }]} yLabel="Peak-depth response (m)" />
                <FigureCaption
                  title="Peak-depth response"
                  caption="Maximum calibrated ensemble-mean depth across the domain at each lead time."
                  insight="This links forcing to peak water-depth response. It is depth focused, not an area metric."
                />
              </figure>
              <figure className="chart-figure">
                <MultiLineChart series={[{ label: "Flooded area", color: "#7c3aed", points: floodedAreaSeries }]} yLabel="Expected wettable flooded-area fraction (%)" />
                <FigureCaption
                  title="Expected flooded-area response"
                  caption="Expected wettable-domain area fraction above 0.05 m through lead time."
                  insight="This converts cell-level exceedance into physical extent, making it easier to compare scenarios with different spatial footprints."
                />
              </figure>
            </div>
          </section>

          <section className="cards">
            <article className="card">
              <span className="card-label">Cumulative stage signal</span>
              <span className="card-value">{formatNumber(cumulative(stageSeries).at(-1)?.y, 2)}</span>
              <span className="card-note">
                stage-hours over uploaded forcing
                <InfoTip text="Integrated stage signal summarizes how long and how strongly the coastal boundary stayed elevated." />
              </span>
            </article>
            <article className="card">
              <span className="card-label">Cumulative rainfall</span>
              <span className="card-value">{formatNumber(cumulative(rainSeries).at(-1)?.y, 2)}</span>
              <span className="card-note">
                precipitation-hours over uploaded forcing
                <InfoTip text="Integrated rainfall summarizes event volume on the uploaded time grid; it complements the peak-intensity bar chart." />
              </span>
            </article>
            <article className="card">
              <span className="card-label">Cumulative flooded-area response</span>
              <span className="card-value">{formatNumber(cumulative(floodedAreaSeries).at(-1)?.y, 2)}</span>
              <span className="card-note">
                percent-wettable-hours at &gt; 0.05 m
                <InfoTip text="This integrates spatial extent through time, so short intense flooding and long persistent flooding can be distinguished." />
              </span>
            </article>
          </section>

          <section className="panel">
            <header className="envelope-head">
              <h2>Cell contribution leaderboard</h2>
              <span className="envelope-sub">
                ranked by expected area contribution to P(WD &gt; {(leaderboard?.threshold_m ?? 0.30).toFixed(2)} m)
              </span>
            </header>
            {!leaderboard && <p className="muted">Leaderboard artifact is generated during postprocessing for new completed runs.</p>}
            {leaderboard && (
              <>
              <div className="leaderboard-table">
                <table className="summary-table">
                  <thead>
                    <tr>
                      <th>Cell</th>
                      <th>Peak P</th>
                      <th>Peak depth</th>
                      <th>Lead</th>
                      <th>Share</th>
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboard.rows.slice(0, 20).map((row) => (
                      <tr
                        key={row.cell_index}
                        className="clickable-row"
                        onClick={() => {
                          setSelectedCell(row.cell_index);
                          setTab("hazard");
                        }}
                      >
                        <td>
                          #{row.cell_index}
                          {row.x !== undefined && row.y !== undefined && (
                            <span className="table-sub">E {Math.round(row.x).toLocaleString()}, N {Math.round(row.y).toLocaleString()}</span>
                          )}
                        </td>
                        <td>{formatPercent(row.peak_probability)}</td>
                        <td>{formatNumber(row.peak_mean_wd_m, 2)} m</td>
                        <td>{row.peak_lead_hours === null ? "—" : `${formatNumber(row.peak_lead_hours, 2)} h`}</td>
                        <td>{formatPercent(row.contribution_share, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <FigureNote
                caption="Rows rank cells by expected contribution to the configured exceedance footprint."
                insight="Clicking a row selects that cell in the Hazard inspector so you can connect aggregate extent to a local 60-member trace."
              />
              </>
            )}
          </section>
        </DriversTab>
      )}

      {tab === "compare" && (
        <CompareTab>
          <section className="panel">
            <header className="envelope-head">
              <h2>Compare</h2>
              <span className="envelope-sub">A/B comparison against another completed run</span>
            </header>
            <div className="compare-picker">
              <div>
                <span className="card-label">Run A</span>
                <strong>{formatRunLabel(run)}</strong>
              </div>
              <label>
                <span className="card-label">Run B</span>
                <select value={compareRunId} onChange={(event) => setCompareRunId(event.target.value)}>
                  <option value="">Choose a completed run</option>
                  {completedCompareRuns.map((candidate) => (
                    <option key={candidate.run_id} value={candidate.run_id}>
                      {formatRunLabel(candidate)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {compareError && <p className="error">{compareError}</p>}
            {!compareRunId && <p className="muted">Choose a second completed run to generate synchronized maps and summary deltas.</p>}
          </section>

          {comparePayload && (
            <>
              <section className="panel">
                <header className="envelope-head">
                  <h2>Synchronized maps</h2>
                  <span className="envelope-sub">{PRODUCT_LABELS[compareProduct]} · B - A delta</span>
                </header>
                <div className="product-row">
                  {(["mean", "spread", "p_gt_0p30m"] as ProductKey[]).map((p) => (
                    <button
                      key={p}
                      type="button"
                      className={`chip ${compareProduct === p ? "active" : ""}`}
                      onClick={() => {
                        setProduct(p);
                        setCompareTimeSlot(0);
                      }}
                    >
                      {PRODUCT_LABELS[p]}
                    </button>
                  ))}
                </div>
                {compareMaxSlot > 0 && (
                  <div className="time-slider">
                    <input
                      type="range"
                      min={0}
                      max={compareMaxSlot}
                      value={compareSafeSlot}
                      step={1}
                      onChange={(event) => setCompareTimeSlot(Number(event.target.value))}
                    />
                    <span>Slot {compareSafeSlot + 1} / {compareMaxSlot + 1}</span>
                  </div>
                )}
                <div className="compare-map-grid">
                  {compareAFrames[compareSafeSlot] && (
                    <figure className="envelope-tile">
                      <img src={artifactUrlFor(runId, compareAFrames[compareSafeSlot].artifactId)} alt="Run A map" />
                      <FigureCaption
                        title="Run A"
                        caption={formatRunLabel(comparePayload.run_a)}
                        insight="Run A is the baseline scenario for the synchronized comparison. The same product and lead slot are used for both runs."
                      />
                    </figure>
                  )}
                  {compareBFrames[compareSafeSlot] && (
                    <figure className="envelope-tile">
                      <img src={artifactUrlFor(comparePayload.run_b.run_id, compareBFrames[compareSafeSlot].artifactId)} alt="Run B map" />
                      <FigureCaption
                        title="Run B"
                        caption={formatRunLabel(comparePayload.run_b)}
                        insight="Run B is compared against Run A with matched product and lead slot so differences are attributable to scenario changes."
                      />
                    </figure>
                  )}
                  {compareDeltaFrames[compareSafeSlot] && (
                    <figure className="envelope-tile">
                      <img src={artifactUrl(compareDeltaFrames[compareSafeSlot])} alt="B minus A delta map" />
                      <FigureCaption
                        title="Delta"
                        caption="Run B minus Run A at the synchronized lead slot."
                        insight="Positive values mean Run B is deeper, more uncertain, or has higher exceedance probability depending on the selected product."
                      />
                    </figure>
                  )}
                </div>
              </section>

              <section className="panel">
                <h2>Overlay hydrographs</h2>
                <MultiLineChart
                  yLabel="Expected flooded-area fraction (%)"
                  series={[
                    {
                      label: "Run A",
                      color: "#0f766e",
                      points: (comparePayload.summary_a.lead_time_hours ?? []).map((t, idx) => ({
                        t,
                        y: 100 * Number(numericArray(comparePayload.summary_a, "expected_flooded_area_fraction_wettable_by_time_gt_0.05m")[idx] ?? 0),
                      })),
                    },
                    {
                      label: "Run B",
                      color: "#b45309",
                      points: (comparePayload.summary_b.lead_time_hours ?? []).map((t, idx) => ({
                        t,
                        y: 100 * Number(numericArray(comparePayload.summary_b, "expected_flooded_area_fraction_wettable_by_time_gt_0.05m")[idx] ?? 0),
                      })),
                    },
                  ]}
                />
                <FigureNote
                  caption="Overlay of expected wettable flooded-area fraction for the two runs."
                  insight="This shows whether the scenario change alters spatial footprint, timing, or persistence rather than only pointwise map values."
                />
                <MultiLineChart
                  yLabel="Peak-depth response (m)"
                  series={[
                    {
                      label: "Run A",
                      color: "#0f766e",
                      points: (comparePayload.summary_a.lead_time_hours ?? []).map((t, idx) => ({
                        t,
                        y: Number(comparePayload.summary_a.peak_mean_wd_by_time_m?.[idx] ?? 0),
                      })),
                    },
                    {
                      label: "Run B",
                      color: "#b45309",
                      points: (comparePayload.summary_b.lead_time_hours ?? []).map((t, idx) => ({
                        t,
                        y: Number(comparePayload.summary_b.peak_mean_wd_by_time_m?.[idx] ?? 0),
                      })),
                    },
                  ]}
                />
                <FigureNote
                  caption="Overlay of domain peak-depth response for Run A and Run B."
                  insight="Use this with the flooded-area overlay: a scenario can increase peak depth without greatly changing spatial footprint, or the reverse."
                />
              </section>

              <section className="panel">
                <h2>Summary deltas</h2>
                <table className="summary-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>A</th>
                      <th>B</th>
                      <th>Delta</th>
                      <th>%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparePayload.summary_delta.map((row) => (
                      <tr key={row.key}>
                        <td>{row.label}</td>
                        <td>{formatNumber(row.a, 3)} {row.unit}</td>
                        <td>{formatNumber(row.b, 3)} {row.unit}</td>
                        <td>{row.delta >= 0 ? "+" : ""}{formatNumber(row.delta, 3)} {row.unit}</td>
                        <td>{row.percent_delta === null ? "—" : `${row.percent_delta >= 0 ? "+" : ""}${formatNumber(row.percent_delta, 1)}%`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <FigureNote
                  caption="A/B deltas are reported for physical area, probability, depth, timing, and uncertainty metrics."
                  insight="Raw cell counts are intentionally not primary metrics because they are less interpretable than area, fraction, and physical depth."
                />
              </section>
            </>
          )}
        </CompareTab>
      )}

      {tab === "hazard" && (
        <HazardTab>
      {envelopeTiles.length > 0 && (
        <section className="panel envelope-panel" aria-label="Hazard envelope maps">
          <header className="envelope-head">
            <h2>Forecast envelope</h2>
            <span className="envelope-sub">
              {envelopeTiles.length} temporal-summary maps from the calibrated ensemble
            </span>
          </header>
          <div className="envelope-grid">
            {envelopeTiles.map((tile) => (
              <figure key={tile.id} className="envelope-tile">
                <img src={artifactUrl(tile.id)} alt={`${tile.title} — ${tile.caption}`} />
                <FigureCaption title={tile.title} caption={tile.caption} insight={tile.insight} />
              </figure>
            ))}
          </div>
        </section>
      )}

      {summary && (
        <section className="cards">
          <article className="card">
            <span className="card-label">Peak expected flooded area</span>
            <span className="card-value">
              {formatNumber(summary["peak_expected_flooded_area_km2_gt_0.05m"], 3)} km²
            </span>
            <span className="card-note">
              {formatNumber(100 * Number(summary["peak_expected_flooded_area_fraction_wettable_gt_0.05m"] ?? 0), 1)}% wettable
            </span>
          </article>
          <article className="card">
            <span className="card-label">Primary exceedance footprint</span>
            <span className="card-value">
              {primaryExceedance ? `${formatNumber(primaryExceedance.peakFraction * 100, 1)}%` : "—"}
            </span>
            <span className="card-note">{primaryExceedance ? `P(WD > ${primaryExceedance.threshold.toFixed(2)} m)` : "No threshold summary"}</span>
          </article>
          <article className="card">
            <span className="card-label">Onset / peak timing</span>
            <span className="card-value">
              {summary["onset_lead_hours_expected_flooded_area_fraction_gt_1pct_gt_0.05m"] === null ||
              summary["onset_lead_hours_expected_flooded_area_fraction_gt_1pct_gt_0.05m"] === undefined
                ? "—"
                : `${formatNumber(summary["onset_lead_hours_expected_flooded_area_fraction_gt_1pct_gt_0.05m"], 2)} h`}
            </span>
            <span className="card-note">peak {formatNumber(summary["peak_expected_flooded_area_lead_hours_gt_0.05m"], 2)} h</span>
          </article>
          <article className="card">
            <span className="card-label">Area-weighted uncertainty width</span>
            <span className="card-value">{formatNumber(summary.peak_area_weighted_iqr_wd_m, 3)} m</span>
            <span className="card-note">IQR; p95-p05 {formatNumber(summary.peak_area_weighted_central_90_wd_m, 3)} m</span>
          </article>
          <article className="card">
            <span className="card-label">Checkpoint disagreement</span>
            <span className="card-value">
              {summary.peak_area_weighted_between_checkpoint_variance_share == null
                ? "—"
                : `${formatNumber(summary.peak_area_weighted_between_checkpoint_variance_share * 100, 1)}%`}
            </span>
            <span className="card-note">
              {summary.peak_area_weighted_between_checkpoint_spread_wd_m == null
                ? "available after new runs"
                : `${formatNumber(summary.peak_area_weighted_between_checkpoint_spread_wd_m, 3)} m between-checkpoint spread`}
            </span>
          </article>
        </section>
      )}

      <>
      <section className="hazard-map-layout">
      <section className="panel hazard-map-panel">
        <TimePlayer>
        <div className="row-between">
          <h2>Forecast maps</h2>
          <div className="toggle">
            <button
              type="button"
              className={view === "calibrated" ? "active" : ""}
              onClick={() => setView("calibrated")}
            >
              Calibrated
            </button>
            <button
              type="button"
              className={view === "raw" ? "active" : ""}
              onClick={() => setView("raw")}
            >
              Raw FGN
            </button>
          </div>
        </div>
        <div className="product-row">
          {(Object.keys(PRODUCT_LABELS) as ProductKey[]).map((p) => (
            <button
              key={p}
              type="button"
              className={`chip ${product === p ? "active" : ""}`}
              onClick={() => {
                setProduct(p);
                setTimeSlot(0);
              }}
            >
              {PRODUCT_LABELS[p]}
            </button>
          ))}
        </div>
        {currentPng ? (
          <>
            <MapWithCellPicker
              src={artifactUrl(currentPng.artifactId)}
              alt={`${view} ${product}${currentLeadHours !== null ? ` at lead ${currentLeadHours.toFixed(2)} h` : ""}`}
              geometry={geometryMeta}
              onPick={setSelectedCell}
              selectedCellIndex={selectedCell}
            />
            <FigureNote
              caption={PRODUCT_CAPTIONS[product].caption}
              insight={PRODUCT_CAPTIONS[product].insight}
            />
            {pngSlots.length > 1 && (
              <div className="forecast-scrubber">
                <div className="forecast-scrubber-controls" role="group" aria-label="Forecast step transport">
                  <button
                    type="button"
                    className="forecast-step-btn"
                    onClick={() => setTimeSlot(0)}
                    disabled={safeTimeSlot === 0}
                    aria-label="Jump to first forecast step"
                    title="First step"
                  >
                    ⏮
                  </button>
                  <button
                    type="button"
                    className="forecast-step-btn"
                    onClick={() => setTimeSlot(Math.max(0, safeTimeSlot - 1))}
                    disabled={safeTimeSlot === 0}
                    aria-label="Previous forecast step"
                    title="Previous step"
                  >
                    ◀
                  </button>
                  <input
                    type="range"
                    className="forecast-scrub-range"
                    min={0}
                    max={pngSlots.length - 1}
                    value={safeTimeSlot}
                    step={1}
                    onChange={(event) => setTimeSlot(Number(event.target.value))}
                    style={{
                      ["--scrub-pct" as string]: `${(safeTimeSlot / Math.max(pngSlots.length - 1, 1)) * 100}%`,
                    }}
                    aria-label="Scrub through forecast steps"
                    aria-valuenow={safeTimeSlot + 1}
                    aria-valuemin={1}
                    aria-valuemax={pngSlots.length}
                  />
                  <button
                    type="button"
                    className="forecast-step-btn"
                    onClick={() => setTimeSlot(Math.min(pngSlots.length - 1, safeTimeSlot + 1))}
                    disabled={safeTimeSlot >= pngSlots.length - 1}
                    aria-label="Next forecast step"
                    title="Next step"
                  >
                    ▶
                  </button>
                  <button
                    type="button"
                    className="forecast-step-btn"
                    onClick={() => setTimeSlot(pngSlots.length - 1)}
                    disabled={safeTimeSlot >= pngSlots.length - 1}
                    aria-label="Jump to last forecast step"
                    title="Last step"
                  >
                    ⏭
                  </button>
                </div>
                <div className="forecast-scrubber-meta">
                  <span className="forecast-step-tag">
                    {currentPng.isScrub ? "Step" : "Sample"} {safeTimeSlot + 1}
                    <span className="forecast-step-of"> / {pngSlots.length}</span>
                  </span>
                  <span className="forecast-lead">
                    {currentLeadHours !== null
                      ? <>lead <strong>+{currentLeadHours.toFixed(2)} h</strong></>
                      : <>frame {currentPng.tIndex}</>}
                  </span>
                  {!currentPng.isScrub && view === "raw" && (
                    <span className="forecast-step-hint" title="Raw view ships with sparse sample frames; switch to Calibrated for per-step scrubbing.">
                      raw view: sparse
                    </span>
                  )}
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="muted">
            Map products are generated after inference completes. Once the run is COMPLETED, snapshots
            for {PRODUCT_LABELS[product]} ({view}) will appear here.
          </p>
        )}
        {hasAnimation && view === "calibrated" && (
          <a
            className="anim-link"
            href={artifactUrl(artifacts.find((a) => a.artifact_id.startsWith("calibrated_p_gt_") && a.artifact_id.endsWith(".gif"))?.artifact_id ?? "calibrated_mean_wd_animation.gif")}
            target="_blank"
            rel="noreferrer"
          >
            Download calibrated animation (GIF)
          </a>
        )}
        </TimePlayer>
      </section>

      <CellInspector selected={selectedCell !== null}>
        {selectedCell === null ? (
          <div className="empty inspector-empty">
            <strong>Click a map cell to inspect local uncertainty.</strong>
            <span>The inspector will show the 60-member depth fan, local exceedance probabilities, and arrival-time distribution.</span>
          </div>
        ) : (
        <>
          <header className="poi-head">
            <div>
              <p className="poi-eyebrow">Per-cell inspector</p>
              <h2>
                Cell #{selectedCell}
                {geometryMeta && (
                  <span className="poi-coords">
                    {" "}· E {Math.round(geometryMeta.x[selectedCell] ?? 0).toLocaleString()},
                    N {Math.round(geometryMeta.y[selectedCell] ?? 0).toLocaleString()}
                  </span>
                )}
              </h2>
              {cellSeries && (
                <p className="poi-headline">
                  peak <strong>{cellSeries.peak_calibrated_mean_wd_m.toFixed(2)} m</strong>
                  {" "}at <strong>{cellSeries.peak_calibrated_lead_hours.toFixed(2)} h</strong>
                  {" "}· wettable {cellSeries.wettable ? "yes" : "no"}
                </p>
              )}
              {cellSeries && (
                <div className="poi-context">
                  <span>elevation {formatNumber(cellSeries.elevation_m ?? geometryMeta?.elevation_m?.[selectedCell] ?? null, 2)} m</span>
                  <span>slope {formatNumber(cellSeries.slope ?? geometryMeta?.slope?.[selectedCell] ?? null, 3)}</span>
                  <span>flow accumulation {formatNumber(cellSeries.flow_accumulation ?? geometryMeta?.flow_accumulation?.[selectedCell] ?? null, 1)}</span>
                </div>
              )}
            </div>
            <button
              type="button"
              className="poi-clear"
              onClick={() => setSelectedCell(null)}
              aria-label="Clear cell selection"
            >
              Clear
            </button>
          </header>
          {cellSeriesLoading && <p className="muted">Loading ensemble trace…</p>}
          {cellSeriesError && (
            <p className="error">Could not load cell timeseries: {cellSeriesError}</p>
          )}
          {cellSeries && (
            <div className="poi-grid">
              <figure className="chart-figure">
                <PoiHydrograph series={cellSeries} />
                <FigureCaption
                  title="Depth fan"
                  caption="All member traces with p05-p95 ribbon and p50 median at the selected cell."
                  insight="This is the local UQ story: member clustering, outliers, and spread through time are visible at one physical point."
                />
              </figure>
              <figure className="chart-figure">
                <PoiExceedanceTrace series={cellSeries} />
                <FigureCaption
                  title="Local exceedance probability"
                  caption="Calibrated probability of exceeding configured water-depth thresholds through lead time."
                  insight="Use this to see whether the selected point has a brief probability spike, sustained risk, or no threshold crossing."
                />
              </figure>
              <figure className="chart-figure">
                <PoiArrivalHistogram series={cellSeries} />
                <FigureCaption
                  title="Arrival-time distribution"
                  caption="Member-wise first-wet timing distribution at the selected cell."
                  insight="A tight histogram means members agree on onset timing; a broad histogram means local arrival timing is uncertain."
                />
              </figure>
            </div>
          )}
        </>
        )}
      </CellInspector>
      </section>

      {figureArtifacts.length > 0 && (
        <section className="panel">
          <h2>Python-generated UQ figures</h2>
          <div className="figure-grid">
            {figureArtifacts.map((figure) => (
              <article className="figure-card" key={figure.id}>
                <h3>{figure.title}</h3>
                <img src={artifactUrl(figure.id)} alt={`${figure.title} for run ${runId}`} />
                <FigureNote caption={figure.caption} insight={figure.insight} />
              </article>
            ))}
          </div>
        </section>
      )}

      <section className="panel">
        <h2>UQ summary</h2>
        {!summary && <p className="muted">Summary JSON will appear once postprocessing finishes.</p>}
        {summary && (
          <table className="summary-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>{view === "calibrated" ? "Calibrated" : "Raw FGN"}</th>
                {otherSummary && <th>{view === "calibrated" ? "Raw FGN" : "Calibrated"}</th>}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Members</td>
                <td>{summary.n_members}</td>
                {otherSummary && <td>{otherSummary.n_members}</td>}
              </tr>
              <tr>
                <td>Peak expected flooded area (km²)</td>
                <td>{formatNumber(summary["peak_expected_flooded_area_km2_gt_0.05m"])}</td>
                {otherSummary && <td>{formatNumber(otherSummary["peak_expected_flooded_area_km2_gt_0.05m"])}</td>}
              </tr>
              <tr>
                <td>Peak expected flooded area (% wettable)</td>
                <td>{formatNumber(100 * Number(summary["peak_expected_flooded_area_fraction_wettable_gt_0.05m"] ?? 0), 2)}</td>
                {otherSummary && <td>{formatNumber(100 * Number(otherSummary["peak_expected_flooded_area_fraction_wettable_gt_0.05m"] ?? 0), 2)}</td>}
              </tr>
              <tr>
                <td>Peak area-weighted IQR / p95-p05 (m)</td>
                <td>{formatNumber(summary.peak_area_weighted_iqr_wd_m)} / {formatNumber(summary.peak_area_weighted_central_90_wd_m)}</td>
                {otherSummary && (
                  <td>{formatNumber(otherSummary.peak_area_weighted_iqr_wd_m)} / {formatNumber(otherSummary.peak_area_weighted_central_90_wd_m)}</td>
                )}
              </tr>
              <tr>
                <td>Isotonic exceedance applied</td>
                <td>{summary.isotonic_calibration_applied ? "yes" : "no"}</td>
                {otherSummary && <td>{otherSummary.isotonic_calibration_applied ? "yes" : "no"}</td>}
              </tr>
            </tbody>
          </table>
        )}
        {summary && exceedanceArea.length > 0 && (
          <table className="summary-table">
            <thead>
              <tr>
                <th>Threshold (m)</th>
                <th>Peak expected footprint (% wettable)</th>
                {otherSummary && <th>{view === "calibrated" ? "Raw" : "Calibrated"}</th>}
              </tr>
            </thead>
            <tbody>
              {exceedanceArea.map((row) => {
                const other = exceedanceAreaEntries(otherSummary).find((entry) => entry.threshold === row.threshold);
                return (
                  <tr key={row.threshold}>
                    <td>{row.threshold.toFixed(2)}</td>
                    <td>{formatNumber(row.peakFraction * 100, 2)}</td>
                    {otherSummary && <td>{other ? formatNumber(other.peakFraction * 100, 2) : "—"}</td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>

      {/* Forcing hydrograph used to live here as a standalone panel; the
         canonical view now lives in the Drivers tab where stage and rainfall
         are shown alongside the response curves on a single time axis. */}

      <ArtifactDrawer
        title={hasHdf5 ? "Downloads and ensemble data" : "Downloads"}
        artifacts={artifacts.slice().sort((a, b) => a.artifact_id.localeCompare(b.artifact_id))}
        hrefForArtifact={(artifact) => artifactUrl(artifact.artifact_id)}
        initiallyOpen={false}
      />
      </>
        </HazardTab>
      )}

      <details className="panel">
        <summary>Run specification</summary>
        <pre>{run ? JSON.stringify(run.spec, null, 2) : "Loading…"}</pre>
      </details>

      <style jsx>{`
        .shell { max-width: 1540px; margin: 0 auto; padding: 24px 28px 54px; font-family: var(--font); color: var(--text); }
        .run-command-header {
          position: sticky;
          top: 58px;
          z-index: 34;
          margin: -4px -10px 14px;
          padding: 10px;
          border: 1px solid rgba(197, 211, 220, 0.72);
          border-radius: 14px;
          background: rgba(247, 250, 252, 0.92);
          backdrop-filter: blur(16px);
          box-shadow: 0 14px 34px rgba(7, 20, 29, 0.08);
        }
        .run-tab-frame {
          display: grid;
          gap: 16px;
        }
        .run-tab-frame[aria-label="Hazard"] > .hazard-map-layout { order: 1; }
        .run-tab-frame[aria-label="Hazard"] > .cards { order: 2; }
        .run-tab-frame[aria-label="Hazard"] > .envelope-panel { order: 3; }
        .run-tab-frame[aria-label="Hazard"] > .panel:not(.hazard-map-panel):not(.envelope-panel) { order: 4; }
        .run-tab-frame[aria-label="Hazard"] > .artifact-drawer { order: 5; }
        .hazard-map-layout {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(360px, 440px);
          gap: 16px;
          align-items: start;
        }
        .hazard-map-panel {
          min-width: 0;
        }
        .time-player {
          display: grid;
          gap: 14px;
        }
        .hazard-inspector {
          position: sticky;
          top: 16px;
          max-height: calc(100vh - 32px);
          overflow: auto;
        }
        .inspector-empty {
          min-height: 260px;
          align-content: center;
          gap: 8px;
        }
        .tabs {
          display: flex;
          gap: 4px;
          margin: 20px 0 6px;
          padding: 4px;
          background: rgba(245, 249, 251, 0.92);
          border-radius: 10px;
          border: 1px solid var(--border);
          width: fit-content;
          backdrop-filter: blur(14px);
        }
        .tab {
          position: relative;
          background: transparent;
          color: var(--text-secondary);
          padding: 8px 16px;
          font-size: 13.5px;
          font-weight: 600;
          letter-spacing: -0.005em;
          border: 0;
          border-radius: 8px;
          cursor: pointer;
          transition: background 120ms ease, color 120ms ease;
          display: inline-flex;
          align-items: center;
          gap: 8px;
        }
        .tab:hover { background: #ffffff; color: var(--brand-strong); }
        .tab.active { background: #ffffff; color: var(--brand-strong); box-shadow: 0 1px 3px rgba(7, 20, 29, 0.12); }
        .tab-intro {
          color: var(--text-secondary);
          font-size: 13.5px;
          line-height: 1.55;
          margin: 8px 0 22px;
          max-width: 880px;
        }
        .tab-intro strong { color: var(--text); }
        .tab-stub { background: var(--surface-muted); }
        .tab-stub h2 { font-size: 18px; font-weight: 800; margin: 0 0 8px; color: var(--text); letter-spacing: -0.01em; }
        .tab-stub p { color: var(--text-secondary); font-size: 13.5px; line-height: 1.55; margin: 0; max-width: 760px; }
        .driver-stack { display: grid; gap: 12px; }
        .compare-picker {
          display: grid;
          grid-template-columns: minmax(220px, 1fr) minmax(260px, 1fr);
          gap: 16px;
          align-items: end;
        }
        .compare-picker select {
          width: 100%;
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 10px 12px;
          background: #ffffff;
          color: var(--text);
          font-size: 13px;
        }
        .compare-map-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 14px;
          margin-top: 14px;
        }
        .leaderboard-table { overflow-x: auto; }
        .clickable-row { cursor: pointer; }
        .clickable-row:hover { background: #f0fdfa; }
        .table-sub { display: block; margin-top: 2px; color: #64748b; font-size: 11px; font-weight: 500; }
        .figure-wide {
          width: 100%;
          max-height: 420px;
          object-fit: contain;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
        }
        .poi-context {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 8px;
        }
        .poi-context span {
          border: 1px solid #dbe3ea;
          border-radius: 999px;
          padding: 4px 8px;
          color: #334155;
          background: #f8fafc;
          font-size: 11px;
          font-weight: 700;
        }
        .envelope-panel { margin-top: 18px; }
        .envelope-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; gap: 14px; flex-wrap: wrap; }
        .envelope-head h2 { font-size: 18px; font-weight: 700; margin: 0; color: #0f172a; letter-spacing: -0.01em; }
        .envelope-sub { color: #94a3b8; font-size: 12px; }
        .envelope-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
          gap: 14px;
        }
        .envelope-tile {
          margin: 0;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .envelope-tile img {
          display: block;
          width: 100%;
          height: auto;
          background: #0f172a;
        }
        .envelope-tile figcaption {
          display: grid;
          gap: 4px;
          padding: 10px 14px 12px;
        }
        .envelope-tile strong { font-size: 13px; font-weight: 700; color: #0f172a; }
        .envelope-tile span { color: #475569; font-size: 12px; line-height: 1.45; }

        /* ===== Uncertainty tab ======================================= */
        .share-strip { margin-top: 16px; }
        .share-bar {
          display: flex;
          height: 14px;
          border-radius: 999px;
          overflow: hidden;
          background: #e2e8f0;
        }
        .share-bar-between { background: #ca8a04; }
        .share-bar-within { background: #0e7490; }
        .share-legend {
          display: flex;
          gap: 18px;
          margin-top: 8px;
          font-size: 12.5px;
          color: #475569;
          font-weight: 600;
        }
        .share-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 6px; vertical-align: middle; }
        .dot-between { background: #ca8a04; }
        .dot-within { background: #0e7490; }
        .reliability-summary {
          display: grid;
          gap: 6px;
          margin-top: 14px;
          padding: 12px 14px;
          background: #f8fafc;
          border: 1px solid #e2e8f0;
          border-radius: 10px;
        }
        .reliability-row {
          display: grid;
          grid-template-columns: 100px 100px 100px 1fr;
          gap: 14px;
          align-items: center;
          font-size: 13px;
          font-variant-numeric: tabular-nums;
        }
        .reliability-row strong { font-weight: 700; color: #0f172a; }
        .reliability-row span { color: #475569; }
        .reliability-row .delta-up { color: #b45309; font-weight: 700; }
        .reliability-row .delta-down { color: #0e7490; font-weight: 700; }
        .reliability-row .delta-flat { color: #94a3b8; font-weight: 600; }

        /* ===== Per-cell inspector ==================================== */
        /* MapWithCellPicker is a child component — styled-jsx scoping
           won't reach its elements. All positioning styles for the map
           frame, marker, and hint are applied as inline styles inside
           that component. Only the poi-panel below (rendered by the
           parent) needs scoped CSS. */
        .poi-panel { margin-top: 18px; background: #ffffff; }
        .poi-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; flex-wrap: wrap; }
        .poi-eyebrow { color: #94a3b8; font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 4px; }
        .poi-head h2 { font-size: 18px; font-weight: 700; margin: 0; color: #0f172a; letter-spacing: -0.01em; }
        .poi-coords { color: #64748b; font-weight: 500; font-size: 14px; }
        .poi-headline { color: #475569; font-size: 13.5px; line-height: 1.55; margin: 8px 0 0; }
        .poi-headline strong { color: #0f172a; font-weight: 700; }
        .poi-clear {
          background: #fef2f2;
          color: #b91c1c;
          border: 1px solid #fecaca;
          padding: 6px 12px;
          font-size: 12px;
          font-weight: 600;
          border-radius: 7px;
          cursor: pointer;
        }
        .poi-clear:hover { background: #fee2e2; }
        .poi-grid { display: grid; gap: 14px; margin-top: 16px; }
        .poi-chart { display: block; width: 100%; height: auto; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 6px; }
        .chart-figure { margin: 0; display: grid; gap: 8px; }
        .back { color: #0f766e; font-weight: 700; text-decoration: none; }
        .back:hover { text-decoration: underline; }
        .hero { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-top: 12px; }
        .hero-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; margin-top: 12px; }
        .action-button {
          border: 1px solid #cbd5e1;
          border-radius: 8px;
          background: #ffffff;
          color: #0f172a;
          cursor: pointer;
          font-weight: 700;
          padding: 8px 12px;
        }
        .action-button:hover:not(:disabled) { border-color: #0f766e; color: #0f766e; }
        .action-button.danger { border-color: #fecaca; background: #fff7f7; color: #b91c1c; }
        .action-button.danger:hover:not(:disabled) { background: #fee2e2; color: #991b1b; }
        .action-button:disabled { cursor: not-allowed; opacity: 0.55; }
        h1 { font-size: 36px; margin: 8px 0 4px; }
        .meta { color: #475569; font-size: 14px; }
        .status { display: inline-block; padding: 2px 10px; border-radius: 999px; background: #e2e8f0; font-weight: 700; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; }
        .status-COMPLETED { background: #dcfce7; color: #166534; }
        .status-RUNNING { background: #fde68a; color: #78350f; }
        .status-POSTPROCESSING { background: #ddd6fe; color: #4c1d95; }
        .status-QUEUED { background: #e0e7ff; color: #312e81; }
        .status-FAILED { background: #fecaca; color: #991b1b; }
        .status-CANCELED { background: #e2e8f0; color: #475569; }
        .status-EXPIRED { background: #fde2e2; color: #7f1d1d; }
        .pin { color: #92400e; font-weight: 700; }
        .panel { border: 1px solid #cbd5e1; border-radius: 18px; padding: 22px; margin-top: 18px; background: #f8fafc; }
        .monitoring-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
        .monitoring-card { display: grid; gap: 5px; border: 1px solid #dbe3ea; background: #ffffff; border-radius: 14px; padding: 14px; }
        .monitoring-card span { color: #64748b; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }
        .monitoring-card strong { color: #0f172a; font-size: 20px; }
        .monitoring-card small { color: #475569; line-height: 1.35; }
        @media (max-width: 860px) { .monitoring-grid { grid-template-columns: 1fr; } }
        .run-progress-panel { background: linear-gradient(180deg, #ffffff, #f8fafc); border-color: #b7d8d4; }
        .run-progress-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; }
        .run-progress-top h2 { margin: 2px 0 0; font-size: 18px; letter-spacing: -0.01em; }
        .run-progress-top strong { font-size: 34px; font-weight: 780; color: #0f766e; line-height: 1; font-variant-numeric: tabular-nums; }
        .progress-track { height: 10px; border-radius: 999px; background: #dbe6ea; overflow: hidden; margin-top: 14px; }
        .progress-track span { display: block; height: 100%; background: linear-gradient(90deg, #0f766e, #2563eb); transition: width 250ms ease; }
        .runtime-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }
        .runtime-strip div { padding: 11px 12px; border: 1px solid #d8e3e6; border-radius: 10px; background: #ffffff; min-width: 0; }
        .runtime-strip span, .runtime-strip small { display: block; color: #64748b; font-size: 12px; line-height: 1.35; }
        .runtime-strip strong { display: block; margin-top: 3px; color: #10231f; font-size: 16px; font-weight: 760; font-variant-numeric: tabular-nums; }
        .runtime-strip small { margin-top: 4px; }
        .error-panel { background: #fee2e2; border-color: #fecaca; }
        .error { color: #991b1b; font-weight: 700; }
        .row-between { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
        .toggle button { border: 1px solid #cbd5e1; background: #ffffff; padding: 8px 14px; font-weight: 600; cursor: pointer; }
        .toggle button:first-child { border-radius: 999px 0 0 999px; }
        .toggle button:last-child { border-radius: 0 999px 999px 0; }
        .toggle button.active { background: #0f766e; color: white; border-color: #0f766e; }
        .product-row { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
        .chip { padding: 6px 12px; border-radius: 999px; border: 1px solid #cbd5e1; background: white; font-size: 13px; cursor: pointer; }
        .chip.active { background: #134e4a; color: white; border-color: #134e4a; }
        .map-image { display: block; margin: 16px auto 8px; max-width: 100%; border-radius: 12px; box-shadow: 0 1px 4px rgba(15,23,42,0.08); }
        .time-slider { display: flex; align-items: center; gap: 14px; margin-top: 6px; }
        .time-slider input[type="range"] { flex: 1; }
        .forecast-scrubber {
          margin-top: 12px;
          padding: 12px 14px;
          background: #f8fafc;
          border: 1px solid #e2e8f0;
          border-radius: 10px;
          display: grid;
          gap: 8px;
        }
        .forecast-scrubber-controls { display: flex; align-items: center; gap: 8px; }
        .forecast-step-btn {
          width: 32px;
          height: 32px;
          padding: 0;
          background: #ffffff;
          color: #475569;
          border: 1px solid #e2e8f0;
          border-radius: 7px;
          font-size: 13px;
          cursor: pointer;
          display: grid;
          place-items: center;
          transition: border-color 120ms ease, color 120ms ease;
        }
        .forecast-step-btn:hover:not(:disabled) { border-color: #0f766e; color: #0f766e; }
        .forecast-step-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .forecast-scrub-range {
          flex: 1;
          -webkit-appearance: none;
          appearance: none;
          height: 6px;
          background: linear-gradient(to right, #0f766e 0%, #0f766e var(--scrub-pct, 0%), #cbd5e1 var(--scrub-pct, 0%), #cbd5e1 100%);
          border-radius: 999px;
          cursor: pointer;
          outline: none;
        }
        .forecast-scrub-range::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #ffffff;
          border: 2px solid #0f766e;
          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.18);
          cursor: grab;
        }
        .forecast-scrub-range::-moz-range-thumb {
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #ffffff;
          border: 2px solid #0f766e;
          cursor: grab;
        }
        .forecast-scrubber-meta {
          display: flex;
          align-items: center;
          gap: 14px;
          font-size: 12.5px;
          color: #475569;
          font-variant-numeric: tabular-nums;
        }
        .forecast-step-tag { font-weight: 700; color: #0f172a; }
        .forecast-step-of { color: #94a3b8; font-weight: 500; }
        .forecast-lead strong { color: #0f172a; font-weight: 700; }
        .forecast-step-hint {
          margin-left: auto;
          padding: 2px 9px;
          background: #fef3c7;
          color: #92400e;
          border-radius: 999px;
          font-size: 10.5px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          cursor: help;
        }
        .anim-link { display: inline-block; margin-top: 10px; color: #0f766e; font-weight: 700; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 20px; }
        .card { border: 1px solid #cbd5e1; border-radius: 14px; padding: 14px 18px; background: linear-gradient(180deg, #ffffff, #f8fafc); }
        .card-label { display: block; font-size: 12px; color: #475569; letter-spacing: 0.04em; text-transform: uppercase; }
        .card-value { display: block; margin-top: 6px; font-size: 22px; font-weight: 700; color: #0f172a; }
        .card-note { display: block; margin-top: 6px; color: #64748b; font-size: 12px; line-height: 1.35; }
        .figure-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
        .figure-card { border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px; background: #ffffff; }
        .figure-card:first-child { grid-column: 1 / -1; }
        .figure-card h3 { margin: 0 0 8px; color: #10231f; font-size: 14px; }
        .figure-card img { display: block; width: 100%; object-fit: contain; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff; }
        .summary-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        .summary-table th, .summary-table td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }
        .summary-table th { color: #475569; font-weight: 700; }
        .hydrograph { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        .hydrograph h3 { margin: 0 0 6px; font-size: 14px; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; }
        .downloads { padding: 0; margin: 0; list-style: none; display: grid; gap: 6px; }
        .downloads a { color: #0f766e; font-weight: 700; }
        details > pre { overflow: auto; background: #0f172a; color: #e2e8f0; padding: 18px; border-radius: 14px; }
        .muted { color: #475569; font-style: italic; }
        @media (max-width: 720px) {
          .hero { flex-direction: column; }
          .hero-actions { justify-content: flex-start; margin-top: 0; }
          .run-progress-top { flex-direction: column; }
          .runtime-strip { grid-template-columns: 1fr; }
          .hydrograph, .figure-grid { grid-template-columns: 1fr; }
          .hazard-map-layout { grid-template-columns: 1fr; }
          .hazard-inspector { position: static; max-height: none; }
          .compare-picker, .compare-map-grid { grid-template-columns: 1fr; }
          .figure-card:first-child { grid-column: auto; }
        }
      `}</style>
    </div>
    </AppShell>
  );
}
