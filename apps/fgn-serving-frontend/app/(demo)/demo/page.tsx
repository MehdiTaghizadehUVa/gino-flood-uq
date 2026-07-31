"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, ArrowRight, Cpu, Database, ListChecks, RefreshCw, ShieldCheck, Waves } from "lucide-react";
import { AppShell } from "../../components/AppShell";
import { MetricCard } from "../../components/MetricCard";
import { PageHeader } from "../../components/PageHeader";
import { ResearchNotice } from "../../components/ResearchNotice";
import { SAMPLE_SCENARIOS, type SampleScenario } from "../../sampleScenarios";
import {
  GUEST_SUBMISSION_DRAFT_KEY,
  GUEST_SUBMISSION_DRAFT_VERSION,
  MAX_GUEST_DRAFT_CSV_CHARS,
  buildSubmissionSignInPath,
  parseGuestSubmissionDraft,
} from "./guestSubmission.mjs";

type User = {
  email: string;
  is_admin: boolean;
  disclaimer_acknowledged: boolean;
};

type Bundle = {
  bundle_id: string;
  domain_name: string;
  max_forecast_steps: number;
  dt_seconds: number;
  n_history: number;
  skip_before_timestep: number;
  n_checkpoints: number;
  members_per_checkpoint: number;
  total_members: number;
  research_disclaimer: string;
  initial_condition?: {
    default_mode?: string;
    reference_scope?: string | null;
    k_neighbors?: number | null;
    has_library?: boolean;
  };
};

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

type RunRow = {
  run_id: string;
  label?: string | null;
  status: RunStatus;
  progress: number;
  progress_label?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  runtime_seconds?: number | null;
  timing?: RunTiming;
  pinned: boolean;
  failure_reason?: string | null;
  result_availability?: Record<string, boolean>;
  cache?: {
    mode?: string;
    materialized_from_cache?: boolean;
    waiting_for_cached_result?: boolean;
    cache_key_prefix?: string | null;
  };
  spec?: {
    forecast_steps?: number;
    request_animation?: boolean;
    request_full_hdf5?: boolean;
    ensemble_count?: number;
    members_per_ensemble?: number;
    exceedance_thresholds_m?: number[];
  };
};

type Artifact = {
  artifact_id: string;
  content_type: string;
  size_bytes: number;
};

type ForcingPoint = {
  timeHours: number;
  stage: number;
  precipitation: number;
};

type ValidationState = {
  valid: boolean;
  messages: string[];
  summary?: Record<string, unknown> | null;
  screening?: {
    available?: boolean;
    monitoring_bundle_id?: string;
    input_novelty_score?: number | null;
    candidate_recommended?: boolean;
    flags?: { code: string; message: string; severity?: string; descriptor?: string; value?: number }[];
  };
};

type RunStatus =
  | "SUBMITTED"
  | "VALIDATING"
  | "WAITING_FOR_CACHE"
  | "QUEUED"
  | "RUNNING"
  | "POSTPROCESSING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELED"
  | "EXPIRED"
  | "DELETED";

type HomeWorkspace = "new" | "runs";
type AuthState = "checking" | "guest" | "authenticated";

const SUBMISSION_SIGN_IN_PATH = buildSubmissionSignInPath();

const DESCRIPTOR_LABELS: Record<string, string> = {
  stage_max: "Peak coastal stage",
  stage_min: "Minimum coastal stage",
  stage_range: "Coastal-stage range",
  precipitation_total: "Total precipitation",
  precipitation_mean: "Mean precipitation",
  precipitation_max: "Peak precipitation",
  precipitation_active_hours: "Precipitation duration",
  precipitation_peak_lead_hours: "Lead time to peak precipitation",
};

function formatDescriptorLabel(descriptor?: string | null): string {
  if (!descriptor) return "Monitored scenario pattern";
  if (DESCRIPTOR_LABELS[descriptor]) return DESCRIPTOR_LABELS[descriptor];
  return descriptor
    .replace(/wd/g, "water depth")
    .replace(/iqr/g, "IQR")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatScreeningFlag(flag: { code: string; message: string; descriptor?: string | null }): string {
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
      return "The combined input pattern is unusual relative to the reference population.";
    default:
      return flag.message.replace(/_/g, " ");
  }
}

const RUN_STAGES: RunStatus[] = ["SUBMITTED", "VALIDATING", "WAITING_FOR_CACHE", "QUEUED", "RUNNING", "POSTPROCESSING", "COMPLETED"];
const ACTIVE_STATUSES: ReadonlySet<RunStatus> = new Set([
  "SUBMITTED",
  "VALIDATING",
  "WAITING_FOR_CACHE",
  "QUEUED",
  "RUNNING",
  "POSTPROCESSING",
]);
const TERMINAL_STATUSES: ReadonlySet<RunStatus> = new Set(["COMPLETED", "FAILED", "CANCELED", "EXPIRED", "DELETED"]);
const TERMINAL_FAILED: ReadonlySet<RunStatus> = new Set(["FAILED", "CANCELED", "EXPIRED", "DELETED"]);
const POLL_INTERVAL_ACTIVE_MS = 4000;
const POLL_INTERVAL_IDLE_MS = 20000;

function workspaceFromUrl(search: string, hash: string): HomeWorkspace {
  if (hash === "#new-run") return "new";
  if (hash === "#runs") return "runs";
  const params = new URLSearchParams(search);
  const requested = params.get("workspace");
  if (requested === "runs") return "runs";
  return "new";
}

function workspaceFromLocation(): HomeWorkspace {
  if (typeof window === "undefined") return "new";
  return workspaceFromUrl(window.location.search, window.location.hash);
}

type StageMark = "done" | "current" | "pending" | "failed" | "canceled";

function deriveStageMarks(status: RunStatus, progress: number): StageMark[] {
  if (status === "COMPLETED") return RUN_STAGES.map(() => "done");
  const idx = RUN_STAGES.indexOf(status);
  if (idx >= 0) {
    return RUN_STAGES.map((_, i) =>
      i < idx ? "done" : i === idx ? "current" : "pending",
    );
  }
  if (TERMINAL_FAILED.has(status)) {
    // Terminal-failed status is not in the happy-path stage list. Estimate the
    // last stage that was reached from the run's reported progress so the user
    // can see where it stopped, and mark that stage with the failure tone.
    const span = RUN_STAGES.length - 1;
    const reached = Math.min(
      span,
      Math.max(0, Math.round((Number.isFinite(progress) ? progress : 0) * span)),
    );
    const tone: StageMark = status === "CANCELED" || status === "DELETED" ? "canceled" : "failed";
    return RUN_STAGES.map((_, i) => (i < reached ? "done" : i === reached ? tone : "pending"));
  }
  return RUN_STAGES.map(() => "pending");
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function formatDuration(seconds: unknown) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "Not enough history";
  const rounded = Math.max(0, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (hours > 0) return `${hours}h ${minutes.toString().padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m ${secs.toString().padStart(2, "0")}s`;
  return `${secs}s`;
}

function statusTone(status: RunStatus) {
  if (status === "COMPLETED") return "good";
  if (status === "FAILED" || status === "CANCELED" || status === "EXPIRED" || status === "DELETED") return "bad";
  if (status === "RUNNING" || status === "POSTPROCESSING") return "active";
  return "waiting";
}

function asNumber(value: unknown) {
  const next = typeof value === "number" ? value : Number(value);
  return Number.isFinite(next) ? next : null;
}

function formatMetric(value: unknown, unit = "") {
  const next = asNumber(value);
  if (next === null) return "-";
  const formatted = Math.abs(next) >= 100 ? next.toFixed(0) : next.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return unit ? `${formatted} ${unit}` : formatted;
}

function mapArtifactLabel(artifactId: string) {
  const withoutExt = artifactId.replace(/\.(png|gif)$/, "");
  const probabilityMatch = withoutExt.match(/p_gt_([0-9]+)p([0-9]+)m/);
  const probabilityLabel = probabilityMatch
    ? `Chance Depth Passes ${Number(`${probabilityMatch[1]}.${probabilityMatch[2]}`).toFixed(2)} m`
    : null;
  const label = withoutExt
    .replace(/^calibrated_/, "Checked ")
    .replace(/^raw_/, "Original ")
    .replace(/_t\d+$/, "")
    .replace("mean_wd_animation", "Expected Depth Animation")
    .replace(/p_gt_[0-9]+p[0-9]+m/, probabilityLabel ?? "Depth-Threshold Probability")
    .replace("iqr", "Middle 50% Range")
    .replace("p95", "95th-Percentile Depth")
    .replace("spread", "Forecast Range")
    .replace("mean", "Expected Depth")
    .replaceAll("_", " ");
  return label.replace(/\b\w/g, (char) => char.toUpperCase());
}

function preferredMapId(artifacts: Artifact[]) {
  const mapIds = artifacts.filter((artifact) => artifact.artifact_id.endsWith(".png")).map((artifact) => artifact.artifact_id);
  return (
    mapIds.find((id) => id.startsWith("calibrated_p_gt_")) ??
    mapIds.find((id) => id.startsWith("calibrated_iqr_")) ??
    mapIds.find((id) => id.startsWith("calibrated_p95_")) ??
    mapIds.find((id) => id.startsWith("calibrated_mean_")) ??
    mapIds.find((id) => id.startsWith("calibrated_p_gt_")) ??
    mapIds.find((id) => id.startsWith("calibrated_")) ??
    mapIds[0] ??
    null
  );
}

async function parseForcingPreview(file: File | null): Promise<ForcingPoint[]> {
  if (!file) return [];
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return [];
  const header = lines[0].split(",").map((part) => part.trim().toLowerCase());
  const timeIndex = header.findIndex((part) => ["time_seconds", "time", "seconds", "t"].includes(part));
  const stageIndex = header.findIndex((part) => ["stage", "stage_m", "coastal_stage"].includes(part));
  const precipIndex = header.findIndex((part) => ["precipitation", "precip", "rainfall", "rain"].includes(part));
  if (timeIndex < 0 || stageIndex < 0 || precipIndex < 0) return [];
  return lines.slice(1).flatMap((line) => {
    const parts = line.split(",").map((part) => part.trim());
    const timeSeconds = asNumber(parts[timeIndex]);
    const stage = asNumber(parts[stageIndex]);
    const precipitation = asNumber(parts[precipIndex]);
    if (timeSeconds === null || stage === null || precipitation === null) return [];
    return [{ timeHours: timeSeconds / 3600, stage, precipitation }];
  });
}

function Sparkline({ points, field, stroke, fill }: { points: ForcingPoint[]; field: "stage" | "precipitation"; stroke: string; fill: string }) {
  const width = 320;
  const height = 90;
  const pad = 8;
  const values = points.map((point) => point[field]).filter(Number.isFinite);
  // Empty after filter: fall back to a [0,1] window so the SVG renders without NaN.
  const dataMax = values.length ? Math.max(...values) : 1;
  const dataMin = values.length ? Math.min(...values) : 0;
  // Precipitation is non-negative by definition, so anchor the baseline at 0.
  // Stage can swing negative (tide trough); auto-fit to the observed range so
  // small variations remain visible instead of being squashed by a 0 anchor.
  const max = field === "precipitation" ? Math.max(dataMax, 1e-3) : dataMax;
  const min = field === "precipitation" ? 0 : dataMin;
  const span = Math.max(max - min, 1e-9);
  const xFor = (index: number) => pad + (index / Math.max(points.length - 1, 1)) * (width - pad * 2);
  const yFor = (value: number) => height - pad - ((value - min) / span) * (height - pad * 2);
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(index).toFixed(2)} ${yFor(point[field]).toFixed(2)}`).join(" ");
  const area = `${path} L ${width - pad} ${height - pad} L ${pad} ${height - pad} Z`;
  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${field} preview`}>
      <path d={area} fill={fill} />
      <path d={path} fill="none" stroke={stroke} strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function formatPercent(value: unknown, digits = 0): string {
  const next = asNumber(value);
  if (next === null) return "—";
  return `${(next * 100).toFixed(digits)}%`;
}

function formatLeadHours(value: unknown): string {
  const next = asNumber(value);
  if (next === null) return "—";
  return `+${next.toFixed(2)} h`;
}

function exceedanceEntries(summary: Record<string, unknown> | null): { thresholdM: number; probability: number }[] {
  if (!summary) return [];
  const out: { thresholdM: number; probability: number }[] = [];
  for (const [key, value] of Object.entries(summary)) {
    const match = key.match(/^p_wd_gt_([0-9eE+\-.]+)m_mean$/);
    if (!match) continue;
    const thr = Number(match[1]);
    const prob = asNumber(value);
    if (!Number.isFinite(thr) || prob === null) continue;
    out.push({ thresholdM: thr, probability: prob });
  }
  return out.sort((a, b) => a.thresholdM - b.thresholdM);
}

type ExceedanceAreaEntry = {
  thresholdM: number;
  peakFraction: number;
  peakKm2: number;
  highConfidenceFraction: number;
  leadHours: number | null;
};

function exceedanceAreaEntries(summary: Record<string, unknown> | null): ExceedanceAreaEntry[] {
  const raw = summary?.exceedance_by_threshold_m;
  if (!raw || typeof raw !== "object") return [];
  return Object.entries(raw as Record<string, Record<string, unknown>>)
    .flatMap(([key, value]) => {
      const thresholdM = Number(value.threshold_m ?? key);
      const peakFraction = asNumber(value.peak_expected_area_fraction_wettable);
      if (!Number.isFinite(thresholdM) || peakFraction === null) return [];
      return [{
        thresholdM,
        peakFraction,
        peakKm2: asNumber(value.peak_expected_area_km2) ?? 0,
        highConfidenceFraction: asNumber(value.peak_high_confidence_area_fraction_wettable) ?? 0,
        leadHours: asNumber(value.peak_expected_area_lead_hours),
      }];
    })
    .sort((a, b) => a.thresholdM - b.thresholdM);
}

function preferredExceedanceAreaEntry(entries: ExceedanceAreaEntry[]): ExceedanceAreaEntry | null {
  if (entries.length === 0) return null;
  return entries.find((entry) => Math.abs(entry.thresholdM - 0.30) < 1e-6) ?? entries[entries.length - 1];
}

function artifactUrl(runId: string, artifactId: string) {
  return `/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}`;
}

function niceTicks(min: number, max: number, count = 4) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (Math.abs(max - min) < 1e-9) {
    const pad = Math.max(Math.abs(max) * 0.1, 0.1);
    min -= pad;
    max += pad;
  }
  const rawStep = Math.abs(max - min) / Math.max(count - 1, 1);
  const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const fraction = rawStep / power;
  const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  const step = niceFraction * power;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let tick = start; tick <= end + step * 0.5; tick += step) {
    ticks.push(Number(tick.toPrecision(12)));
  }
  return ticks;
}

function formatAxisTick(value: number) {
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

// ---------------------------------------------------------------------------
// Lightweight visualization primitives
// ---------------------------------------------------------------------------
// These are intentionally inline SVG rather than a chart library: each chart
// is small, the data shapes are stable, and we want zero added bundle weight
// in the user-facing entry route.

function InfoTip({ label, children }: { label: React.ReactNode; children: React.ReactNode }) {
  return (
    <span className="info-tip" tabIndex={0}>
      <span className="info-tip-label">{label}</span>
      <span className="info-tip-icon" aria-hidden>
        ⓘ
      </span>
      <span className="info-tip-bubble" role="tooltip">
        {children}
      </span>
    </span>
  );
}

type TimeSeriesPoint = { t: number; y: number };

function TimeSeriesChart({
  points,
  yLabel,
  yUnit,
  stroke,
  fill,
  markerT,
  markerLabel,
  height = 140,
  decimals = 2,
}: {
  points: TimeSeriesPoint[];
  yLabel: string;
  yUnit: string;
  stroke: string;
  fill: string;
  markerT?: number | null;
  markerLabel?: string;
  height?: number;
  decimals?: number;
}) {
  const width = 520;
  const padL = 44;
  const padR = 12;
  const padT = 14;
  const padB = 28;
  const cleaned = points.filter((p) => Number.isFinite(p.t) && Number.isFinite(p.y));
  if (cleaned.length < 2) {
    return <p className="chart-empty">Not enough samples to plot.</p>;
  }
  const tMin = cleaned[0].t;
  const tMax = cleaned[cleaned.length - 1].t;
  const yMin = Math.min(0, ...cleaned.map((p) => p.y));
  const yMax = Math.max(...cleaned.map((p) => p.y), yMin + 1e-6);
  const tSpan = Math.max(tMax - tMin, 1e-9);
  const ySpan = Math.max(yMax - yMin, 1e-9);
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const xFor = (t: number) => padL + ((t - tMin) / tSpan) * innerW;
  const yFor = (y: number) => padT + (1 - (y - yMin) / ySpan) * innerH;
  const path = cleaned.map((p, i) => `${i === 0 ? "M" : "L"} ${xFor(p.t).toFixed(2)} ${yFor(p.y).toFixed(2)}`).join(" ");
  const area = `${path} L ${xFor(tMax).toFixed(2)} ${yFor(yMin).toFixed(2)} L ${xFor(tMin).toFixed(2)} ${yFor(yMin).toFixed(2)} Z`;
  // 4 evenly-spaced y-axis ticks.
  const yTicks = Array.from({ length: 4 }, (_, i) => yMin + ((yMax - yMin) * i) / 3);
  const xTicks = [tMin, tMin + (tMax - tMin) / 2, tMax];
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${yLabel} over forecast horizon`}>
      <rect x={padL} y={padT} width={innerW} height={innerH} fill="#ffffff" stroke="#d5e3df" />
      {yTicks.map((tick) => (
        <g key={`y${tick}`}>
          <line x1={padL} x2={padL + innerW} y1={yFor(tick)} y2={yFor(tick)} stroke="#eef2f1" />
          <text x={padL - 6} y={yFor(tick) + 3} fontSize="10" textAnchor="end" fill="#5b6f6a">
            {tick.toFixed(decimals)}
          </text>
        </g>
      ))}
      {xTicks.map((tick) => (
        <text key={`x${tick}`} x={xFor(tick)} y={height - 8} fontSize="10" textAnchor="middle" fill="#5b6f6a">
          {tick.toFixed(1)} h
        </text>
      ))}
      <path d={area} fill={fill} />
      <path d={path} fill="none" stroke={stroke} strokeWidth={2.4} strokeLinecap="round" />
      {markerT !== undefined && markerT !== null && Number.isFinite(markerT) && markerT >= tMin && markerT <= tMax && (
        <g>
          <line x1={xFor(markerT)} x2={xFor(markerT)} y1={padT} y2={padT + innerH} stroke="#dc2626" strokeDasharray="3 3" />
          <text x={xFor(markerT) + 4} y={padT + 12} fontSize="10" fill="#b91c1c" fontWeight={700}>
            {markerLabel ?? `+${markerT.toFixed(2)} h`}
          </text>
        </g>
      )}
      <text x={padL} y={padT - 4} fontSize="11" fontWeight={700} fill="#0f3f39">
        {yLabel} ({yUnit})
      </text>
    </svg>
  );
}

function ExceedanceBars({
  entries,
  calibrated,
}: {
  entries: { thresholdM: number; probability: number }[];
  calibrated: boolean;
}) {
  if (entries.length === 0) {
    return <p className="chart-empty">No depth-threshold probabilities were reported for this run.</p>;
  }
  const rowH = 26;
  const padL = 56;
  const padR = 60;
  const padT = 4;
  const padB = 4;
  const width = 520;
  const height = padT + padB + entries.length * rowH;
  const innerW = width - padL - padR;
  // Color ramp by threshold severity, not by probability — the threshold itself
  // encodes the consequence; the bar length encodes likelihood.
  const colorFor = (thr: number) => (thr >= 0.5 ? "#8b1e3f" : thr >= 0.3 ? "#c2410c" : thr >= 0.1 ? "#d89000" : "#0b766d");
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Probability that water passes each depth threshold">
      {entries.map((entry, i) => {
        const y = padT + i * rowH + 4;
        const w = Math.max(2, entry.probability * innerW);
        return (
          <g key={entry.thresholdM}>
            <text x={padL - 8} y={y + 12} fontSize="11" textAnchor="end" fill="#21343b" fontWeight={700}>
              &gt; {entry.thresholdM.toFixed(2)} m
            </text>
            <rect x={padL} y={y} width={innerW} height={rowH - 8} fill="#e8eef1" rx={3} />
            <rect x={padL} y={y} width={w} height={rowH - 8} fill={colorFor(entry.thresholdM)} rx={3} />
            <text x={padL + w + 6} y={y + 13} fontSize="11" fill="#21343b" fontWeight={700}>
              {(entry.probability * 100).toFixed(entry.probability < 0.1 ? 1 : 0)}%
            </text>
          </g>
        );
      })}
      <text x={padL} y={height - 2} fontSize="9" fill="#6a7d84">
        {calibrated ? "checked probabilities" : "original forecast-group percentages"}
      </text>
    </svg>
  );
}

function scenarioSpinupRows(bundle: Bundle | null) {
  return (bundle?.skip_before_timestep ?? 12) + (bundle?.n_history ?? 3);
}

function scenarioForecastSteps(bundle: Bundle | null, scenario: SampleScenario) {
  const availableRows = Math.min(scenario.stage.length, scenario.precipitation.length);
  const availableForecastSteps = Math.max(1, availableRows - scenarioSpinupRows(bundle));
  return Math.min(
    scenario.forecastSteps,
    availableForecastSteps,
    bundle?.max_forecast_steps ?? scenario.forecastSteps,
  );
}

function scenarioDurationHours(bundle: Bundle | null, scenario: SampleScenario) {
  const rows = scenarioSpinupRows(bundle) + scenarioForecastSteps(bundle, scenario);
  const dt = bundle?.dt_seconds ?? scenario.dtSeconds;
  return ((Math.max(rows, 1) - 1) * dt) / 3600;
}

function sumFinite(values: number[]) {
  return values.reduce((acc, value) => (Number.isFinite(value) ? acc + value : acc), 0);
}

function ForcingPreviewPanel({
  points,
  bundle,
  forecastStepsText,
}: {
  points: ForcingPoint[];
  bundle: Bundle | null;
  forecastStepsText: string;
}) {
  if (points.length < 2) return null;
  const width = 760;
  const height = 360;
  const padL = 66;
  const padR = 26;
  const padT = 34;
  const padB = 40;
  const gap = 30;
  const panelH = (height - padT - padB - gap) / 2;
  const stageY0 = padT;
  const precipY0 = padT + panelH + gap;
  const innerW = width - padL - padR;
  const spinupRows = scenarioSpinupRows(bundle);
  const requestedSteps = Number.parseInt(forecastStepsText, 10);
  const availableForecastSteps = Math.max(1, points.length - spinupRows);
  const forecastSteps = Number.isFinite(requestedSteps)
    ? Math.min(Math.max(1, requestedSteps), availableForecastSteps)
    : availableForecastSteps;
  const forecastStartIndex = Math.min(spinupRows, points.length - 1);
  const forecastEndIndex = Math.min(spinupRows + forecastSteps - 1, points.length - 1);
  const forecastStartH = points[forecastStartIndex]?.timeHours ?? points[0]?.timeHours ?? 0;
  const plottedPoints = points
    .slice(forecastStartIndex, forecastEndIndex + 1)
    .map((point) => ({ ...point, leadHours: Math.max(0, point.timeHours - forecastStartH) }));
  if (plottedPoints.length < 2) return null;
  const xValues = plottedPoints.map((point) => point.leadHours);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const xSpan = Math.max(xMax - xMin, 1e-9);
  const xFor = (t: number) => padL + ((t - xMin) / xSpan) * innerW;

  const stageMinRaw = Math.min(...plottedPoints.map((point) => point.stage));
  const stageMaxRaw = Math.max(...plottedPoints.map((point) => point.stage));
  const stagePad = Math.max((stageMaxRaw - stageMinRaw) * 0.08, 0.01);
  const stageTicks = niceTicks(stageMinRaw - stagePad, stageMaxRaw + stagePad, 4);
  const stageMin = stageTicks[0];
  const stageMax = stageTicks[stageTicks.length - 1];
  const stageSpan = Math.max(stageMax - stageMin, 1e-9);
  const stageYFor = (value: number) => stageY0 + (1 - (value - stageMin) / stageSpan) * panelH;
  const stagePath = plottedPoints
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(point.leadHours).toFixed(2)} ${stageYFor(point.stage).toFixed(2)}`)
    .join(" ");

  const precipMaxRaw = Math.max(...plottedPoints.map((point) => point.precipitation), 1e-3);
  const precipTicks = niceTicks(0, precipMaxRaw * 1.08, 4).filter((tick) => tick >= -1e-9);
  const precipMax = Math.max(precipTicks[precipTicks.length - 1] ?? precipMaxRaw, precipMaxRaw);
  const precipYFor = (value: number) => precipY0 + (1 - value / Math.max(precipMax, 1e-9)) * panelH;
  const xTicks = Array.from(new Set([xMin, ...niceTicks(xMin, xMax, 6), xMax]))
    .filter((tick) => tick >= xMin - 1e-9 && tick <= xMax + 1e-9)
    .sort((a, b) => a - b);
  const barW = Math.max(2, (innerW / Math.max(plottedPoints.length, 1)) * 0.68);
  const totalPrecip = sumFinite(plottedPoints.map((point) => point.precipitation));
  const hiddenRows = Math.max(0, forecastStartIndex);

  return (
    <div className="forcing-preview">
      <div className="preview-head">
        <div>
          <p className="eyebrow">Input preview</p>
          <h3>Water-level and rainfall inputs during the forecast</h3>
          <p className="preview-subtitle">Earlier rows used to establish the starting state are not shown here.</p>
        </div>
        <span className="chart-pill calibrated">
          {plottedPoints.length} steps
        </span>
      </div>
      <table className="preview-stats" aria-label="Water-level and rainfall input summary">
        <tbody>
          <tr>
            <th scope="row">Forecast rows</th>
            <td><strong>{plottedPoints.length}</strong></td>
          </tr>
          <tr>
            <th scope="row">Forecast window</th>
            <td><strong>{(xMax - xMin).toFixed(1)} h</strong></td>
          </tr>
          <tr>
            <th scope="row">Stage range</th>
            <td><strong>{stageMinRaw.toFixed(2)}-{stageMaxRaw.toFixed(2)} m</strong></td>
          </tr>
          <tr>
            <th scope="row">Peak rain</th>
            <td><strong>{precipMaxRaw.toFixed(2)} mm/step</strong></td>
          </tr>
          <tr>
            <th scope="row">Rain sum</th>
            <td><strong>{totalPrecip.toFixed(1)} mm</strong></td>
          </tr>
          {hiddenRows > 0 && (
            <tr>
              <th scope="row">Hidden setup rows</th>
              <td><strong>{hiddenRows}</strong></td>
            </tr>
          )}
        </tbody>
      </table>
      <svg className="forcing-preview-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Coastal water-level and rainfall preview">
        <rect x={0} y={0} width={width} height={height} fill="#ffffff" />
        <rect x={padL} y={stageY0} width={innerW} height={panelH} fill="#ffffff" stroke="#cbd5e1" />
        <rect x={padL} y={precipY0} width={innerW} height={panelH} fill="#ffffff" stroke="#cbd5e1" />
        {xTicks.map((tick) => (
          <g key={`x-preview-${tick}`}>
            <line x1={xFor(tick)} x2={xFor(tick)} y1={stageY0} y2={stageY0 + panelH} stroke="#eef2f4" />
            <line x1={xFor(tick)} x2={xFor(tick)} y1={precipY0} y2={precipY0 + panelH} stroke="#eef2f4" />
            <text x={xFor(tick)} y={height - 12} fontSize="11" textAnchor="middle" fill="#475569">
              {formatAxisTick(tick)}
            </text>
          </g>
        ))}
        {stageTicks.map((tick) => (
          <g key={`stage-preview-${tick}`}>
            <line x1={padL} x2={padL + innerW} y1={stageYFor(tick)} y2={stageYFor(tick)} stroke="#e2e8f0" />
            <text x={padL - 8} y={stageYFor(tick) + 3.5} fontSize="11" textAnchor="end" fill="#475569">
              {formatAxisTick(tick)}
            </text>
          </g>
        ))}
        {precipTicks.map((tick) => (
          <g key={`precip-preview-${tick}`}>
            <line x1={padL} x2={padL + innerW} y1={precipYFor(tick)} y2={precipYFor(tick)} stroke="#e2e8f0" />
            <text x={padL - 8} y={precipYFor(tick) + 3.5} fontSize="11" textAnchor="end" fill="#475569">
              {formatAxisTick(tick)}
            </text>
          </g>
        ))}
        {plottedPoints.map((point, index) => {
          if (!Number.isFinite(point.precipitation) || point.precipitation <= 0) return null;
          const x = xFor(point.leadHours) - barW / 2;
          const y = precipYFor(point.precipitation);
          const h = precipY0 + panelH - y;
          return (
            <rect key={`precip-bar-${index}`} x={x} y={y} width={barW} height={Math.max(h, 0.5)} fill="#1d4ed8" opacity={0.82} />
          );
        })}
        <path d={stagePath} fill="none" stroke="#0f766e" strokeWidth={2.6} strokeLinecap="round" strokeLinejoin="round" />
        <text x={padL} y={stageY0 - 12} fontSize="12" fontWeight={700} fill="#0f172a">
          Coastal stage (m)
        </text>
        <text x={padL} y={precipY0 - 12} fontSize="12" fontWeight={700} fill="#0f172a">
          Precipitation (mm/step)
        </text>
        <text x={padL + innerW / 2} y={height - 2} fontSize="11" textAnchor="middle" fill="#0f172a">
          Lead time in forecast window (h)
        </text>
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Forcing parser (full CSV → forecast-window slice) for the Time Player.
// Stage and rainfall are fed back into the player's synchronized subplots so
// the user can read the driver alongside the response.
// ---------------------------------------------------------------------------

type ForcingSeries = {
  leadHours: number[];
  stage: number[];
  precipitation: number[];
  forecastStart: number;
  forecastEnd: number;
};

async function fetchForcingSeries(
  runId: string,
  bundle: Bundle | null,
): Promise<ForcingSeries | null> {
  if (!bundle) return null;
  const res = await fetch(
    `/api/runs/${encodeURIComponent(runId)}/artifacts/forcing.csv`,
    { cache: "no-store" },
  );
  if (!res.ok) return null;
  const text = await res.text();
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) return null;
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const tIdx = header.findIndex((h) => ["time_seconds", "time", "t", "seconds"].includes(h));
  const sIdx = header.findIndex((h) => ["stage", "stage_m", "coastal_stage"].includes(h));
  const pIdx = header.findIndex((h) => ["precipitation", "precip", "rainfall", "rain"].includes(h));
  if (sIdx < 0 || pIdx < 0) return null;
  const dt = bundle.dt_seconds;
  const spinupRows = bundle.skip_before_timestep + bundle.n_history;
  const ts: number[] = [];
  const stage: number[] = [];
  const precip: number[] = [];
  lines.slice(1).forEach((line, idx) => {
    const cols = line.split(",");
    const stageVal = Number(cols[sIdx]);
    const precipVal = Number(cols[pIdx]);
    if (!Number.isFinite(stageVal) || !Number.isFinite(precipVal)) return;
    const tSec = tIdx >= 0 ? Number(cols[tIdx]) : idx * dt;
    ts.push(Number.isFinite(tSec) ? tSec : idx * dt);
    stage.push(stageVal);
    precip.push(precipVal);
  });
  if (ts.length === 0) return null;
  // Slice to the forecast window. The ML run's time t=0 is anchored at the
  // first forecast step; lead time = (forecast index + 1) * dt / 3600. We
  // emit lead hours starting at +0 for the row immediately after spin-up so
  // the marker on the subplots aligns with the map slider.
  const forecastSlice = ts
    .map((t, i) => ({ t, i }))
    .filter(({ i }) => i >= spinupRows)
    .map(({ t, i }) => ({ leadH: (t - ts[Math.min(spinupRows, ts.length - 1)]) / 3600, i }));
  if (forecastSlice.length === 0) return null;
  const leadHours = forecastSlice.map((p) => p.leadH);
  const sliceStage = forecastSlice.map((p) => stage[p.i]);
  const slicePrecip = forecastSlice.map((p) => precip[p.i]);
  return {
    leadHours,
    stage: sliceStage,
    precipitation: slicePrecip,
    forecastStart: leadHours[0] ?? 0,
    forecastEnd: leadHours[leadHours.length - 1] ?? 0,
  };
}

// ---------------------------------------------------------------------------
// Synced sub-plots for the Time Player.
// Both share the same x-axis (lead hours) and accept a `markerT` prop driven
// by the player's current scrub position.
// ---------------------------------------------------------------------------

function StageLineChart({ series, markerT }: { series: ForcingSeries | null; markerT: number | null }) {
  if (!series || series.leadHours.length < 2) {
    return <p className="player-chart-empty">Stage hydrograph unavailable.</p>;
  }
  const width = 620;
  const height = 184;
  const padL = 64;
  const padR = 24;
  const padT = 32;
  const padB = 48;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const tMin = series.forecastStart;
  const tMax = series.forecastEnd;
  const rawYMin = Math.min(...series.stage);
  const rawYMax = Math.max(...series.stage);
  const yPad = Math.max((rawYMax - rawYMin) * 0.08, 0.01);
  const yTicks = niceTicks(rawYMin - yPad, rawYMax + yPad, 5);
  const yMin = yTicks[0];
  const yMax = yTicks[yTicks.length - 1];
  const xTicks = Array.from(new Set([
    tMin,
    ...niceTicks(tMin, tMax, 5).filter((tick) => tick >= tMin - 1e-9 && tick <= tMax + 1e-9),
    tMax,
  ])).sort((a, b) => a - b);
  const ySpan = Math.max(yMax - yMin, 1e-6);
  const tSpan = Math.max(tMax - tMin, 1e-9);
  const xFor = (t: number) => padL + ((t - tMin) / tSpan) * innerW;
  const yFor = (y: number) => padT + (1 - (y - yMin) / ySpan) * innerH;
  const path = series.leadHours
    .map((t, i) => `${i === 0 ? "M" : "L"} ${xFor(t).toFixed(2)} ${yFor(series.stage[i]).toFixed(2)}`)
    .join(" ");
  const markerIndex =
    markerT !== null && Number.isFinite(markerT)
      ? series.leadHours.reduce((best, t, i) => (Math.abs(t - markerT) < Math.abs(series.leadHours[best] - markerT) ? i : best), 0)
      : null;
  return (
    <svg className="player-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Coastal water-level input">
      <rect x={0} y={0} width={width} height={height} fill="#ffffff" />
      <text x={padL} y={17} fontSize="13" fontWeight={700} fill="#102027">
        Coastal stage forcing
      </text>
      <text x={width - padR} y={17} fontSize="11" fill="#51666f" textAnchor="end">
        units: m
      </text>
      <rect x={padL} y={padT} width={innerW} height={innerH} fill="#ffffff" stroke="#aebfc7" />
      {xTicks.map((tick) => (
        <g key={`x-stage-${tick}`}>
          <line x1={xFor(tick)} x2={xFor(tick)} y1={padT} y2={padT + innerH} stroke="#eef2f4" />
          <text x={xFor(tick)} y={padT + innerH + 18} fontSize="10" textAnchor="middle" fill="#51666f">
            {formatAxisTick(tick)}
          </text>
        </g>
      ))}
      {yTicks.map((tick) => (
        <g key={`y-stage-${tick}`}>
          <line x1={padL} x2={padL + innerW} y1={yFor(tick)} y2={yFor(tick)} stroke="#e6ecef" />
          <text x={padL - 8} y={yFor(tick) + 3.2} fontSize="10" textAnchor="end" fill="#51666f">
            {formatAxisTick(tick)}
          </text>
        </g>
      ))}
      <line x1={padL} x2={padL + innerW} y1={padT + innerH} y2={padT + innerH} stroke="#102027" strokeWidth={1.1} />
      <line x1={padL} x2={padL} y1={padT} y2={padT + innerH} stroke="#102027" strokeWidth={1.1} />
      <path d={path} fill="none" stroke="#0868ac" strokeWidth={2.6} strokeLinecap="round" strokeLinejoin="round" />
      {markerT !== null && Number.isFinite(markerT) && markerT >= tMin && markerT <= tMax && (
        <>
          <line x1={xFor(markerT)} x2={xFor(markerT)} y1={padT} y2={padT + innerH} stroke="#b42318" strokeWidth={1.3} strokeDasharray="4 4" />
          {markerIndex !== null && (
            <circle cx={xFor(series.leadHours[markerIndex])} cy={yFor(series.stage[markerIndex])} r={3.8} fill="#ffffff" stroke="#b42318" strokeWidth={1.6} />
          )}
        </>
      )}
      <text x={padL + innerW / 2} y={height - 10} fontSize="11" textAnchor="middle" fill="#102027">
        Lead time (h)
      </text>
      <text x={17} y={padT + innerH / 2} fontSize="11" textAnchor="middle" fill="#102027" transform={`rotate(-90 17 ${padT + innerH / 2})`}>
        Stage (m)
      </text>
    </svg>
  );
}

function RainfallBars({ series, markerT }: { series: ForcingSeries | null; markerT: number | null }) {
  if (!series || series.leadHours.length < 1) {
    return <p className="player-chart-empty">Rainfall hydrograph unavailable.</p>;
  }
  const width = 620;
  const height = 184;
  const padL = 64;
  const padR = 24;
  const padT = 32;
  const padB = 48;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const tMin = series.forecastStart;
  const tMax = series.forecastEnd;
  const rawYMax = Math.max(...series.precipitation, 1e-3);
  const yTicks = niceTicks(0, rawYMax * 1.08, 5).filter((tick) => tick >= -1e-9);
  const yMax = Math.max(yTicks[yTicks.length - 1] ?? rawYMax, rawYMax);
  const xTicks = Array.from(new Set([
    tMin,
    ...niceTicks(tMin, tMax, 5).filter((tick) => tick >= tMin - 1e-9 && tick <= tMax + 1e-9),
    tMax,
  ])).sort((a, b) => a - b);
  const tSpan = Math.max(tMax - tMin, 1e-9);
  const xFor = (t: number) => padL + ((t - tMin) / tSpan) * innerW;
  const yFor = (y: number) => padT + (1 - y / yMax) * innerH;
  const barW = Math.max(2.2, (innerW / Math.max(series.leadHours.length, 1)) * 0.68);
  return (
    <svg className="player-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Rainfall input">
      <rect x={0} y={0} width={width} height={height} fill="#ffffff" />
      <text x={padL} y={17} fontSize="13" fontWeight={700} fill="#102027">
        Precipitation forcing
      </text>
      <text x={width - padR} y={17} fontSize="11" fill="#51666f" textAnchor="end">
        units: mm/step
      </text>
      <rect x={padL} y={padT} width={innerW} height={innerH} fill="#ffffff" stroke="#aebfc7" />
      {xTicks.map((tick) => (
        <g key={`x-rain-${tick}`}>
          <line x1={xFor(tick)} x2={xFor(tick)} y1={padT} y2={padT + innerH} stroke="#eef2f4" />
          <text x={xFor(tick)} y={padT + innerH + 18} fontSize="10" textAnchor="middle" fill="#51666f">
            {formatAxisTick(tick)}
          </text>
        </g>
      ))}
      {yTicks.map((tick) => (
        <g key={`y-rain-${tick}`}>
          <line x1={padL} x2={padL + innerW} y1={yFor(tick)} y2={yFor(tick)} stroke="#e6ecef" />
          <text x={padL - 8} y={yFor(tick) + 3.2} fontSize="10" textAnchor="end" fill="#51666f">
            {formatAxisTick(tick)}
          </text>
        </g>
      ))}
      {series.leadHours.map((t, i) => {
        const value = series.precipitation[i];
        if (!Number.isFinite(value) || value <= 0) return null;
        const x = xFor(t) - barW / 2;
        const y = yFor(value);
        const h = padT + innerH - y;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barW}
            height={Math.max(h, 0.5)}
            fill="#6a51a3"
            stroke="#3f2a75"
            strokeWidth={0.35}
            rx={0}
          />
        );
      })}
      <line x1={padL} x2={padL + innerW} y1={padT + innerH} y2={padT + innerH} stroke="#102027" strokeWidth={1.1} />
      <line x1={padL} x2={padL} y1={padT} y2={padT + innerH} stroke="#102027" strokeWidth={1.1} />
      {markerT !== null && Number.isFinite(markerT) && markerT >= tMin && markerT <= tMax && (
        <line x1={xFor(markerT)} x2={xFor(markerT)} y1={padT} y2={padT + innerH} stroke="#b42318" strokeWidth={1.3} strokeDasharray="4 4" />
      )}
      <text x={padL + innerW / 2} y={height - 10} fontSize="11" textAnchor="middle" fill="#102027">
        Lead time (h)
      </text>
      <text x={17} y={padT + innerH / 2} fontSize="11" textAnchor="middle" fill="#102027" transform={`rotate(-90 17 ${padT + innerH / 2})`}>
        Precipitation (mm/step)
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Time Player: scrubbable map + transport controls + synced forcing subplots.
// Uses the per-step calibrated mean PNGs (calibrated_mean_scrub_t{NNN}.png)
// generated by the worker in postprocessing. Falls back gracefully if the
// scrub frames aren't present yet (run still postprocessing, or older run).
// ---------------------------------------------------------------------------

type ScrubFrame = { artifactId: string; tIndex: number; leadHours: number };

// P3: contract with backend's ForecastProductBuilder.SCRUB_PRODUCTS. Keep
// in lockstep; the matcher regex below depends on these literal strings.
type ScrubProduct = "mean" | "spread" | "p_gt_0p30m";

const SCRUB_PRODUCT_META: Record<
  ScrubProduct,
  { label: string; short: string; description: string; unitTone: "depth" | "spread" | "prob" }
> = {
  mean: {
    label: "Expected water depth",
    short: "Expected depth",
    description: "Average water depth across the full group of plausible forecasts at this time.",
    unitTone: "depth",
  },
  spread: {
    label: "Forecast range",
    short: "Forecast range",
    description: "How far plausible water-depth forecasts spread apart at each location; wider spread means less agreement.",
    unitTone: "spread",
  },
  p_gt_0p30m: {
    label: "Chance depth passes 0.30 m",
    short: "Chance > 0.30 m",
    description: "Checked probability that water depth passes 0.30 m at each location and time.",
    unitTone: "prob",
  },
};

function buildScrubFrames(
  artifacts: Artifact[],
  leadHours: number[],
  product: ScrubProduct = "mean",
): ScrubFrame[] {
  // Anchored regex so we don't accidentally collide on substrings (e.g. a
  // future "mean_norm" product wouldn't be matched by the "mean" pattern).
  const re = new RegExp(`^calibrated_${product}_scrub_t(\\d{3})\\.png$`);
  const matches: ScrubFrame[] = [];
  for (const artifact of artifacts) {
    const m = artifact.artifact_id.match(re);
    if (!m) continue;
    const tIndex = parseInt(m[1], 10);
    if (!Number.isFinite(tIndex)) continue;
    const lead = leadHours[tIndex - 1];
    matches.push({
      artifactId: artifact.artifact_id,
      tIndex,
      leadHours: Number.isFinite(lead) ? lead : tIndex,
    });
  }
  return matches.sort((a, b) => a.tIndex - b.tIndex);
}

function buildScrubFramesByProduct(
  artifacts: Artifact[],
  leadHours: number[],
): Record<ScrubProduct, ScrubFrame[]> {
  return {
    mean: buildScrubFrames(artifacts, leadHours, "mean"),
    spread: buildScrubFrames(artifacts, leadHours, "spread"),
    p_gt_0p30m: buildScrubFrames(artifacts, leadHours, "p_gt_0p30m"),
  };
}

function TimePlayer({
  runId,
  runLabel,
  framesByProduct,
  fallbackImage,
  fallbackLabel,
}: {
  runId: string;
  runLabel: string;
  framesByProduct: Record<ScrubProduct, ScrubFrame[]>;
  fallbackImage: { artifactId: string; label: string } | null;
  fallbackLabel: string;
}) {
  // P3: product toggle. The Player auto-selects the first product that
  // actually has frames so that older runs (which only have "mean") still
  // work, and runs where one product failed silently still surface what's
  // available.
  const availableProducts = (Object.keys(framesByProduct) as ScrubProduct[]).filter(
    (key) => framesByProduct[key].length > 0,
  );
  const [product, setProduct] = useState<ScrubProduct>(
    availableProducts[0] ?? "mean",
  );
  const frames = framesByProduct[product] ?? [];
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const lastFrameCount = useRef(frames.length);

  // Re-home the product selection if frames arrive for what's currently a
  // missing product (e.g. user opened the page mid-postprocessing).
  useEffect(() => {
    if (framesByProduct[product].length === 0 && availableProducts.length > 0) {
      setProduct(availableProducts[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [framesByProduct.mean.length, framesByProduct.spread.length, framesByProduct.p_gt_0p30m.length]);

  // Keep idx in range when frame set changes (e.g. scrub frames arrive after
  // initial render once postprocessing finishes, or the user switches
  // products and the new product has a different count).
  useEffect(() => {
    if (frames.length !== lastFrameCount.current) {
      lastFrameCount.current = frames.length;
      setIdx((cur) => Math.min(cur, Math.max(frames.length - 1, 0)));
    }
  }, [frames.length]);

  // Auto-advance loop. setInterval ticks while `playing`; speed multiplier
  // adjusts the delay (1× = 350 ms, 2× = 175 ms, 0.5× = 700 ms).
  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const delay = Math.max(60, 350 / Math.max(speed, 0.25));
    const handle = setInterval(() => {
      setIdx((cur) => {
        const next = cur + 1;
        if (next >= frames.length) {
          setPlaying(false);
          return frames.length - 1;
        }
        return next;
      });
    }, delay);
    return () => clearInterval(handle);
  }, [playing, speed, frames.length]);

  if (frames.length === 0) {
    // Graceful fallback while scrub frames are still being rendered (or for
    // older runs that pre-date the scrub-frame contract).
    return (
      <div className="player">
        <header className="player-head">
          <div>
            <p className="player-eyebrow">Forecast animation</p>
            <h3 className="player-title">{runLabel}</h3>
          </div>
          <span className="chart-pill">{fallbackLabel}</span>
        </header>
        {fallbackImage ? (
          <img
            className="player-map"
            src={`/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(fallbackImage.artifactId)}`}
            alt={fallbackImage.label}
          />
        ) : (
          <div className="player-empty">Frame-by-frame animation appears once postprocessing finishes.</div>
        )}
      </div>
    );
  }

  const current = frames[Math.min(idx, frames.length - 1)];
  const markerT = current?.leadHours ?? null;
  const setSafe = (next: number) => setIdx(Math.max(0, Math.min(frames.length - 1, next)));

  return (
    <div className="player">
      <header className="player-head">
        <div>
          <p className="player-eyebrow">Forecast animation</p>
          <h3 className="player-title">
            {runLabel}{" "}
            <span className="player-sub">· checked {SCRUB_PRODUCT_META[product].label.toLowerCase()}</span>
          </h3>
          <p className="player-product-hint">{SCRUB_PRODUCT_META[product].description}</p>
        </div>
        <div className="player-meta">
          <span className="player-time">
            lead <strong>+{markerT?.toFixed(2) ?? "0.00"} h</strong>
          </span>
          <span className="chart-pill calibrated">frame {current.tIndex} / {frames.length}</span>
        </div>
      </header>

      <div className="player-products" role="tablist" aria-label="Map product">
        {(Object.keys(framesByProduct) as ScrubProduct[]).map((key) => {
          const meta = SCRUB_PRODUCT_META[key];
          const count = framesByProduct[key].length;
          const disabled = count === 0;
          const active = key === product;
          return (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={active}
              className={`player-product ${active ? "active" : ""} ${disabled ? "disabled" : ""}`}
              disabled={disabled}
              onClick={() => setProduct(key)}
              title={disabled ? "No frames for this product" : meta.description}
            >
              {meta.short}
              <span className="player-product-count">{disabled ? "—" : count}</span>
            </button>
          );
        })}
      </div>

      <div className="player-stage">
        {/* Preload neighbouring frames so scrubbing feels instant. */}
        {frames
          .slice(Math.max(0, idx - 1), Math.min(frames.length, idx + 2))
          .map((frame) => (
            <link
              key={frame.artifactId}
              rel="preload"
              as="image"
              href={`/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(frame.artifactId)}`}
            />
          ))}
        <img
          className="player-map"
          key={current.artifactId}
          src={`/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(current.artifactId)}`}
          alt={`Calibrated ${SCRUB_PRODUCT_META[product].label} at lead ${markerT?.toFixed(2) ?? "0.00"} hours for ${runLabel}`}
        />
      </div>

      <div className="player-controls">
        <div className="player-transport" role="group" aria-label="Animation transport">
          <button type="button" onClick={() => setSafe(0)} aria-label="Jump to start" title="Jump to start">⏮</button>
          <button type="button" onClick={() => setSafe(idx - 1)} aria-label="Step back" title="Step back">◀</button>
          <button
            type="button"
            className="player-play"
            onClick={() => setPlaying((p) => !p)}
            aria-label={playing ? "Pause" : "Play"}
            aria-pressed={playing}
          >
            {playing ? "⏸" : "▶"}
          </button>
          <button type="button" onClick={() => setSafe(idx + 1)} aria-label="Step forward" title="Step forward">▶</button>
          <button type="button" onClick={() => setSafe(frames.length - 1)} aria-label="Jump to end" title="Jump to end">⏭</button>
        </div>
        <input
          className="player-scrub"
          type="range"
          min={0}
          max={frames.length - 1}
          step={1}
          value={idx}
          onChange={(event) => setIdx(Number(event.target.value))}
          style={{ ["--scrub-pct" as string]: `${(idx / Math.max(frames.length - 1, 1)) * 100}%` }}
          aria-label="Scrub to frame"
          aria-valuenow={current.tIndex}
          aria-valuemin={1}
          aria-valuemax={frames.length}
        />
        <label className="player-speed">
          <span>Speed</span>
          <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))} aria-label="Playback speed">
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </label>
      </div>
      {/* Forcing subplots intentionally not duplicated here — the canonical
         post-result forcing visualization lives in the Drivers tab on
         /runs/[runId], where stage and rainfall are shown alongside the
         response curves on a shared time axis. Keeping them here as well
         was redundant and made the page heavier without adding insight. */}
    </div>
  );
}

function buildScenarioCsv(bundle: Bundle | null, scenario: SampleScenario) {
  const dt = bundle?.dt_seconds ?? scenario.dtSeconds;
  const spinup = scenarioSpinupRows(bundle);
  const forecastSteps = scenarioForecastSteps(bundle, scenario);
  const rows = Math.min(spinup + forecastSteps, scenario.stage.length, scenario.precipitation.length);
  const lines = ["time_seconds,stage,precipitation"];
  for (let i = 0; i < rows; i += 1) {
    const stage = Math.max(-20, Math.min(20, scenario.stage[i] ?? 0));
    const precipitation = Math.max(0, Math.min(500, scenario.precipitation[i] ?? 0));
    lines.push(`${i * dt},${stage.toFixed(4)},${precipitation.toFixed(4)}`);
  }
  return { csv: `${lines.join("\n")}\n`, forecastSteps };
}

export default function Page() {
  const [workspaceMode, setWorkspaceMode] = useState<HomeWorkspace>("new");
  const [authState, setAuthState] = useState<AuthState>("checking");
  const [user, setUser] = useState<User | null>(null);
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<RunRow | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [activeMapId, setActiveMapId] = useState<string | null>(null);
  const [calibratedSummary, setCalibratedSummary] = useState<Record<string, unknown> | null>(null);
  const [comparisonSummary, setComparisonSummary] = useState<Record<string, unknown> | null>(null);
  const [selectedForDelete, setSelectedForDelete] = useState<Set<string>>(() => new Set());
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteReport, setDeleteReport] = useState<null | { deleted: string[]; skipped: { run_id: string; reason: string; detail?: string }[] }>(null);
  const [file, setFile] = useState<File | null>(null);
  const [forcingPreview, setForcingPreview] = useState<ForcingPoint[]>([]);
  const [label, setLabel] = useState("");
  const [forecastSteps, setForecastSteps] = useState("");
  const [thresholds, setThresholds] = useState("0.01,0.05,0.1,0.3,0.5");
  const [ensembleCount, setEnsembleCount] = useState("3");
  const [membersPerEnsemble, setMembersPerEnsemble] = useState("20");
  const [requestAnimation, setRequestAnimation] = useState(true);
  const [requestFullHdf5] = useState(true);
  const [validation, setValidation] = useState<ValidationState | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const restoredDraftRef = useRef(false);

  useEffect(() => {
    const syncWorkspaceMode = () => setWorkspaceMode(workspaceFromLocation());
    syncWorkspaceMode();
    window.addEventListener("popstate", syncWorkspaceMode);
    window.addEventListener("hashchange", syncWorkspaceMode);
    return () => {
      window.removeEventListener("popstate", syncWorkspaceMode);
      window.removeEventListener("hashchange", syncWorkspaceMode);
    };
  }, []);

  const navigateWorkspace = useCallback((nextWorkspace: HomeWorkspace) => {
    setWorkspaceMode(nextWorkspace);
    if (typeof window === "undefined") return;

    const targetId = nextWorkspace === "runs" ? "runs" : "new-run";
    const nextUrl = nextWorkspace === "runs"
      ? "/demo?workspace=runs#runs"
      : "/demo?workspace=new#new-run";
    window.history.pushState({ workspace: nextWorkspace }, "", nextUrl);
    window.requestAnimationFrame(() => {
      document.getElementById(targetId)?.scrollIntoView({ block: "start" });
    });
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function bootstrapGuestOrUserSession() {
      try {
        const [sessionRes, bundleRes] = await Promise.all([
          fetch("/oauth2/auth", { cache: "no-store" }),
          fetch("/api/model-bundle", { cache: "no-store" }),
        ]);
        if (cancelled) return;
        if (bundleRes.ok) setBundle(await bundleRes.json());
        setAuthState(sessionRes.ok ? "authenticated" : "guest");
      } catch (exc) {
        if (cancelled) return;
        setAuthState("guest");
        setMessage(`The public console loaded, but session status could not be checked: ${String(exc)}`);
      }
    }
    bootstrapGuestOrUserSession();
    return () => {
      cancelled = true;
    };
  }, []);

  const refresh = useCallback(async () => {
    if (authState !== "authenticated") return;
    const [meRes, bundleRes, runsRes] = await Promise.all([
      fetch("/api/me", { cache: "no-store", redirect: "manual" }),
      fetch("/api/model-bundle", { cache: "no-store" }),
      fetch("/api/runs", { cache: "no-store", redirect: "manual" }),
    ]);
    if (bundleRes.ok) setBundle(await bundleRes.json());
    if (!meRes.ok) {
      setUser(null);
      setRuns([]);
      if (meRes.status === 401 || meRes.status === 0) setAuthState("guest");
      else setMessage(`Signed in, but compute access returned HTTP ${meRes.status}.`);
      return;
    }
    setUser(await meRes.json());
    if (runsRes.ok) {
      const nextRuns: RunRow[] = await runsRes.json();
      setRuns(nextRuns);
      // Auto-select the most recent run on first load, but never overwrite an
      // explicit user selection. Functional setState so this handler can stay
      // dependency-free and the polling effect doesn't churn.
      setSelectedRunId((prev) => prev ?? nextRuns[0]?.run_id ?? null);
    }
  }, [authState]);

  useEffect(() => {
    if (!bundle) return;
    setEnsembleCount((current) => current || String(bundle.n_checkpoints));
    setMembersPerEnsemble((current) => current || String(bundle.members_per_checkpoint));
  }, [bundle]);

  useEffect(() => {
    if (authState !== "guest") return;
    setUser(null);
    setRuns([]);
    setSelectedRunId(null);
    setSelectedRun(null);
    setArtifacts([]);
    setActiveMapId(null);
    setCalibratedSummary(null);
    setComparisonSummary(null);
    setSelectedForDelete(new Set());
  }, [authState]);

  const loadRun = useCallback(async (runId: string) => {
    const [runRes, artifactRes] = await Promise.all([
      fetch(`/api/runs/${encodeURIComponent(runId)}`, { cache: "no-store" }),
      fetch(`/api/runs/${encodeURIComponent(runId)}/artifacts`, { cache: "no-store" }),
    ]);
    if (!runRes.ok) throw new Error(`Run ${runId} returned HTTP ${runRes.status}`);
    const runPayload = await runRes.json();
    setSelectedRun(runPayload);
    if (artifactRes.ok) {
      const nextArtifacts: Artifact[] = await artifactRes.json();
      setArtifacts(nextArtifacts);
      const preferred = preferredMapId(nextArtifacts);
      setActiveMapId((current) =>
        current && nextArtifacts.some((artifact) => artifact.artifact_id === current) ? current : preferred,
      );
      if (nextArtifacts.some((artifact) => artifact.artifact_id === "calibrated_summary.json")) {
        const summaryRes = await fetch(
          `/api/runs/${encodeURIComponent(runId)}/artifacts/calibrated_summary.json`,
          { cache: "no-store" },
        );
        setCalibratedSummary(summaryRes.ok ? await summaryRes.json() : null);
      } else {
        setCalibratedSummary(null);
      }
      if (nextArtifacts.some((artifact) => artifact.artifact_id === "comparison_summary.json")) {
        const comparisonRes = await fetch(
          `/api/runs/${encodeURIComponent(runId)}/artifacts/comparison_summary.json`,
          { cache: "no-store" },
        );
        setComparisonSummary(comparisonRes.ok ? await comparisonRes.json() : null);
      } else {
        setComparisonSummary(null);
      }
    } else {
      setArtifacts([]);
      setActiveMapId(null);
      setCalibratedSummary(null);
      setComparisonSummary(null);
    }
  }, []);

  // Note: this page no longer renders the forcing series itself. The
  // canonical forcing visualization lives in the Drivers tab on
  // /runs/[runId]; fetching forcing.csv on every selection here would be a
  // wasted HTTP cycle, so the fetch + state are deliberately omitted.

  // Refs let the single polling loop read the latest selection / status without
  // restarting the interval on every state change (which is what caused the
  // previous double-poll: each effect re-created its own interval).
  const selectedRunIdRef = useRef<string | null>(selectedRunId);
  const runsRef = useRef<RunRow[]>(runs);
  const selectedRunRef = useRef<RunRow | null>(selectedRun);
  useEffect(() => {
    selectedRunIdRef.current = selectedRunId;
  }, [selectedRunId]);
  useEffect(() => {
    runsRef.current = runs;
  }, [runs]);
  useEffect(() => {
    selectedRunRef.current = selectedRun;
  }, [selectedRun]);

  // Single adaptive polling loop:
  // - one HTTP cycle per tick (list + selected detail when one is selected),
  // - 4 s while anything is active, 20 s when everything terminal,
  // - paused while the tab is hidden, resumed immediately on focus.
  // Recursive setTimeout (not setInterval) so we can recompute the cadence
  // after each tick without restarting the effect.
  useEffect(() => {
    if (authState !== "authenticated") return;
    let cancelled = false;
    let handle: ReturnType<typeof setTimeout> | null = null;

    const intervalMs = () => {
      const list = runsRef.current;
      const sel = selectedRunRef.current;
      const hasActive =
        list.some((run) => ACTIVE_STATUSES.has(run.status)) ||
        (sel ? ACTIVE_STATUSES.has(sel.status) : false);
      return hasActive ? POLL_INTERVAL_ACTIVE_MS : POLL_INTERVAL_IDLE_MS;
    };

    const tick = async () => {
      if (cancelled) return;
      if (typeof document !== "undefined" && document.hidden) {
        // Re-arm cheaply; visibilitychange will wake us up the moment the tab
        // becomes visible. The 30 s ceiling is just a safety net in case the
        // browser swallows the visibilitychange event.
        handle = setTimeout(tick, 30000);
        return;
      }
      try {
        await refresh();
        const id = selectedRunIdRef.current;
        if (id) await loadRun(id);
      } catch {
        // Errors surface to the per-action UI; the poll loop must not die so
        // that transient failures (server restart, brief 5xx) self-recover.
      }
      if (cancelled) return;
      handle = setTimeout(tick, intervalMs());
    };

    tick();

    const onVisibility = () => {
      if (typeof document === "undefined" || document.hidden) return;
      if (handle) clearTimeout(handle);
      tick();
    };
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", onVisibility);
    }

    return () => {
      cancelled = true;
      if (handle) clearTimeout(handle);
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", onVisibility);
      }
    };
  }, [authState, refresh, loadRun]);

  // Side effect of changing selection: clear stale per-run state, then prime
  // the new run's data immediately so users don't wait for the next poll tick.
  useEffect(() => {
    if (!selectedRunId) {
      setSelectedRun(null);
      setArtifacts([]);
      setActiveMapId(null);
      setCalibratedSummary(null);
      setComparisonSummary(null);
      return;
    }
    setActiveMapId(null);
    setCalibratedSummary(null);
    setComparisonSummary(null);
    loadRun(selectedRunId).catch((exc) => setMessage(`Could not open run: ${String(exc)}`));
  }, [loadRun, selectedRunId]);

  // Validates the uploaded forcing CSV against the current bundle. Surfacing
  // backend failures explicitly is critical: the previous implementation
  // silently swallowed non-OK responses, so a failing validator looked exactly
  // like a not-yet-validated file and users would submit anyway.
  const validateFile = useCallback(
    async (nextFile: File | null, nextSteps = forecastSteps) => {
      setValidation(null);
      if (!nextFile) return;
      parseForcingPreview(nextFile)
        .then(setForcingPreview)
        .catch(() => setForcingPreview([]));
      const form = new FormData();
      form.append("file", nextFile);
      if (nextSteps.trim()) form.append("forecast_steps", nextSteps.trim());
      try {
        const res = await fetch("/api/forcing/validate", { method: "POST", body: form });
        if (res.ok) {
          setValidation(await res.json());
          return;
        }
        const detail = await res.json().catch(() => ({}));
        const message =
          (typeof detail.detail === "string" && detail.detail) ||
          `Validator returned HTTP ${res.status}`;
        setValidation({ valid: false, messages: [message], summary: null });
      } catch (exc) {
        const message = exc instanceof Error ? exc.message : String(exc);
        setValidation({
          valid: false,
          messages: [`Could not reach validator: ${message}`],
          summary: null,
        });
      }
    },
    [forecastSteps],
  );

  useEffect(() => {
    if (authState !== "authenticated" || restoredDraftRef.current) return;
    restoredDraftRef.current = true;
    const raw = window.sessionStorage.getItem(GUEST_SUBMISSION_DRAFT_KEY);
    if (!raw) return;
    window.sessionStorage.removeItem(GUEST_SUBMISSION_DRAFT_KEY);
    const draft = parseGuestSubmissionDraft(raw);
    if (!draft) {
      setMessage("The saved guest scenario could not be restored. Please select the CSV again.");
      return;
    }
    const restoredFile = new File([draft.csv], draft.fileName, { type: draft.fileType });
    setFile(restoredFile);
    setLabel(draft.label);
    setForecastSteps(draft.forecastSteps);
    setThresholds(draft.thresholds);
    setEnsembleCount(draft.ensembleCount);
    setMembersPerEnsemble(draft.membersPerEnsemble);
    setRequestAnimation(draft.requestAnimation);
    setMessage("Signed in. Your scenario is restored; review the settings and launch the analysis.");
    validateFile(restoredFile, draft.forecastSteps).catch(() => undefined);
  }, [authState, validateFile]);

  // Lifted before any artifact derivations so scrubFrames can use it. The
  // worker emits one calibrated_mean_scrub_t{NNN}.png per forecast step, and
  // we need the matching lead-time array to label and align the scrubber.
  const leadHoursArray: number[] = Array.isArray(calibratedSummary?.lead_time_hours)
    ? (calibratedSummary?.lead_time_hours as number[])
    : [];

  // Static map gallery — keep the slot-based PNGs for product comparison
  // (mean / spread / p95 / exceedance) but exclude the dense scrub frames so
  // the gallery doesn't get drowned by them.
  const galleryArtifacts = useMemo(
    () =>
      artifacts
        .filter((a) => a.artifact_id.endsWith(".png") && !/_mean_scrub_t\d{3}\.png$/.test(a.artifact_id))
        .sort((a, b) => a.artifact_id.localeCompare(b.artifact_id)),
    [artifacts],
  );
  const animationArtifacts = useMemo(
    () => artifacts.filter((artifact) => artifact.artifact_id.endsWith(".gif")).sort((a, b) => a.artifact_id.localeCompare(b.artifact_id)),
    [artifacts],
  );
  const activeMap = activeMapId ?? galleryArtifacts[0]?.artifact_id ?? null;
  const primaryAnimation =
    animationArtifacts.find((artifact) => artifact.artifact_id.startsWith("calibrated_p_gt_")) ??
    animationArtifacts.find((artifact) => artifact.artifact_id.startsWith("calibrated_mean_")) ??
    animationArtifacts[0] ??
    null;
  const scrubFramesByProduct = useMemo(
    () => buildScrubFramesByProduct(artifacts, leadHoursArray),
    [artifacts, leadHoursArray],
  );
  // Backwards-compat aliases so older call sites and the fallback chain
  // below still read the cheap "is anything in the player at all?" signal.
  const scrubFrames = scrubFramesByProduct.mean;
  const latestRun = runs[0];
  const runningCount = runs.filter((run) => ["RUNNING", "POSTPROCESSING"].includes(run.status)).length;
  const queuedCount = runs.filter((run) => ["SUBMITTED", "VALIDATING", "WAITING_FOR_CACHE", "QUEUED"].includes(run.status)).length;
  // UQ snapshot derivation. Completed-run cards are intentionally area- and
  // probability-first; legacy cell-count fields remain only for old JSON.
  const peakDepthM = asNumber(calibratedSummary?.max_mean_wd_m);
  const peakFloodAreaKm2 = asNumber(calibratedSummary?.["peak_expected_flooded_area_km2_gt_0.05m"]);
  const peakFloodAreaFraction = asNumber(calibratedSummary?.["peak_expected_flooded_area_fraction_wettable_gt_0.05m"]);
  const peakFloodAreaLead = asNumber(calibratedSummary?.["peak_expected_flooded_area_lead_hours_gt_0.05m"]);
  const onsetLeadHours = asNumber(calibratedSummary?.["onset_lead_hours_expected_flooded_area_fraction_gt_1pct_gt_0.05m"]);
  const uncertaintyWidthM = asNumber(calibratedSummary?.peak_area_weighted_iqr_wd_m);
  const central90WidthM = asNumber(calibratedSummary?.peak_area_weighted_central_90_wd_m);
  const uncertaintyToSignal = asNumber(calibratedSummary?.uncertainty_to_signal_ratio);
  const calibrationShiftPct = asNumber(comparisonSummary?.["delta_peak_expected_flooded_area_fraction_wettable_percentage_points_gt_0.05m"]);
  const calibrationApplied = Boolean(calibratedSummary?.isotonic_calibration_applied);
  const extentByTime = Array.isArray(calibratedSummary?.["expected_flooded_area_fraction_wettable_by_time_gt_0.05m"])
    ? (calibratedSummary?.["expected_flooded_area_fraction_wettable_by_time_gt_0.05m"] as number[])
    : [];
  const iqrByTime = Array.isArray(calibratedSummary?.area_weighted_iqr_wd_m_by_time)
    ? (calibratedSummary?.area_weighted_iqr_wd_m_by_time as number[])
    : [];
  const extentSeries: TimeSeriesPoint[] = leadHoursArray.map((t, i) => ({ t, y: (extentByTime[i] ?? NaN) * 100 }));
  const iqrSeries: TimeSeriesPoint[] = leadHoursArray.map((t, i) => ({ t, y: iqrByTime[i] ?? NaN }));
  const exceedanceArea = exceedanceAreaEntries(calibratedSummary);
  const primaryExceedanceArea = preferredExceedanceAreaEntry(exceedanceArea);
  const figureArtifacts = [
    { id: "forcing_hydrograph.svg", title: "Water-Level and Rainfall Inputs" },
    { id: "uq_extent_by_time.svg", title: "Expected Flooded-Area Fraction" },
    { id: "uq_exceedance_bars.svg", title: "Peak Depth-Threshold Footprint" },
    { id: "uq_uncertainty_width.svg", title: "Forecast Range Through Time" },
    { id: "calibration_effect.svg", title: "Probability Adjustment" },
  ].filter((figure) => artifacts.some((artifact) => artifact.artifact_id === figure.id));

  async function acknowledgeDisclaimer() {
    setBusy(true);
    try {
      const res = await fetch("/api/me/disclaimer", { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setUser(await res.json());
      setMessage("Model-use notice acknowledged.");
    } catch (exc) {
      setMessage(`Could not acknowledge disclaimer: ${String(exc)}`);
    } finally {
      setBusy(false);
    }
  }

  async function applyScenario(scenario: SampleScenario) {
    const generated = buildScenarioCsv(bundle, scenario);
    const nextFile = new File([generated.csv], `${scenario.id}.csv`, { type: "text/csv" });
    setFile(nextFile);
    setLabel(scenario.name);
    setForecastSteps(String(generated.forecastSteps));
    setMessage(`${scenario.name} loaded. Review options, then submit.`);
    await validateFile(nextFile, String(generated.forecastSteps));
  }

  async function signInForSubmission() {
    if (!file) return;
    setBusy(true);
    try {
      const csv = await file.text();
      if (csv.length > MAX_GUEST_DRAFT_CSV_CHARS) {
        setMessage("This CSV is too large to preserve securely across sign-in. Reduce it below 2 MiB and try again.");
        setBusy(false);
        return;
      }
      window.sessionStorage.setItem(
        GUEST_SUBMISSION_DRAFT_KEY,
        JSON.stringify({
          version: GUEST_SUBMISSION_DRAFT_VERSION,
          fileName: file.name,
          fileType: file.type || "text/csv",
          csv,
          label,
          forecastSteps,
          thresholds,
          ensembleCount,
          membersPerEnsemble,
          requestAnimation,
        }),
      );
      window.location.assign(SUBMISSION_SIGN_IN_PATH);
    } catch (exc) {
      setMessage(`Could not preserve the scenario for sign-in: ${String(exc)}`);
      setBusy(false);
    }
  }

  async function submitRun() {
    if (!file) {
      setMessage("Choose or generate a scenario CSV first.");
      return;
    }
    if (validation && !validation.valid) {
      setMessage(`Fix validation errors first: ${validation.messages.join("; ")}`);
      return;
    }
    if (!validation) {
      setMessage("Wait for CSV validation to finish before continuing.");
      return;
    }
    if (authState === "checking") {
      setMessage("Checking sign-in status. Please try again in a moment.");
      return;
    }
    if (authState === "guest") {
      await signInForSubmission();
      return;
    }
    if (!user) {
      setMessage("Your Google session is active, but compute access is not available for this account.");
      return;
    }
    if (!user.disclaimer_acknowledged) {
      setMessage("Acknowledge the model-use notice before submitting.");
      return;
    }
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      if (label.trim()) form.append("label", label.trim());
      if (forecastSteps.trim()) form.append("forecast_steps", forecastSteps.trim());
      if (ensembleCount.trim()) form.append("ensemble_count", ensembleCount.trim());
      if (membersPerEnsemble.trim()) form.append("members_per_ensemble", membersPerEnsemble.trim());
      form.append("output_detail", "standard");
      form.append("exceedance_thresholds_m", thresholds);
      form.append("request_animation", String(requestAnimation));
      form.append("request_full_hdf5", "true");
      const res = await fetch("/api/runs", { method: "POST", body: form });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
      setMessage(`Submitted run ${payload.run_id}.`);
      setSelectedRunId(payload.run_id);
      setFile(null);
      setForcingPreview([]);
      setLabel("");
      setValidation(null);
      await refresh();
    } catch (exc) {
      setMessage(`Submission rejected: ${String(exc)}`);
    } finally {
      setBusy(false);
    }
  }

  async function cancelSelectedRun() {
    if (!selectedRunId) return;
    setBusy(true);
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(selectedRunId)}/cancel`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadRun(selectedRunId);
      await refresh();
    } catch (exc) {
      setMessage(`Could not cancel run: ${String(exc)}`);
    } finally {
      setBusy(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Multi-select delete: checkboxes on rows, batch POST, then refresh.
  // The checkbox cell stops click propagation so toggling for delete does
  // not also change the inspection selection.
  // ---------------------------------------------------------------------------
  function toggleForDelete(runId: string) {
    setSelectedForDelete((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  }

  function clearDeleteSelection() {
    setSelectedForDelete(new Set());
  }

  const deletableRuns = runs.filter((run) => TERMINAL_STATUSES.has(run.status) && run.status !== "DELETED");
  const selectableIds = new Set(deletableRuns.map((run) => run.run_id));
  // Trim selection if any of its members are no longer terminal (rare; keeps
  // state coherent if a stale selection survives a state transition).
  useEffect(() => {
    let drift = false;
    for (const id of selectedForDelete) if (!selectableIds.has(id)) drift = true;
    if (drift) {
      setSelectedForDelete(
        (prev) => new Set(Array.from(prev).filter((id) => selectableIds.has(id))),
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs]);

  const allDeletableChecked =
    deletableRuns.length > 0 && deletableRuns.every((run) => selectedForDelete.has(run.run_id));
  const someDeletableChecked = selectedForDelete.size > 0 && !allDeletableChecked;

  function toggleSelectAllDeletable() {
    if (allDeletableChecked) {
      clearDeleteSelection();
    } else {
      setSelectedForDelete(new Set(deletableRuns.map((run) => run.run_id)));
    }
  }

  async function performDelete() {
    if (selectedForDelete.size === 0) return;
    setDeleteBusy(true);
    try {
      const ids = Array.from(selectedForDelete);
      const res = await fetch("/api/runs/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: ids }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const report = (await res.json()) as { deleted: string[]; skipped: { run_id: string; reason: string; detail?: string }[] };
      setDeleteReport(report);
      // If the currently inspected run was deleted, clear selection.
      if (selectedRunId && report.deleted.includes(selectedRunId)) {
        setSelectedRunId(null);
      }
      // Remove deleted ids from the to-delete set; keep any skipped so the
      // user can see why and decide whether to retry.
      setSelectedForDelete((prev) => {
        const next = new Set(prev);
        for (const id of report.deleted) next.delete(id);
        return next;
      });
      await refresh();
      // Close the modal if everything succeeded; keep it open with the
      // report if anything was skipped so the user sees the reason.
      if (report.skipped.length === 0) {
        setDeleteModalOpen(false);
      }
    } catch (exc) {
      setMessage(`Delete failed: ${exc instanceof Error ? exc.message : String(exc)}`);
    } finally {
      setDeleteBusy(false);
    }
  }

  return (
    <AppShell
      active={workspaceMode === "runs" ? "runs" : "home"}
      userEmail={user?.email}
      guestMode={authState === "guest"}
    >
      <div className="shell">
        <PageHeader
          kicker="Coastal flood scenario workspace"
          title="Explore possible flooding and how closely forecasts agree."
          subtitle={
            bundle ? (
              <>
                Run a group of plausible coastal flood forecasts from water-level and rainfall inputs. Review expected
                depth, the chance of passing chosen depths, forecast agreement, and the record behind each result.
                <br />
                Active model: {bundle.domain_name} · up to {bundle.total_members} plausible forecasts · {bundle.max_forecast_steps}
                -step forecast horizon
                {bundle.initial_condition ? (
                  <>
                    <br />
                    Initial condition policy:{" "}
                    {bundle.initial_condition.default_mode === "forcing_conditioned_baseline"
                      ? `starting water depth selected from similar model-building cases (${bundle.initial_condition.k_neighbors ?? 5} neighbors)`
                      : "zero-water diagnostic starting state"}
                  </>
                ) : null}
              </>
            ) : (
              "Loading the model bundle, queue state, and governance controls."
            )
          }
          actions={
            user ? (
              <>
              <span>{user.email}</span>
              <a className="signout" href="/oauth2/sign_out">Sign out</a>
              </>
            ) : (
              <span>{authState === "checking" ? "Checking sign-in status" : "Guest exploration"}</span>
            )
          }
        />

        <section className="home-hero" aria-labelledby="home-hero-title">
          <div className="home-hero-copy">
            <p className="eyebrow">Coastal flood probability workspace</p>
            <h2 id="home-hero-title">From water-level and rainfall inputs to clear flood-probability maps.</h2>
            <p>
              FloodUQ checks a scenario file, generates a group of plausible forecasts, and presents expected depth,
              the chance of passing selected depths, timing, and forecast agreement. Each result keeps the technical
              record needed for review or later analysis.
            </p>
            <div className="hero-actions" aria-label="Primary actions">
              <button className="button primary" type="button" onClick={() => navigateWorkspace("new")}>
                Configure scenario <ArrowRight size={14} aria-hidden="true" />
              </button>
              <button className="button secondary" type="button" onClick={() => navigateWorkspace("runs")}>
                Review analyses
              </button>
            </div>
            <div className="hero-capabilities" aria-label="Platform capabilities">
              <span><Waves size={14} aria-hidden="true" /> Portsmouth coastal model</span>
              <span><Activity size={14} aria-hidden="true" /> Checked probabilities and forecast ranges</span>
              <span><Database size={14} aria-hidden="true" /> Traceable result files</span>
              <span><ShieldCheck size={14} aria-hidden="true" /> Managed compute access</span>
            </div>
          </div>
          <aside className="workflow-card" aria-label="Analysis workflow">
            <div className="workflow-card-head">
              <span>Analysis workflow</span>
              <strong>{bundle ? `${bundle.total_members} plausible forecasts` : "Model loading"}</strong>
            </div>
            <ol className="workflow-steps">
              <li>
                <span>01</span>
                <div>
                  <strong>Check scenario inputs</strong>
                  <p>Check water-level and rainfall format, forecast length, and similarity to model-building evidence.</p>
                </div>
              </li>
              <li>
                <span>02</span>
                <div>
                  <strong>Generate a range of forecasts</strong>
                  <p>Run the Portsmouth model with a controlled number of plausible outcomes and a stored technical record.</p>
                </div>
              </li>
              <li>
                <span>03</span>
                <div>
                  <strong>Review probability and agreement</strong>
                  <p>Open expected-depth maps, chance maps, animations, location traces, scenario checks, and downloads.</p>
                </div>
              </li>
            </ol>
          </aside>
        </section>

        <ResearchNotice title="Model use and governance.">
          Review outputs within the documented domain, calibration, validation, and data-quality context.
          Preserve expert review and local governance when using results for planning or asset evaluation.
          {authState === "guest" ? (
            <span>You can configure and validate a scenario now. Google sign-in is requested only when you launch shared compute.</span>
          ) : !user?.disclaimer_acknowledged && (
            <button type="button" onClick={acknowledgeDisclaimer} disabled={busy}>
              Acknowledge
            </button>
          )}
        </ResearchNotice>

        <section className="metric-grid home-metrics">
          {authState === "guest" ? (
            <>
              <MetricCard label="Access" value="Public" detail="Explore without connecting a Google account" icon={<Waves size={17} />} />
              <MetricCard label="Scenario setup" value="Available" detail="Load a representative event or upload a CSV" icon={<ListChecks size={17} />} />
              <MetricCard label="Input checks" value="Available" detail="Check format, forecast length, and similarity to model-building evidence" icon={<ShieldCheck size={17} />} />
              <MetricCard label="Shared compute" value="Sign in" detail="Authentication begins only when you launch" icon={<Cpu size={17} />} />
            </>
          ) : (
            <>
              <MetricCard label="Analyses" value={runs.length} detail="Runs in your analysis history" icon={<ListChecks size={17} />} />
              <MetricCard label="Queue" value={queuedCount} detail="Validated work waiting to start" icon={<RefreshCw size={17} />} />
              <MetricCard label="Active inference" value={runningCount} detail="One GPU job runs at a time" icon={<Cpu size={17} />} />
              <MetricCard
                label="Latest analysis"
                value={latestRun ? latestRun.status : "-"}
                detail={latestRun ? latestRun.label || latestRun.run_id.slice(0, 12) : "No runs yet"}
                icon={<Activity size={17} />}
              />
            </>
          )}
        </section>

      <section className={`workspace workspace-${workspaceMode}`} id={workspaceMode === "runs" ? "runs" : "new-run"}>
        {workspaceMode === "new" && (
        <aside className="panel input-panel">
          <div className="section-head">
            <div>
              <p className="eyebrow">Inputs</p>
              <h2>Scenario Configuration</h2>
            </div>
            <a className="button ghost" href="/api/forcing-template" download>
              Download template
            </a>
          </div>

          <div className="scenario-list" role="list">
            {SAMPLE_SCENARIOS.map((scenario) => {
              const scenarioSteps = scenarioForecastSteps(bundle, scenario);
              const durationHours = scenarioDurationHours(bundle, scenario);
              const stageMax = Math.max(...scenario.stage);
              const rainTotal = sumFinite(scenario.precipitation);
              return (
                <button
                  className="scenario"
                  type="button"
                  key={scenario.id}
                  onClick={() => applyScenario(scenario)}
                  disabled={busy}
                  aria-label={`Load ${scenario.name} scenario: ${scenario.description}`}
                >
                  <span className="scenario-kicker">{scenario.testCaseId} · preloaded historical inputs</span>
                  <strong>{scenario.name}</strong>
                  <span className="scenario-desc">{scenario.description}</span>
                  <span className="scenario-meta">
                    <span>{scenario.stage.length} rows</span>
                    <span>{scenarioSteps} forecast steps</span>
                    <span>{durationHours.toFixed(1)} h</span>
                    <span>peak water level {stageMax.toFixed(2)} m</span>
                    <span>total rain {rainTotal.toFixed(1)} mm</span>
                  </span>
                </button>
              );
            })}
          </div>

          <div className="form-grid">
            <label>
              Label
              <input value={label} onChange={(event) => setLabel(event.target.value)} placeholder="Optional run label" />
            </label>
            <label>
              Forecast steps
              <input
                value={forecastSteps}
                onChange={(event) => {
                  setForecastSteps(event.target.value);
                  validateFile(file, event.target.value).catch(() => undefined);
                }}
                placeholder="Default from CSV"
              />
            </label>
            <label>
              Independently trained models
              <input
                type="number"
                min={1}
                max={bundle?.n_checkpoints ?? 3}
                step={1}
                value={ensembleCount}
                onChange={(event) => setEnsembleCount(event.target.value)}
                aria-describedby="member-budget-note"
              />
            </label>
            <label>
              Forecasts per model
              <input
                type="number"
                min={1}
                max={bundle?.members_per_checkpoint ?? 20}
                step={1}
                value={membersPerEnsemble}
                onChange={(event) => setMembersPerEnsemble(event.target.value)}
                aria-describedby="member-budget-note"
              />
            </label>
            <p className="member-budget-note wide" id="member-budget-note">
              The full setting uses {bundle?.n_checkpoints ?? 3} trained models x {bundle?.members_per_checkpoint ?? 20} plausible forecasts = {bundle?.total_members ?? 60} forecasts. Smaller settings run faster for exploratory checks.
            </p>
            <label className="wide">
              Water-depth thresholds to evaluate (m)
              <input value={thresholds} onChange={(event) => setThresholds(event.target.value)} />
            </label>
          </div>

          <label className="file-picker">
            Upload CSV
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => {
                const next = event.target.files?.[0] ?? null;
                setFile(next);
                validateFile(next).catch(() => undefined);
              }}
            />
          </label>
          {file && <p className="file-name">{file.name}</p>}

          {forcingPreview.length > 1 && (
            <ForcingPreviewPanel points={forcingPreview} bundle={bundle} forecastStepsText={forecastSteps} />
          )}

          <div className="checks">
            <label><input type="checkbox" checked={requestAnimation} onChange={(event) => setRequestAnimation(event.target.checked)} /> Save animated map (GIF)</label>
            <label><input type="checkbox" checked={requestFullHdf5} disabled readOnly /> Store all plausible forecasts for location inspection</label>
          </div>

          {validation && (
            <div className={validation.valid ? "valid" : "invalid"}>
              <strong>{validation.valid ? "Scenario file validated" : "Scenario validation failed"}</strong>
              {validation.messages.length > 0 && <span>{validation.messages.join("; ")}</span>}
              {validation.summary && (
                <small>
                  {String(validation.summary.n_rows)} rows - {String(validation.summary.forecast_steps)} forecast steps
                </small>
              )}
              {validation.screening?.available && (
                <div className="screening-card">
                  <strong>
                    Scenario familiarity check:{" "}
                    {validation.screening.candidate_recommended
                      ? "additional review recommended"
                      : (validation.screening.flags?.length ?? 0) > 0
                        ? "differences noted for context"
                        : "similar to the model-building evidence"}
                  </strong>
                  <small>
                    Familiarity score{" "}
                    {typeof validation.screening.input_novelty_score === "number"
                      ? validation.screening.input_novelty_score.toFixed(2)
                      : "—"}{" "}
                    · compared with the current coastal evidence set
                  </small>
                  {(validation.screening.flags ?? []).slice(0, 3).map((flag) => (
                    <small key={`${flag.code}-${flag.descriptor ?? ""}`}>{formatScreeningFlag(flag)}</small>
                  ))}
                </div>
              )}
            </div>
          )}

          <button
            className="button primary wide-button"
            type="button"
            onClick={submitRun}
            disabled={
              busy ||
              !file ||
              !validation?.valid ||
              authState === "checking" ||
              (authState === "authenticated" && !user)
            }
          >
            {authState === "authenticated"
              ? user
                ? "Launch Analysis"
                : "Loading account access"
              : "Sign in to launch analysis"}
          </button>
          {message && <p className="message">{message}</p>}
        </aside>
        )}

        {workspaceMode === "runs" && (
        <section className="panel run-panel" id="runs">
          <div className="section-head">
            <div>
              <p className="eyebrow">Operations</p>
              <h2>Analysis Queue</h2>
            </div>
            <div className="queue-actions">
              {deletableRuns.length > 0 && (
                <label className="select-all" title="Select every terminal run for deletion">
                  <input
                    type="checkbox"
                    checked={allDeletableChecked}
                    ref={(node) => {
                      if (node) node.indeterminate = someDeletableChecked;
                    }}
                    onChange={toggleSelectAllDeletable}
                    aria-label="Select all completed runs"
                  />
                  All
                </label>
              )}
              {authState === "authenticated" ? (
                <button type="button" className="button ghost" onClick={() => refresh()} disabled={busy}>
                  Refresh
                </button>
              ) : null}
            </div>
          </div>

          {selectedForDelete.size > 0 && (
            <div className="delete-toolbar" role="region" aria-label="Bulk delete">
              <span>
                <strong>{selectedForDelete.size}</strong> selected for deletion
              </span>
              <div className="delete-toolbar-actions">
                <button type="button" className="button ghost" onClick={clearDeleteSelection}>
                  Clear
                </button>
                <button
                  type="button"
                  className="button danger"
                  onClick={() => {
                    setDeleteReport(null);
                    setDeleteModalOpen(true);
                  }}
                  disabled={deleteBusy}
                >
                  Delete {selectedForDelete.size}…
                </button>
              </div>
            </div>
          )}

          <ul className="run-list" role="list">
            {runs.map((run) => {
              const isSelected = selectedRunId === run.run_id;
              const isChecked = selectedForDelete.has(run.run_id);
              const isDeletable = selectableIds.has(run.run_id);
              const progressPct = Math.round((run.progress ?? 0) * 100);
              return (
                <li
                  key={run.run_id}
                  className={`run-row ${isSelected ? "selected" : ""} ${isChecked ? "queued-for-delete" : ""}`}
                >
                  <label
                    className="row-check"
                    title={isDeletable ? "Mark for deletion" : "Only terminal runs can be deleted; cancel active runs first."}
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      disabled={!isDeletable}
                      onChange={() => toggleForDelete(run.run_id)}
                      aria-label={`Mark run ${run.label || run.run_id} for deletion`}
                    />
                  </label>
                  <button
                    type="button"
                    className="run-row-body"
                    onClick={() => setSelectedRunId(run.run_id)}
                    aria-pressed={isSelected}
                    aria-label={`Run ${run.label || run.run_id}, status ${run.status}, ${progressPct}% complete`}
                  >
                  <div className="run-row-text">
                    <strong>{run.label || run.run_id.slice(0, 12)}</strong>
                    <span>{new Date(run.created_at).toLocaleString()}</span>
                    {run.cache?.materialized_from_cache && <span className="run-progress-label">Loaded from verified cache</span>}
                    {run.cache?.waiting_for_cached_result && <span className="run-progress-label">Waiting for matching run to finish</span>}
                    {run.progress_label && <span className="run-progress-label">{run.progress_label}</span>}
                  </div>
                    <span className={`status ${statusTone(run.status)}`}>{run.status}</span>
                    <div
                      className="mini-progress"
                      role="progressbar"
                      aria-valuenow={progressPct}
                      aria-valuemin={0}
                      aria-valuemax={100}
                    >
                      <span style={{ width: `${progressPct}%` }} />
                    </div>
                  </button>
                  <a className="run-open-link" href={`/demo/runs/${encodeURIComponent(run.run_id)}`}>
                    Open
                  </a>
                </li>
              );
            })}
            {authState === "guest" ? (
              <li className="empty guest-queue-empty">
                <strong>Your private analysis queue appears after Google sign-in.</strong>
                <span>
                  Configure and validate scenarios without signing in. Authentication begins only when you launch
                  shared compute.
                </span>
              </li>
            ) : runs.length === 0 ? (
              <li className="empty">No analyses yet. Load a representative scenario or upload a scenario CSV to begin.</li>
            ) : null}
          </ul>
        </section>
        )}
      </section>

      {deleteModalOpen && (
        <div
          className="modal-backdrop"
          onClick={(event) => {
            if (event.target === event.currentTarget && !deleteBusy) setDeleteModalOpen(false);
          }}
          role="dialog"
          aria-modal="true"
          aria-labelledby="delete-modal-title"
        >
          <div className="modal">
            <header className="modal-head">
              <h2 id="delete-modal-title">Delete {selectedForDelete.size} run{selectedForDelete.size === 1 ? "" : "s"}?</h2>
              <p>
                This permanently removes the selected runs from your history and deletes every
                artifact they produced (maps, animations, summaries, raw ensemble files). This
                action cannot be undone.
              </p>
            </header>
            <ul className="modal-list">
              {Array.from(selectedForDelete).slice(0, 12).map((id) => {
                const run = runs.find((r) => r.run_id === id);
                return (
                  <li key={id}>
                    <strong>{run?.label || id.slice(0, 12)}</strong>
                    <span>{run ? new Date(run.created_at).toLocaleString() : ""}</span>
                    <span className={`status ${run ? statusTone(run.status) : "waiting"}`}>{run?.status ?? "?"}</span>
                  </li>
                );
              })}
              {selectedForDelete.size > 12 && (
                <li className="modal-overflow">+{selectedForDelete.size - 12} more…</li>
              )}
            </ul>
            {deleteReport && deleteReport.skipped.length > 0 && (
              <div className="modal-warning" role="status">
                <strong>{deleteReport.deleted.length}</strong> deleted ·{" "}
                <strong>{deleteReport.skipped.length}</strong> skipped:
                <ul>
                  {deleteReport.skipped.slice(0, 6).map((row) => (
                    <li key={row.run_id}>
                      <code>{row.run_id.slice(0, 8)}…</code> {row.reason === "active" ? "still active — cancel first" : row.reason === "forbidden" ? "not owned by you" : row.reason === "not_found" ? "already gone" : row.reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <footer className="modal-foot">
              <button
                type="button"
                className="button ghost"
                onClick={() => setDeleteModalOpen(false)}
                disabled={deleteBusy}
              >
                Cancel
              </button>
              <button
                type="button"
                className="button danger"
                onClick={performDelete}
                disabled={deleteBusy || selectedForDelete.size === 0}
              >
                {deleteBusy ? "Deleting…" : `Delete ${selectedForDelete.size}`}
              </button>
            </footer>
          </div>
        </div>
      )}

      <style jsx>{`
        /* ===== Design tokens ============================================ */
        :global(:root) {
          --bg: #e7edf2;
          --surface: #ffffff;
          --surface-muted: #f7fafc;
          --surface-tinted: #eef6f8;
          --border: #c5d3dc;
          --border-strong: #8ea8b6;
          --text: #10202b;
          --text-secondary: #395262;
          --text-muted: #6b8390;
          --brand: #13a395;
          --brand-strong: #08766d;
          --brand-soft: #e4f7f5;
          --accent: #38bdf8;
          --shadow-sm: 0 1px 2px rgba(7, 20, 29, 0.08), 0 1px 1px rgba(7, 20, 29, 0.06);
          --shadow-md: 0 18px 44px rgba(7, 20, 29, 0.12);
          --shadow-lg: 0 28px 70px rgba(7, 20, 29, 0.18);
          --radius-sm: 6px;
          --radius: 8px;
          --radius-lg: 10px;
        }
        :global(body) {
          margin: 0;
          background: var(--bg);
          color: var(--text);
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-feature-settings: "ss01", "cv11", "tnum";
          -webkit-font-smoothing: antialiased;
        }
        :global(*),
        :global(*::before),
        :global(*::after) {
          box-sizing: border-box;
        }
        .shell {
          max-width: 1480px;
          margin: 0 auto;
          padding: 24px 28px 54px;
          color: var(--text);
        }

        /* ===== Home platform intro ===================================== */
        .home-hero {
          display: grid;
          grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
          gap: 18px;
          align-items: stretch;
          margin-bottom: 18px;
        }
        .home-hero-copy,
        .workflow-card {
          border: 1px solid rgba(142, 168, 182, 0.78);
          border-radius: 12px;
          box-shadow: var(--shadow-md);
        }
        .home-hero-copy {
          position: relative;
          overflow: hidden;
          padding: 26px 28px;
          background:
            linear-gradient(135deg, rgba(7, 20, 29, 0.96), rgba(11, 31, 45, 0.93)),
            radial-gradient(circle at 92% 10%, rgba(19, 163, 149, 0.35), transparent 22rem);
          color: #e8f5f7;
        }
        .home-hero-copy::after {
          content: "";
          position: absolute;
          inset: 16px;
          pointer-events: none;
          border: 1px solid rgba(145, 208, 219, 0.14);
          border-radius: 10px;
        }
        .home-hero-copy .eyebrow {
          color: #72e1d6;
        }
        .home-hero-copy h2 {
          position: relative;
          z-index: 1;
          max-width: 880px;
          margin: 8px 0 0;
          color: #ffffff;
          font-size: clamp(30px, 4.6vw, 58px);
          font-weight: 900;
          letter-spacing: -0.035em;
          line-height: 0.98;
        }
        .home-hero-copy p {
          position: relative;
          z-index: 1;
          max-width: 820px;
          margin: 16px 0 0;
          color: #c7dce3;
          font-size: 15px;
          line-height: 1.62;
        }
        .hero-actions {
          position: relative;
          z-index: 1;
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 22px;
        }
        .hero-actions .button.secondary {
          border-color: rgba(200, 225, 232, 0.28);
          background: rgba(255, 255, 255, 0.07);
          color: #eaf8fb;
        }
        .hero-actions .button.secondary:hover {
          background: rgba(255, 255, 255, 0.12);
          border-color: rgba(114, 225, 214, 0.56);
        }
        .hero-capabilities {
          position: relative;
          z-index: 1;
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 20px;
        }
        .hero-capabilities span {
          display: inline-flex;
          gap: 7px;
          align-items: center;
          min-height: 30px;
          padding: 6px 9px;
          border: 1px solid rgba(145, 208, 219, 0.2);
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.06);
          color: #d8edf1;
          font-size: 12px;
          font-weight: 760;
        }
        .workflow-card {
          display: grid;
          align-content: start;
          gap: 14px;
          padding: 20px;
          background:
            linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,250,252,0.94)),
            #ffffff;
        }
        .workflow-card-head {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: baseline;
          padding-bottom: 12px;
          border-bottom: 1px solid var(--border);
        }
        .workflow-card-head span {
          color: var(--text-muted);
          font-size: 10.5px;
          font-weight: 900;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .workflow-card-head strong {
          color: var(--brand-strong);
          font-size: 12px;
          font-weight: 900;
          white-space: nowrap;
        }
        .workflow-steps {
          display: grid;
          gap: 12px;
          margin: 0;
          padding: 0;
          list-style: none;
        }
        .workflow-steps li {
          display: grid;
          grid-template-columns: 38px minmax(0, 1fr);
          gap: 12px;
          padding: 11px 0;
          border-bottom: 1px solid #e1eaf0;
        }
        .workflow-steps li:last-child {
          border-bottom: 0;
        }
        .workflow-steps li > span {
          display: grid;
          width: 34px;
          height: 34px;
          place-items: center;
          border: 1px solid rgba(19, 163, 149, 0.28);
          border-radius: 999px;
          background: var(--brand-soft);
          color: var(--brand-strong);
          font-size: 11px;
          font-weight: 900;
          font-variant-numeric: tabular-nums;
        }
        .workflow-steps strong {
          color: var(--text);
          font-size: 13px;
          font-weight: 900;
        }
        .workflow-steps p {
          margin: 3px 0 0;
          color: var(--text-secondary);
          font-size: 12.5px;
          line-height: 1.45;
        }

        /* ===== Top bar ================================================== */
        .topbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 18px;
          padding: 18px 0 14px;
          border-bottom: 1px solid var(--border);
          margin-bottom: 16px;
        }
        .brand-mark {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .brand-mark .glyph {
          width: 34px;
          height: 34px;
          border-radius: 8px;
          background: linear-gradient(135deg, var(--brand) 0%, #06b6d4 100%);
          display: grid;
          place-items: center;
          color: white;
          font-weight: 800;
          font-size: 14px;
          letter-spacing: -0.02em;
          box-shadow: var(--shadow-sm);
        }
        .brand-text { display: grid; gap: 2px; }
        .brand-text .eyebrow { color: var(--text-muted); font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 0; }
        .brand-text h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.01em; line-height: 1.1; margin: 0; color: var(--text); }
        .identity { display: flex; align-items: center; gap: 14px; font-size: 13px; }
        .identity .who { color: var(--text-secondary); font-weight: 500; }
        .identity .signout { color: var(--brand); font-weight: 600; text-decoration: none; padding: 6px 10px; border-radius: 6px; transition: background 120ms ease; }
        .identity .signout:hover { background: var(--brand-soft); }

        /* ===== Disclaimer ribbon ======================================== */
        .notice {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          padding: 9px 14px;
          margin-bottom: 18px;
          background: #fffbeb;
          border: 1px solid #fde68a;
          border-radius: 8px;
          font-size: 13px;
          color: #78350f;
        }
        .notice strong { color: #7c2d12; font-weight: 700; margin-right: 6px; }
        .notice button { background: #b45309; color: white; padding: 6px 12px; font-size: 12px; font-weight: 700; border-radius: 6px; }
        .notice button:hover { background: #92400e; }

        /* ===== Stat tiles =============================================== */
        .metrics {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 14px;
          margin-bottom: 18px;
        }
        .metrics > div {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 14px 16px;
          box-shadow: var(--shadow-sm);
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .metrics span { font-size: 24px; font-weight: 700; letter-spacing: -0.02em; line-height: 1; color: var(--text); }
        .metrics p { color: var(--text-muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 0; }

        /* ===== Layout shell ============================================= */
        .panel {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          box-shadow: var(--shadow-sm);
          padding: 20px;
        }
        .workspace { display: grid; grid-template-columns: minmax(340px, 420px) 1fr; gap: 16px; align-items: start; }
        .workspace-new, .workspace-runs { grid-template-columns: minmax(0, 1fr); }
        .workspace-new .input-panel { max-width: 760px; }
        .section-head { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 8px; }
        .section-head h2 { font-size: 16px; font-weight: 700; letter-spacing: -0.01em; color: var(--text); margin: 0; }
        .section-head .eyebrow { color: var(--text-muted); font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 2px; }
        .section-head-tight { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 10px; }
        .section-head-tight h3 { font-size: 14px; font-weight: 700; color: var(--text); margin: 0; }
        .section-sub { font-size: 12px; color: var(--text-muted); }

        /* ===== Inputs panel ============================================= */
        h3 { margin: 0; }
        .scenario-list, .run-list, .artifact-list { display: grid; gap: 8px; margin-top: 12px; }
        .scenario, .run-row {
          width: 100%;
          text-align: left;
          background: var(--surface-muted);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 11px 12px;
          cursor: pointer;
          color: var(--text);
          transition: border-color 120ms ease, background 120ms ease;
        }
        .scenario:hover, .run-row:hover { border-color: var(--brand); background: #ffffff; }
        .run-row.selected { border-color: var(--brand); background: var(--brand-soft); box-shadow: inset 3px 0 0 var(--brand); }
        .scenario strong, .run-row strong { display: block; font-size: 13.5px; font-weight: 700; color: var(--text); }
        .scenario span, .run-row span { display: block; margin-top: 3px; color: var(--text-secondary); font-size: 12.5px; }
        .scenario {
          display: grid;
          gap: 5px;
        }
        .scenario .scenario-kicker {
          margin: 0;
          color: var(--brand);
          font-size: 10.5px;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .scenario .scenario-desc {
          margin: 0;
          color: var(--text-secondary);
          line-height: 1.35;
        }
        .scenario .scenario-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 5px;
          margin-top: 4px;
        }
        .scenario .scenario-meta span {
          display: inline-flex;
          margin: 0;
          padding: 3px 7px;
          border: 1px solid var(--border);
          border-radius: 999px;
          background: #ffffff;
          color: var(--text-secondary);
          font-size: 10.5px;
          font-weight: 700;
          letter-spacing: 0.02em;
        }
        .form-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(120px, 130px); gap: 12px; margin-top: 14px; }
        .form-grid > label { min-width: 0; }
        .wide { grid-column: 1 / -1; }
        .member-budget-note {
          margin: -2px 0 0;
          padding: 9px 11px;
          border: 1px solid var(--border);
          border-radius: 6px;
          background: var(--surface-muted);
          color: var(--text-secondary);
          font-size: 12.5px;
          line-height: 1.45;
        }
        label { display: grid; gap: 6px; color: var(--text-secondary); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
        input, select { width: 100%; min-width: 0; border: 1px solid var(--border-strong); border-radius: 6px; padding: 9px 11px; background: var(--surface); color: var(--text); font: inherit; font-size: 13px; transition: border-color 120ms ease, box-shadow 120ms ease; }
        input:focus, select:focus { outline: none; border-color: var(--brand); box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12); }
        input[type="file"] { padding: 8px; }
        .file-picker { margin-top: 12px; }
        .file-name { color: var(--text-secondary); font-size: 12.5px; margin-top: 6px; }
        .checks { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 16px; margin: 14px 0 12px; align-items: start; }
        .checks label { display: grid; grid-template-columns: 18px minmax(0, 1fr); align-items: start; column-gap: 8px; font-weight: 500; text-transform: none; letter-spacing: 0; color: var(--text); font-size: 13px; line-height: 1.35; }
        .checks input[type="checkbox"] { width: 14px; min-width: 14px; height: 14px; margin: 2px 0 0; justify-self: center; padding: 0; }
        .forcing-preview {
          margin-top: 14px;
          padding: 14px;
          border: 1px solid var(--border);
          border-radius: 10px;
          background: #ffffff;
          box-shadow: var(--shadow-sm);
        }
        .preview-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
          margin-bottom: 10px;
        }
        .preview-head .eyebrow {
          margin: 0 0 2px;
          color: var(--text-muted);
          font-size: 10px;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .preview-head h3 {
          font-size: 14px;
          font-weight: 750;
          color: var(--text);
        }
        .preview-subtitle {
          margin: 4px 0 0;
          color: var(--text-secondary);
          font-size: 12px;
          line-height: 1.35;
        }
        .preview-stats {
          width: 100%;
          border-collapse: collapse;
          table-layout: fixed;
          margin-bottom: 10px;
          border: 1px solid var(--border);
          border-radius: 7px;
          background: var(--surface-muted);
          overflow: hidden;
        }
        .preview-stats th,
        .preview-stats td {
          padding: 7px 9px;
          border-bottom: 1px solid var(--border);
          vertical-align: top;
          font-size: 12px;
          line-height: 1.3;
        }
        .preview-stats tr:last-child th,
        .preview-stats tr:last-child td {
          border-bottom: 0;
        }
        .preview-stats th {
          width: 54%;
          text-align: left;
          color: var(--text-secondary);
          font-weight: 700;
          letter-spacing: 0.03em;
          text-transform: uppercase;
        }
        .preview-stats td {
          color: var(--text);
          font-variant-numeric: tabular-nums;
          overflow-wrap: anywhere;
        }
        .forcing-preview-chart {
          display: block;
          width: 100%;
          height: auto;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: #ffffff;
        }
        .sparkline { display: block; width: 100%; height: 78px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; }

        /* ===== Buttons ================================================== */
        button { border: 0; border-radius: 7px; padding: 9px 14px; font-weight: 600; cursor: pointer; font: inherit; font-size: 13px; transition: background 120ms ease, transform 60ms ease; }
        .button { display: inline-flex; align-items: center; justify-content: center; border: 0; border-radius: 7px; padding: 9px 14px; font-weight: 600; font-size: 13px; text-decoration: none; cursor: pointer; transition: background 120ms ease, transform 60ms ease; }
        .button.primary { background: var(--brand); color: white; }
        .button.primary:hover:not(:disabled) { background: var(--brand-strong); }
        .button.primary:active:not(:disabled) { transform: translateY(1px); }
        .button.ghost { background: var(--surface-muted); color: var(--text); border: 1px solid var(--border); text-decoration: none; }
        .button.ghost:hover:not(:disabled) { border-color: var(--brand); color: var(--brand); }
        .button.danger { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
        .button.danger:hover:not(:disabled) { background: #fee2e2; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .wide-button { width: 100%; padding: 11px; }

        /* ===== Validation messages ====================================== */
        .valid, .invalid, .message, .failure { display: grid; gap: 4px; padding: 10px 12px; border-radius: 7px; margin: 12px 0; font-size: 13px; line-height: 1.4; }
        .valid { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
        .invalid, .failure { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
        .message { background: #eff6ff; color: #1e3a8a; border: 1px solid #bfdbfe; }
        .screening-card { margin-top: 8px; padding: 10px; border-radius: 7px; background: rgba(255,255,255,0.7); border: 1px solid rgba(6,95,70,0.22); display: grid; gap: 3px; }

        /* ===== Run queue ================================================ */
        .queue-actions { display: flex; align-items: center; gap: 10px; }
        .select-all { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-secondary); text-transform: none; letter-spacing: 0; font-weight: 600; cursor: pointer; }
        .select-all input { margin: 0; }
        .run-list { list-style: none; padding: 0; }
        .run-row {
          display: grid;
          grid-template-columns: auto 1fr auto;
          align-items: stretch;
          gap: 0;
          padding: 0;
          background: var(--surface-muted);
          border: 1px solid var(--border);
          border-radius: 8px;
          overflow: hidden;
          transition: border-color 120ms ease, background 120ms ease, box-shadow 120ms ease;
        }
        .run-row:hover { border-color: var(--brand); background: var(--surface); }
        .run-row.selected { border-color: var(--brand); background: var(--brand-soft); box-shadow: inset 3px 0 0 var(--brand); }
        .run-row.queued-for-delete { border-color: #fca5a5; background: #fef2f2; box-shadow: inset 3px 0 0 #b91c1c; }
        .row-check {
          display: grid;
          place-items: center;
          padding: 0 10px;
          border-right: 1px solid var(--border);
          cursor: pointer;
          background: transparent;
          color: inherit;
          font: inherit;
          letter-spacing: 0;
          text-transform: none;
        }
        .row-check input { width: 16px; height: 16px; margin: 0; cursor: pointer; accent-color: var(--brand); }
        .row-check input:disabled { cursor: not-allowed; opacity: 0.4; }
        .run-row-body {
          display: grid;
          grid-template-columns: 1fr auto;
          align-items: center;
          gap: 10px;
          padding: 11px 12px;
          text-align: left;
          background: transparent;
          border: 0;
          color: var(--text);
          cursor: pointer;
          font: inherit;
          width: 100%;
        }
        .run-row-text strong { display: block; font-size: 13.5px; font-weight: 700; color: var(--text); }
        .run-row-text span { display: block; margin-top: 3px; color: var(--text-secondary); font-size: 12.5px; }
        .run-row-text .run-progress-label {
          color: #0f766e;
          font-weight: 650;
        }
        .run-open-link {
          display: inline-flex;
          align-items: center;
          padding: 0 12px;
          border-left: 1px solid var(--border);
          color: var(--brand-strong);
          font-size: 12px;
          font-weight: 800;
          text-decoration: none;
        }
        .run-open-link:hover {
          background: #ffffff;
          color: var(--brand);
        }
        .delete-toolbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          padding: 9px 12px;
          margin: 10px 0 8px;
          background: #fef2f2;
          border: 1px solid #fecaca;
          border-radius: 8px;
          color: #7f1d1d;
          font-size: 13px;
          font-weight: 500;
        }
        .delete-toolbar strong { font-weight: 700; }
        .delete-toolbar-actions { display: flex; gap: 8px; }
        .mini-progress, .progress-track { grid-column: 1 / -1; height: 5px; border-radius: 999px; background: var(--border); overflow: hidden; }
        .mini-progress span, .progress-track span { display: block; height: 100%; background: var(--brand); transition: width 250ms ease; }
        .status { display: inline-flex; align-items: center; justify-content: center; border-radius: 999px; padding: 3px 9px; font-size: 10.5px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
        .good { background: #d1fae5; color: #065f46; }
        .bad { background: #fee2e2; color: #991b1b; }
        .active { background: #dbeafe; color: #1e40af; }
        .waiting { background: #fef3c7; color: #92400e; }

        /* ===== Detail panel ============================================= */
        .detail-panel { margin-top: 16px; padding: 22px; }
        .detail-panel .section-head h2 { font-size: 18px; font-weight: 700; }
        .progress-card { border: 1px solid var(--border); background: var(--surface-muted); border-radius: var(--radius); padding: 14px 16px; margin-top: 14px; }
        .progress-header { display: flex; justify-content: space-between; align-items: center; }
        .progress-header strong { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; color: var(--text); }
        .progress-track { margin-top: 10px; height: 8px; }
        .progress-caption {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          margin-top: 10px;
          color: var(--text-secondary);
          font-size: 12.5px;
          line-height: 1.35;
        }
        .progress-caption strong {
          color: var(--text);
          font-weight: 700;
        }
        .runtime-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 8px;
          margin-top: 12px;
        }
        .runtime-grid div {
          padding: 10px 11px;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: #ffffff;
          min-width: 0;
        }
        .runtime-grid span,
        .runtime-grid small {
          display: block;
          color: var(--text-secondary);
          font-size: 11.5px;
          line-height: 1.3;
        }
        .runtime-grid strong {
          display: block;
          margin-top: 3px;
          color: var(--text);
          font-size: 15px;
          font-weight: 760;
          font-variant-numeric: tabular-nums;
        }
        .runtime-grid small { margin-top: 3px; }
        .detail-callout {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 18px;
          margin-top: 14px;
          padding: 16px 18px;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          background: #ffffff;
        }
        .detail-callout h3 {
          margin: 0;
          font-size: 16px;
          font-weight: 750;
          color: var(--text);
        }
        .detail-callout p:not(.eyebrow) {
          margin: 6px 0 0;
          max-width: 680px;
          color: var(--text-secondary);
          font-size: 13px;
          line-height: 1.45;
        }
        .run-spec-line {
          margin: 9px 0 0;
          color: var(--text-secondary);
          font-size: 12.5px;
          font-weight: 600;
        }
        .stage-track { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin: 12px 0 0; padding: 0; list-style: none; }
        .stage-track li {
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          border-radius: 6px;
          border: 1px solid var(--border);
          padding: 6px 8px;
          text-align: center;
          font-size: 10.5px;
          font-weight: 700;
          letter-spacing: 0.06em;
          color: var(--text-muted);
          background: var(--surface);
          text-transform: uppercase;
        }
        .stage-track li.done { color: #065f46; border-color: #6ee7b7; background: #ecfdf5; }
        .stage-track li.current { color: #1e40af; border-color: #93c5fd; background: #dbeafe; box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18); }
        .stage-track li.failed { color: #991b1b; border-color: #fca5a5; background: #fee2e2; }
        .stage-track li.canceled { color: #475569; border-color: #cbd5e1; background: #e2e8f0; }

        /* ===== Decision Summary ========================================= */
        .decision-summary { display: grid; grid-template-columns: 1.25fr 1fr 1fr 1fr; gap: 12px; margin-top: 18px; }
        .decision-card {
          border: 1px solid var(--border);
          background: var(--surface);
          border-radius: var(--radius);
          padding: 16px 18px;
          display: flex;
          flex-direction: column;
          gap: 6px;
          min-height: 138px;
          box-shadow: var(--shadow-sm);
        }
        .decision-card-headline { background: linear-gradient(160deg, var(--surface), var(--surface-tinted)); }
        .decision-eyebrow { color: var(--text-muted); font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        .decision-headline { font-size: 28px; font-weight: 800; color: var(--text); line-height: 1.05; letter-spacing: -0.02em; }
        .decision-value { font-size: 26px; font-weight: 700; color: var(--text); line-height: 1.05; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
        .decision-description { color: var(--text-secondary); font-size: 12.5px; line-height: 1.45; margin: 2px 0 0; }
        .decision-detail { color: var(--text-secondary); font-size: 12.5px; margin: 0; }
        .decision-detail strong { color: var(--text); font-weight: 700; }
        .decision-reasons { margin: 4px 0 0; padding: 0 0 0 14px; color: var(--text-secondary); font-size: 12px; line-height: 1.5; }
        .decision-reasons li { margin-top: 2px; }
        .decision-card.tone-good { border-color: #6ee7b7; background: linear-gradient(160deg, var(--surface), #ecfdf5); }
        .decision-card.tone-good .decision-headline, .decision-card.tone-good .decision-value { color: #047857; }
        .decision-card.tone-warning { border-color: #fcd34d; background: linear-gradient(160deg, var(--surface), #fffbeb); }
        .decision-card.tone-warning .decision-headline, .decision-card.tone-warning .decision-value { color: #b45309; }
        .decision-card.tone-bad { border-color: #fca5a5; background: linear-gradient(160deg, var(--surface), #fef2f2); }
        .decision-card.tone-bad .decision-headline, .decision-card.tone-bad .decision-value { color: #b91c1c; }
        .decision-card.tone-severe { border-color: #b91c1c; background: linear-gradient(160deg, #fee2e2, #fecaca); box-shadow: 0 0 0 1px rgba(185, 28, 28, 0.18), var(--shadow-md); }
        .decision-card.tone-severe .decision-headline, .decision-card.tone-severe .decision-value { color: #7f1d1d; }
        .decision-card.tone-waiting { border-color: var(--border); background: var(--surface-muted); }
        .decision-card.tone-waiting .decision-headline, .decision-card.tone-waiting .decision-value { color: var(--text-secondary); }

        /* ===== InfoTip =================================================== */
        .info-tip { position: relative; display: inline-flex; align-items: baseline; gap: 4px; cursor: help; }
        .info-tip:focus { outline: none; }
        .info-tip-icon { color: var(--text-muted); font-size: 11px; line-height: 1; transform: translateY(-1px); }
        .info-tip-bubble {
          position: absolute;
          top: 100%;
          left: 0;
          margin-top: 6px;
          background: #0f172a;
          color: #e2e8f0;
          font-size: 12px;
          line-height: 1.45;
          font-weight: 400;
          padding: 10px 12px;
          border-radius: 8px;
          box-shadow: 0 12px 28px rgba(15, 23, 42, 0.28);
          width: 270px;
          z-index: 30;
          opacity: 0;
          pointer-events: none;
          transform: translateY(-3px);
          transition: opacity 140ms ease, transform 140ms ease;
          text-transform: none;
          letter-spacing: 0;
        }
        .info-tip-label { font-size: inherit; font-weight: inherit; color: inherit; }
        .info-tip:hover .info-tip-bubble, .info-tip:focus .info-tip-bubble, .info-tip:focus-within .info-tip-bubble { opacity: 1; transform: translateY(0); pointer-events: auto; }

        /* ===== UQ figure products ======================================= */
        .figure-grid, .evolution-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; }
        .chart-card { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); padding: 14px 16px; box-shadow: var(--shadow-sm); }
        .figure-card { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); padding: 14px 16px; box-shadow: var(--shadow-sm); }
        .figure-card:first-child { grid-column: 1 / -1; }
        .figure-image { display: block; width: 100%; min-height: 220px; object-fit: contain; border: 1px solid var(--border); border-radius: 8px; background: #ffffff; }
        .chart-card-wide { grid-column: 1 / -1; }
        .chart-card-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }
        .chart-card-head h3 { font-size: 13px; color: var(--text); font-weight: 700; }
        .chart-pill { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; padding: 3px 9px; border-radius: 999px; background: var(--surface-muted); color: var(--text-secondary); border: 1px solid var(--border); }
        .chart-pill.calibrated { background: #ecfdf5; color: #047857; border-color: #6ee7b7; }
        .chart-pill.raw { background: #fffbeb; color: #b45309; border-color: #fde68a; }
        .chart { display: block; width: 100%; height: auto; }
        .chart-empty { color: var(--text-muted); font-style: italic; font-size: 12px; padding: 18px 4px; }

        /* ===== Time Player ============================================== */
        .player-section { margin-top: 18px; }
        .player {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 18px 20px 20px;
          box-shadow: var(--shadow-md);
        }
        .player-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 10px; }
        .player-product-hint { color: var(--text-secondary); font-size: 12.5px; line-height: 1.45; margin: 4px 0 0; max-width: 540px; }
        .player-products { display: flex; gap: 6px; padding: 4px; background: var(--surface-muted); border: 1px solid var(--border); border-radius: 10px; width: fit-content; margin-bottom: 12px; }
        .player-product {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          background: transparent;
          color: var(--text-secondary);
          border: 1px solid transparent;
          padding: 6px 12px;
          font-size: 12.5px;
          font-weight: 700;
          letter-spacing: 0.01em;
          border-radius: 7px;
          cursor: pointer;
          transition: background 120ms ease, color 120ms ease;
        }
        .player-product:hover:not(:disabled) { background: var(--surface); color: var(--text); }
        .player-product.active { background: var(--surface); color: var(--text); box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08); }
        .player-product.disabled, .player-product:disabled { opacity: 0.45; cursor: not-allowed; }
        .player-product-count {
          font-size: 10px;
          font-weight: 700;
          color: var(--text-muted);
          background: var(--surface);
          border: 1px solid var(--border);
          padding: 1px 6px;
          border-radius: 999px;
          font-variant-numeric: tabular-nums;
        }
        .player-product.active .player-product-count { background: var(--brand-soft); color: var(--brand); border-color: var(--brand-soft); }
        .player-eyebrow { color: var(--text-muted); font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 4px; }
        .player-title { font-size: 17px; font-weight: 700; color: var(--text); letter-spacing: -0.01em; }
        .player-sub { color: var(--text-muted); font-weight: 500; font-size: 14px; }
        .player-meta { display: flex; align-items: center; gap: 12px; }
        .player-time { font-size: 13px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
        .player-time strong { color: var(--text); font-weight: 700; }
        .player-stage {
          background: #0f172a;
          border-radius: var(--radius);
          overflow: hidden;
          display: grid;
          place-items: center;
          padding: 8px;
          min-height: 320px;
        }
        .player-map {
          display: block;
          max-width: 100%;
          max-height: 620px;
          width: auto;
          height: auto;
          border-radius: 6px;
          background: white;
        }
        .player-empty {
          color: #94a3b8;
          font-size: 13px;
          text-align: center;
          padding: 60px 20px;
        }
        .player-controls {
          display: flex;
          align-items: center;
          gap: 14px;
          margin-top: 14px;
          padding: 10px 14px;
          background: var(--surface-muted);
          border: 1px solid var(--border);
          border-radius: var(--radius);
        }
        .player-transport { display: flex; gap: 4px; }
        .player-transport button {
          width: 34px;
          height: 34px;
          padding: 0;
          background: var(--surface);
          color: var(--text-secondary);
          border: 1px solid var(--border);
          border-radius: 7px;
          font-size: 14px;
          display: grid;
          place-items: center;
          transition: all 120ms ease;
        }
        .player-transport button:hover:not(:disabled) { border-color: var(--brand); color: var(--brand); }
        .player-transport .player-play { background: var(--brand); color: white; border-color: var(--brand); width: 40px; height: 40px; font-size: 16px; }
        .player-transport .player-play:hover:not(:disabled) { background: var(--brand-strong); }
        .player-transport .player-play[aria-pressed="true"] { background: #1e40af; border-color: #1e40af; }
        .player-scrub {
          flex: 1;
          -webkit-appearance: none;
          appearance: none;
          height: 6px;
          background: linear-gradient(to right, var(--brand) 0%, var(--brand) var(--scrub-pct, 0%), var(--border) var(--scrub-pct, 0%), var(--border) 100%);
          border-radius: 999px;
          cursor: pointer;
          outline: none;
        }
        .player-scrub::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: var(--surface);
          border: 2px solid var(--brand);
          box-shadow: var(--shadow-sm);
          cursor: grab;
        }
        .player-scrub::-moz-range-thumb {
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: var(--surface);
          border: 2px solid var(--brand);
          cursor: grab;
        }
        .player-speed { display: flex; flex-direction: row; align-items: center; gap: 8px; font-size: 12px; color: var(--text-secondary); text-transform: none; letter-spacing: 0; font-weight: 600; }
        .player-speed select { padding: 5px 8px; font-size: 12px; }
        .player-subplots { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
        .player-chart { display: block; width: 100%; height: auto; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 6px; }
        .player-chart-empty { color: var(--text-muted); font-style: italic; font-size: 12px; padding: 30px 12px; text-align: center; }

        /* ===== Comparison gallery + downloads =========================== */
        .comparison-grid { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.6fr); gap: 16px; margin-top: 16px; }
        .comparison-maps, .artifact-aside { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; box-shadow: var(--shadow-sm); }
        .comparison-title { color: var(--text); font-size: 13px; font-weight: 700; margin: 6px 0; }
        .comparison-image { display: block; width: 100%; max-height: 560px; object-fit: contain; background: #0f172a; border-radius: 8px; padding: 8px; }
        .comparison-empty, .empty {
          color: var(--text-secondary);
          background: var(--surface-muted);
          border: 1px dashed var(--border-strong);
          border-radius: var(--radius);
          padding: 22px;
          font-size: 13px;
          text-align: center;
        }
        .guest-queue-empty {
          display: grid;
          gap: 8px;
          justify-items: center;
        }
        .guest-queue-empty strong { color: var(--text); }
        .comparison-tabs { display: flex; gap: 6px; overflow-x: auto; padding: 12px 0 4px; scrollbar-width: thin; }
        .comparison-tabs button {
          white-space: nowrap;
          background: var(--surface-muted);
          color: var(--text-secondary);
          font-size: 11.5px;
          font-weight: 600;
          padding: 6px 12px;
          border: 1px solid var(--border);
          border-radius: 999px;
        }
        .comparison-tabs button:hover { border-color: var(--brand); color: var(--brand); }
        .comparison-tabs button.active { background: var(--brand); color: white; border-color: var(--brand); }
        .artifact-list a {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          padding: 9px 0;
          border-bottom: 1px solid var(--border);
          font-size: 13px;
          font-weight: 600;
          color: var(--brand);
          text-decoration: none;
          font-variant-numeric: tabular-nums;
        }
        .artifact-list a:hover { color: var(--brand-strong); }
        .artifact-list small { color: var(--text-muted); white-space: nowrap; font-weight: 500; }

        /* ===== Delete confirmation modal ================================ */
        .modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(15, 23, 42, 0.55);
          backdrop-filter: blur(2px);
          display: grid;
          place-items: center;
          z-index: 50;
          padding: 24px;
        }
        .modal {
          background: var(--surface);
          border-radius: var(--radius-lg);
          box-shadow: 0 24px 60px rgba(15, 23, 42, 0.28);
          max-width: 520px;
          width: 100%;
          padding: 22px;
          max-height: 90vh;
          overflow-y: auto;
        }
        .modal-head h2 { font-size: 18px; font-weight: 700; margin: 0 0 6px; color: var(--text); letter-spacing: -0.01em; }
        .modal-head p { font-size: 13.5px; color: var(--text-secondary); line-height: 1.5; margin: 0; }
        .modal-list { margin: 14px 0 0; padding: 0; list-style: none; max-height: 240px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; background: var(--surface-muted); }
        .modal-list li {
          display: grid;
          grid-template-columns: 1fr auto auto;
          gap: 10px;
          align-items: center;
          padding: 9px 12px;
          border-bottom: 1px solid var(--border);
          font-size: 13px;
        }
        .modal-list li:last-child { border-bottom: 0; }
        .modal-list strong { font-weight: 700; color: var(--text); }
        .modal-list span { color: var(--text-secondary); font-size: 12px; }
        .modal-overflow {
          grid-template-columns: 1fr !important;
          font-style: italic;
          color: var(--text-muted);
          justify-content: center;
        }
        .modal-warning {
          margin-top: 12px;
          padding: 10px 12px;
          background: #fef3c7;
          border: 1px solid #fde68a;
          border-radius: 8px;
          color: #78350f;
          font-size: 12.5px;
          line-height: 1.5;
        }
        .modal-warning ul { margin: 6px 0 0; padding: 0 0 0 16px; }
        .modal-warning code { background: rgba(0,0,0,0.05); padding: 1px 5px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }
        .modal-foot { display: flex; justify-content: flex-end; gap: 10px; margin-top: 16px; }

        /* ===== Professional redesign pass =============================== */
        :global(:root) {
          --bg: #e9eef2;
          --surface: #ffffff;
          --surface-muted: #f6f8f9;
          --surface-tinted: #eef7f6;
          --border: #d7e2e6;
          --border-strong: #aebfc7;
          --text: #102027;
          --text-secondary: #465b63;
          --text-muted: #73848c;
          --brand: #0b766d;
          --brand-strong: #064f49;
          --brand-soft: #e8f6f4;
          --accent: #2f6db3;
          --accent-soft: #edf5ff;
          --warning: #c77700;
          --danger: #b42318;
          --ink: #0a1820;
          --shadow-sm: 0 1px 2px rgba(16, 32, 39, 0.08);
          --shadow-md: 0 10px 24px rgba(16, 32, 39, 0.10), 0 2px 6px rgba(16, 32, 39, 0.06);
          --shadow-lg: 0 22px 56px rgba(16, 32, 39, 0.16), 0 8px 18px rgba(16, 32, 39, 0.08);
          --radius-sm: 6px;
          --radius: 8px;
          --radius-lg: 8px;
        }
        :global(body) {
          background:
            linear-gradient(180deg, rgba(16, 32, 39, 0.035), rgba(16, 32, 39, 0) 280px),
            var(--bg);
          color: var(--text);
          font-family: Inter, Aptos, "Segoe UI", Arial, sans-serif;
          letter-spacing: 0;
        }
        :global(*) {
          letter-spacing: 0 !important;
        }
        .shell {
          max-width: 1720px;
          padding: 18px 32px 64px;
        }
        .topbar {
          position: relative;
          margin: 0 0 14px;
          padding: 24px 28px;
          border: 1px solid rgba(255, 255, 255, 0.10);
          border-radius: 8px;
          color: #eef7f6;
          background:
            linear-gradient(135deg, rgba(11, 118, 109, 0.86) 0%, rgba(13, 56, 62, 0.92) 48%, rgba(12, 31, 46, 0.98) 100%),
            #0c2730;
          box-shadow: var(--shadow-lg);
        }
        .brand-mark {
          gap: 16px;
        }
        .brand-mark .glyph {
          width: 44px;
          height: 44px;
          border-radius: 8px;
          background: #ffffff;
          color: var(--brand-strong);
          box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18);
          letter-spacing: 0;
        }
        .brand-text .eyebrow {
          color: rgba(238, 247, 246, 0.70);
          font-size: 11px;
          letter-spacing: 0.12em;
        }
        .brand-text h1 {
          color: #ffffff;
          font-size: 26px;
          font-weight: 750;
          line-height: 1.05;
          letter-spacing: 0;
        }
        .identity {
          color: rgba(238, 247, 246, 0.76);
          align-self: flex-start;
        }
        .identity .who {
          color: rgba(238, 247, 246, 0.76);
        }
        .identity .signout {
          color: #ffffff;
          background: rgba(255, 255, 255, 0.10);
          border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .identity .signout:hover {
          background: rgba(255, 255, 255, 0.18);
        }
        .notice {
          padding: 12px 16px;
          margin-bottom: 16px;
          border-color: #f0c36a;
          background: #fff7df;
          color: #6d4708;
          box-shadow: var(--shadow-sm);
        }
        .notice strong {
          color: #5f3500;
        }
        .metrics {
          grid-template-columns: repeat(4, minmax(180px, 1fr));
          gap: 12px;
          margin-bottom: 16px;
        }
        .metrics > div {
          position: relative;
          padding: 16px 18px;
          border-color: var(--border);
          border-radius: 8px;
          background: linear-gradient(180deg, #ffffff, #f8fbfb);
          box-shadow: var(--shadow-sm);
          overflow: hidden;
        }
        .metrics > div::before {
          content: "";
          position: absolute;
          inset: 0 auto 0 0;
          width: 4px;
          background: linear-gradient(180deg, var(--brand), var(--accent));
        }
        .metrics span {
          font-size: 28px;
          font-weight: 760;
          color: var(--ink);
          letter-spacing: 0;
        }
        .metrics p {
          color: var(--text-muted);
          letter-spacing: 0.09em;
        }
        .workspace {
          grid-template-columns: minmax(360px, 420px) minmax(0, 1fr);
          gap: 18px;
        }
        .workspace-new,
        .workspace-runs {
          grid-template-columns: minmax(0, 1fr);
        }
        .workspace-new .input-panel {
          max-width: 760px;
        }
        .panel {
          border-color: var(--border);
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.94);
          box-shadow: var(--shadow-md);
        }
        .input-panel {
          position: sticky;
          top: 16px;
        }
        .section-head {
          margin-bottom: 14px;
          padding-bottom: 10px;
          border-bottom: 1px solid var(--border);
        }
        .section-head h2,
        .section-head-tight h3,
        .chart-card-head h3 {
          color: var(--ink);
          letter-spacing: 0;
        }
        .section-head .eyebrow,
        .player-eyebrow,
        .decision-eyebrow {
          letter-spacing: 0.1em;
        }
        .scenario {
          border-color: #c8d9de;
          background: linear-gradient(180deg, #ffffff, #f5faf9);
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
        }
        .scenario:hover {
          transform: translateY(-1px);
          box-shadow: var(--shadow-sm);
        }
        .form-grid {
          grid-template-columns: minmax(0, 1fr) minmax(140px, 150px);
        }
        label {
          color: #334950;
          letter-spacing: 0.05em;
        }
        input,
        select {
          border-color: #b7c9d0;
          background: #ffffff;
        }
        .checks {
          padding: 8px 0 2px;
          border-top: 1px solid var(--border);
        }
        .button.primary {
          background: linear-gradient(180deg, #118980, var(--brand));
          box-shadow: 0 8px 16px rgba(11, 118, 109, 0.20);
        }
        .button.primary:hover:not(:disabled) {
          background: linear-gradient(180deg, var(--brand), var(--brand-strong));
        }
        .button.ghost {
          background: #ffffff;
        }
        .run-panel {
          min-height: 260px;
        }
        .run-list {
          gap: 10px;
        }
        .run-row {
          border-color: #cddce1;
          background: #ffffff;
          box-shadow: var(--shadow-sm);
        }
        .run-row:hover {
          transform: translateY(-1px);
          box-shadow: var(--shadow-md);
        }
        .run-row.selected {
          border-color: var(--brand);
          background: #f1fbfa;
          box-shadow: inset 4px 0 0 var(--brand), var(--shadow-sm);
        }
        .run-row-body {
          padding: 13px 14px;
        }
        .status {
          border: 1px solid transparent;
          letter-spacing: 0.06em;
        }
        .good {
          border-color: #9ad7bf;
        }
        .active {
          border-color: #9fc2ec;
        }
        .waiting {
          border-color: #eccb85;
        }
        .bad {
          border-color: #f4a6a0;
        }
        .mini-progress span,
        .progress-track span {
          background: linear-gradient(90deg, var(--brand), var(--accent));
        }
        .detail-panel {
          margin-top: 18px;
          padding: 24px;
          border-top: 4px solid var(--brand);
        }
        .progress-card {
          padding: 16px 18px;
          background: linear-gradient(180deg, #fbfdfd, #f3f8f8);
          border-color: var(--border);
        }
        .progress-header strong {
          font-size: 26px;
          color: var(--ink);
          letter-spacing: 0;
        }
        .progress-track {
          height: 10px;
          background: #dbe6ea;
        }
        .stage-track li {
          padding: 8px 10px;
          letter-spacing: 0.07em;
          background: #ffffff;
        }
        .decision-summary {
          grid-template-columns: minmax(280px, 1.15fr) repeat(3, minmax(210px, 1fr));
          gap: 14px;
        }
        .decision-card {
          min-height: 148px;
          padding: 18px;
          background: linear-gradient(180deg, #ffffff, #f8fbfb);
          border-color: var(--border);
          box-shadow: var(--shadow-sm);
        }
        .decision-card-headline {
          background:
            linear-gradient(135deg, rgba(11,118,109,0.12), rgba(47,109,179,0.08)),
            #ffffff;
        }
        .decision-headline,
        .decision-value {
          color: var(--ink);
          letter-spacing: 0;
        }
        .decision-headline {
          font-size: 30px;
        }
        .decision-value {
          font-size: 28px;
        }
        .decision-card.tone-bad {
          border-color: #e9a38e;
          background: linear-gradient(180deg, #fff8f3, #fff1e9);
        }
        .decision-card.tone-severe {
          border-color: #d98989;
          background: linear-gradient(180deg, #fff7f6, #fdeceb);
          box-shadow: var(--shadow-sm);
        }
        .figure-grid,
        .evolution-grid {
          gap: 16px;
          margin-top: 18px;
        }
        .figure-card {
          border-color: var(--border);
          background: #ffffff;
          box-shadow: var(--shadow-md);
        }
        .figure-image {
          background: #ffffff;
          border-color: var(--border);
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
        }
        .chart-card {
          border-color: var(--border);
          background: #ffffff;
          box-shadow: var(--shadow-sm);
        }
        .chart-card-head {
          padding-bottom: 8px;
          border-bottom: 1px solid var(--border);
        }
        .chart,
        .player-chart,
        .forcing-preview-chart {
          font-family: Inter, Aptos, "Segoe UI", Arial, sans-serif;
        }
        .chart text,
        .player-chart text,
        .forcing-preview-chart text,
        .sparkline text {
          font-family: Inter, Aptos, "Segoe UI", Arial, sans-serif;
          letter-spacing: 0;
        }
        .chart-pill {
          border-radius: 999px;
          letter-spacing: 0.08em;
        }
        .player-section {
          margin-top: 20px;
        }
        .player {
          padding: 0;
          overflow: hidden;
          border: 1px solid #cddce1;
          background: #ffffff;
          box-shadow: var(--shadow-lg);
        }
        .player-head {
          margin: 0;
          padding: 18px 20px;
          background: linear-gradient(180deg, #ffffff, #f4f8f9);
          border-bottom: 1px solid var(--border);
        }
        .player-title {
          color: var(--ink);
          font-size: 18px;
          letter-spacing: 0;
        }
        .player-sub {
          color: var(--text-secondary);
        }
        .player-stage {
          min-height: 460px;
          padding: 18px;
          background:
            linear-gradient(180deg, rgba(11, 24, 32, 0.96), rgba(16, 41, 54, 0.96)),
            #0a1820;
          border-radius: 0;
          border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .player-map {
          width: min(100%, 1120px);
          max-height: 70vh;
          border-radius: 8px;
          box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
        }
        .player-controls {
          margin: 0;
          padding: 14px 18px;
          border: 0;
          border-radius: 0;
          background: #f7fafb;
          border-bottom: 1px solid var(--border);
        }
        .player-transport button {
          border-color: #c8d9de;
          background: #ffffff;
        }
        .player-transport .player-play {
          background: var(--brand);
          border-color: var(--brand);
        }
        .player-scrub {
          background: linear-gradient(to right, var(--brand) 0%, var(--brand) var(--scrub-pct, 0%), #cfdbe0 var(--scrub-pct, 0%), #cfdbe0 100%);
        }
        .player-speed select {
          border-color: #c8d9de;
        }
        .player-subplots {
          padding: 16px 18px 18px;
          margin: 0;
          background: #ffffff;
        }
        .player-chart {
          border-color: var(--border);
          background: #fbfdfe;
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
        }
        .comparison-grid {
          grid-template-columns: minmax(0, 1.55fr) minmax(320px, 0.45fr);
          gap: 18px;
          margin-top: 18px;
        }
        .comparison-maps,
        .artifact-aside {
          border-color: var(--border);
          background: #ffffff;
          box-shadow: var(--shadow-md);
        }
        .comparison-image {
          max-height: 660px;
          background:
            linear-gradient(180deg, rgba(10, 24, 32, 0.98), rgba(17, 39, 51, 0.98)),
            #0a1820;
          padding: 14px;
          border-radius: 8px;
        }
        .comparison-tabs {
          gap: 8px;
          padding: 14px 0 4px;
        }
        .comparison-tabs button {
          border-radius: 999px;
          background: #f3f7f8;
          border-color: #cfdde1;
        }
        .comparison-tabs button.active {
          background: var(--ink);
          border-color: var(--ink);
        }
        .artifact-list {
          max-height: 680px;
          overflow: auto;
          padding-right: 4px;
        }
        .artifact-list a {
          color: var(--brand-strong);
          border-bottom-color: var(--border);
        }
        .artifact-list a:hover {
          color: var(--accent);
        }
        .modal {
          border-radius: 8px;
        }

        /* ===== Responsive =============================================== */
        @media (max-width: 1100px) {
          .decision-summary { grid-template-columns: 1fr 1fr; }
          .figure-grid { grid-template-columns: 1fr; }
          .player-subplots { grid-template-columns: 1fr; }
        }
        @media (max-width: 980px) {
          .shell { padding: 0 16px 36px; }
          .home-hero, .workspace, .comparison-grid, .metrics, .decision-summary, .figure-grid, .evolution-grid, .chart-grid { grid-template-columns: 1fr; }
          .home-hero-copy { padding: 22px; }
          .home-hero-copy h2 { font-size: clamp(28px, 9vw, 44px); }
          .workflow-card { padding: 16px; }
          .preview-stats { grid-template-columns: 1fr 1fr; }
          .topbar { flex-direction: column; align-items: flex-start; }
          .stage-track { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .runtime-grid { grid-template-columns: 1fr; }
          .detail-callout { flex-direction: column; align-items: flex-start; }
          .player-controls { flex-wrap: wrap; }
          .player-scrub { order: 3; flex: 1 0 100%; }
        }
      `}</style>
      </div>
    </AppShell>
  );
}
