import Image from "next/image";
import type { EvidenceRegistry } from "../content";
import { BenchmarkBars } from "./BenchmarkBars";
import { MicroLabel } from "./Primitives";

type CaseStudy = {
  label: string;
  title: string;
  intro: string;
  deploymentStats: readonly { value: string; label: string }[];
  runtimeBars: readonly { label: string; value: number; display: string; tone: string }[];
  fullWorkflowNote: string;
  historicalNote: string;
};

export function CaseStudyEvidence({
  caseStudy,
  evidence
}: {
  caseStudy: CaseStudy;
  evidence: EvidenceRegistry;
}) {
  return (
    <div className="case-study-evidence">
      <div className="case-study-stats" aria-label="Portsmouth deployment configuration">
        {caseStudy.deploymentStats.map((stat) => (
          <div key={stat.label}>
            <strong>{stat.value}</strong>
            <span>{stat.label}</span>
          </div>
        ))}
      </div>

      <div className="evidence-claim-grid" aria-label="Measured Portsmouth comparison results">
        {evidence.claims.map((claim) => (
          <article key={claim.id}>
            <strong>{claim.value}</strong>
            <h3>{claim.label}</h3>
            <p>{claim.scope}</p>
          </article>
        ))}
      </div>

      <div className="skill-cost-layout">
        <figure className="skill-cost-figure">
          <Image
            src={evidence.benchmark.figure}
            alt="Forecast speed and quality comparison for FloodUQ, diffusion AI, and dropout-based forecast groups on Portsmouth events not used for training"
            width={4479}
            height={3339}
            sizes="(max-width: 900px) 100vw, 68vw"
          />
          <figcaption>
            The headline comparison uses 20 plausible forecasts from each method. Larger forecast-group timings in the
            source analysis are excluded when they were estimated rather than measured.
          </figcaption>
        </figure>
        <aside className="benchmark-methodology" aria-label="Benchmark scope">
          <MicroLabel>What the comparison covers</MicroLabel>
          <dl>
            <div><dt>Events</dt><dd>{evidence.benchmark.sample}</dd></div>
            <div><dt>Forecasts</dt><dd>{evidence.benchmark.ensembleBudget}</dd></div>
            <div><dt>Hardware</dt><dd>{evidence.benchmark.hardware}</dd></div>
            <div><dt>Speed measurement</dt><dd>{evidence.benchmark.timingScope}</dd></div>
            <div><dt>Forecast-quality measures</dt><dd>{evidence.benchmark.skillMetrics}</dd></div>
          </dl>
          <p>Results are specific to this benchmark and do not guarantee the same ratios for another domain or hardware stack.</p>
        </aside>
      </div>

      <div className="case-study-runtime">
        <div>
          <MicroLabel>Full service workflow</MicroLabel>
          <h3>Model forecast generation is only one part of a delivered result.</h3>
          <p>{caseStudy.fullWorkflowNote}</p>
        </div>
        <BenchmarkBars
          bars={caseStudy.runtimeBars}
          max={8}
          ariaLabel="Measured Portsmouth full workflow phase timings"
        />
      </div>

    </div>
  );
}
