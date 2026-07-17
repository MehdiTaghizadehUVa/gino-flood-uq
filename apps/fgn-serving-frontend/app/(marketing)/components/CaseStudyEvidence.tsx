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
            alt="Skill-cost comparison of FGNO, diffusion, and MC-dropout ensembles in the Portsmouth held-out benchmark"
            width={4479}
            height={3339}
            sizes="(max-width: 900px) 100vw, 68vw"
          />
          <figcaption>
            The measured 20-member point supports the headline comparison. Larger ensemble timings shown in the source
            analysis are not used for marketing claims when they are extrapolated.
          </figcaption>
        </figure>
        <aside className="benchmark-methodology" aria-label="Benchmark scope">
          <MicroLabel>Benchmark scope</MicroLabel>
          <dl>
            <div><dt>Events</dt><dd>{evidence.benchmark.sample}</dd></div>
            <div><dt>Budget</dt><dd>{evidence.benchmark.ensembleBudget}</dd></div>
            <div><dt>Hardware</dt><dd>{evidence.benchmark.hardware}</dd></div>
            <div><dt>Timing</dt><dd>{evidence.benchmark.timingScope}</dd></div>
            <div><dt>Skill</dt><dd>{evidence.benchmark.skillMetrics}</dd></div>
          </dl>
          <p>Results are specific to this benchmark and do not guarantee the same ratios for another domain or hardware stack.</p>
        </aside>
      </div>

      <div className="case-study-runtime">
        <div>
          <MicroLabel>Full service workflow</MicroLabel>
          <h3>Inference is only one part of a delivered result.</h3>
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
