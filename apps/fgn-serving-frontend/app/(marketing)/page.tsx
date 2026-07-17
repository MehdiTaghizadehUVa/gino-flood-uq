import Image from "next/image";
import caseStudyManifest from "../../public/marketing/portsmouth/manifest.json";
import { caseStudyAsset } from "./caseStudyAsset";
import { ArrowDown, ArrowRight, ArrowUpRight, Check, Play, ShieldCheck } from "lucide-react";
import { CalibrationStory } from "./components/CalibrationStory";
import { ComparisonMatrix } from "./components/ComparisonMatrix";
import { DecisionRiskStory } from "./components/DecisionRiskStory";
import { HeroFloodVideo } from "./components/HeroFloodVideo";
import { MarketingFooter } from "./components/MarketingFooter";
import { MarketingNav } from "./components/MarketingNav";
import { MonitoringStory } from "./components/MonitoringStory";
import { NumberedGrid } from "./components/NumberedGrid";
import { PortsmouthCaseStudy } from "./components/PortsmouthCaseStudy";
import { MicroLabel, Section } from "./components/Primitives";
import { ScrollChapterProgress } from "./components/ScrollChapterProgress";
import { ScrollReveal } from "./components/ScrollReveal";
import { ServiceLifecycle } from "./components/ServiceLifecycle";
import { UncertaintyStory } from "./components/UncertaintyStory";
import {
  calibrationNotes,
  comparisonMethods,
  comparisonRows,
  decisionQuestions,
  deploymentSteps,
  evidenceRegistry,
  monitoringLoop,
  PILOT_MAILTO,
  portsmouthCaseStudy,
  serviceOutcomes,
  servicePillars,
  speedPrinciples,
  uncertaintySources,
  useCases
} from "./content";

const structuredData = {
  "@context": "https://schema.org",
  "@type": "Service",
  name: "FloodUQ managed probabilistic coastal flood modeling",
  serviceType: "Domain-specific coastal flood uncertainty modeling",
  description:
    "A managed coastal flood intelligence service for rapid scenario evaluation with calibrated probabilities, explainable uncertainty, and monitored model behavior.",
  url: "https://flooduq.app",
  provider: {
    "@type": "Organization",
    name: "University of Virginia"
  }
};

const varianceDecompositionMath = String.raw`
  <math display="block" aria-label="Total variance equals epistemic uncertainty plus aleatoric uncertainty">
    <mrow>
      <mi mathvariant="normal">Var</mi><mo stretchy="false">(</mo><mi>H</mi><mo stretchy="false">)</mo><mo>=</mo>
      <munder class="variance-term epistemic-term">
        <munder accentunder="true">
          <mrow>
            <msub><mi mathvariant="normal">Var</mi><mi>Θ</mi></msub>
            <mo>[</mo>
            <msub><mi mathvariant="normal">E</mi><mi>Z</mi></msub>
            <mo>(</mo><mi>H</mi><mo>|</mo><mi>Θ</mi><mo>)</mo>
            <mo>]</mo>
          </mrow>
          <mo stretchy="true">⏟</mo>
        </munder>
        <mtext>epistemic uncertainty</mtext>
      </munder>
      <mo>+</mo>
      <munder class="variance-term aleatoric-term">
        <munder accentunder="true">
          <mrow>
            <msub><mi mathvariant="normal">E</mi><mi>Θ</mi></msub>
            <mo>[</mo>
            <msub><mi mathvariant="normal">Var</mi><mi>Z</mi></msub>
            <mo>(</mo><mi>H</mi><mo>|</mo><mi>Θ</mi><mo>)</mo>
            <mo>]</mo>
          </mrow>
          <mo stretchy="true">⏟</mo>
        </munder>
        <mtext>aleatoric uncertainty</mtext>
      </munder>
    </mrow>
  </math>
`;

export default function MarketingPage() {
  return (
    <>
      <a className="skip-link" href="#main-content">Skip to service overview</a>
      <MarketingNav />
      <main id="main-content">
        <section id="top" className="marketing-hero" aria-labelledby="hero-title">
          <HeroFloodVideo
            posterSrc={caseStudyAsset(caseStudyManifest.flagship.hero.posterSrc)}
            mp4Src={caseStudyAsset(caseStudyManifest.flagship.hero.mp4Src)}
            webmSrc={caseStudyAsset(caseStudyManifest.flagship.hero.webmSrc)}
          />
          <div className="hero-scrim" aria-hidden="true" />
          <div className="hero-content">
            <p className="hero-kicker">Managed probabilistic coastal flood modeling</p>
            <h1 id="hero-title" className="hero-logo-heading">
              <span className="visually-hidden">FloodUQ</span>
              <Image
                className="hero-brand-lockup"
                src="/marketing/brand/flooduq-logo-lockup.png"
                alt=""
                width={1453}
                height={327}
                sizes="(max-width: 760px) calc(100vw - 32px), 1100px"
                priority
              />
            </h1>
            <p className="hero-statement">
              Know where flooding may go.<br />Know how certain the model is.
            </p>
            <p className="hero-copy">
              Compare coastal flood scenarios quickly, evaluate calibrated exceedance probabilities, understand why
              forecasts diverge, and identify where new high-fidelity evidence will add the most value.
            </p>
            <div className="hero-actions">
              <a className="primary-cta" href={PILOT_MAILTO}>
                Request a pilot <ArrowUpRight size={17} aria-hidden="true" />
              </a>
              <a className="secondary-cta" href="/demo">
                <Play size={16} fill="currentColor" aria-hidden="true" /> Explore the Portsmouth demo
              </a>
            </div>
            <a className="scroll-cue" href="#capabilities">
              Explore the service <ArrowDown size={15} aria-hidden="true" />
            </a>
          </div>
          <div className="hero-proof" aria-label="FloodUQ service capabilities">
            {servicePillars.map((item) => (
              <div key={item.title}>
                <strong>{item.title}</strong>
                <span>{item.body}</span>
              </div>
            ))}
          </div>
        </section>

        <Section
          id="capabilities"
          eyebrow="The decision problem"
          title={<>A single flood map hides<br /><span>the decision risk.</span></>}
          intro={(
            <p>
              A central estimate shows one expected outcome. Better scenario decisions also require probability,
              timing, uncertainty, and a clear signal when the model is outside familiar evidence.
            </p>
          )}
          className="problem-section"
        >
          <DecisionRiskStory questions={decisionQuestions} />
        </Section>

        <Section
          eyebrow="Service outcomes"
          title={<>See the probability, timing, and uncertainty<br /><span>behind every scenario.</span></>}
          intro={(
            <p>
              Compare forcing scenarios, inspect individual locations, and trace every conclusion back to the evidence
              that produced it.
            </p>
          )}
          className="products-section"
        >
          <div className="product-gallery">
            {serviceOutcomes.map((outcome) => (
              <figure key={outcome.label}>
                <Image
                  src={caseStudyAsset(outcome.src)}
                  alt={outcome.alt}
                  width={outcome.width}
                  height={outcome.height}
                  sizes="(max-width: 760px) 100vw, 33vw"
                />
                <figcaption>
                  <strong>{outcome.label}</strong>
                  <p>{outcome.body}</p>
                </figcaption>
              </figure>
            ))}
          </div>
          <div className="service-audit-band">
            <MicroLabel>Evidence that stays connected</MicroLabel>
            <p>
              Review local forecast traces, compare raw and calibrated probabilities, and trace each result to its
              forcing, model version, calibration, and monitoring record.
            </p>
          </div>
        </Section>

        <Section
          eyebrow="Scenario capacity"
          title={<>Explore more alternatives<br /><span>within the same study cycle.</span></>}
          intro={(
            <p>
              Direct neural-operator inference reduces the cost of generating probabilistic scenarios, making it
              practical to test changing assumptions instead of stopping at one deterministic map.
            </p>
          )}
          className="speed-section"
        >
          <NumberedGrid items={speedPrinciples} columns={3} />
          <a className="evidence-link" href="#evidence">
            Review the measured Portsmouth comparison <ArrowDown size={16} aria-hidden="true" />
          </a>
        </Section>

        <Section
          id="calibration"
          eyebrow="Calibrated confidence"
          title={<>Turn ensemble counts into<br /><span>probabilities you can evaluate.</span></>}
          intro={(
            <p>
              Threshold-specific calibration aligns raw exceedance frequencies with held-out reference simulations,
              while preserving the original ensemble signal for comparison.
            </p>
          )}
          className="calibration-section"
        >
          <CalibrationStory notes={calibrationNotes} />
        </Section>

        <Section
          id="uncertainty"
          eyebrow="Explainable uncertainty"
          title={<>See the spread.<br /><span>Know what is driving it.</span></>}
          intro={(
            <p>
              FloodUQ preserves its nested ensemble structure so epistemic uncertainty is not confused with
              aleatoric uncertainty retained among plausible members from the same model.
            </p>
          )}
          className="method-section"
        >
          <UncertaintyStory sources={uncertaintySources} equationHtml={varianceDecompositionMath} />
        </Section>

        <Section
          id="comparison"
          eyebrow="Why FloodUQ"
          title={<>Fast forecasts are more useful<br /><span>when uncertainty stays visible.</span></>}
          intro={(
            <p>
              FloodUQ complements high-fidelity physics with rapid calibrated ensembles, interpretable uncertainty
              sources, and monitoring that identifies when a scenario needs stronger evidence.
            </p>
          )}
          className="comparison-section"
        >
          <ComparisonMatrix methods={comparisonMethods} rows={comparisonRows} />
        </Section>

        <Section
          id="monitoring"
          eyebrow="Model governance"
          title={<>Know when a scenario<br /><span>needs more evidence.</span></>}
          intro={(
            <p>
              FloodUQ flags unfamiliar or high-disagreement scenarios for expert review, helping teams direct new
              high-fidelity simulations toward the largest evidence gaps.
            </p>
          )}
          className="monitoring-section"
        >
          <MonitoringStory items={monitoringLoop} />
        </Section>

        <Section
          id="evidence"
          eyebrow="Portsmouth, Virginia / Deployment proof"
          title={<>From historical forcing to<br /><span>calibrated flood probabilities.</span></>}
          intro={<p>{portsmouthCaseStudy.intro}</p>}
          className="case-study-section"
        >
          <PortsmouthCaseStudy evidence={evidenceRegistry} />
        </Section>

        <Section
          id="deployment"
          eyebrow="Managed deployment"
          title={<>A proven path from your coastline<br /><span>to a working FloodUQ service.</span></>}
          intro={(
            <p>
              Each deployment is built from the coastline's own terrain, forcing scenarios, reference simulations,
              validation criteria, and decision needs.
            </p>
          )}
          className="deployment-section"
        >
          <ServiceLifecycle items={deploymentSteps} ariaLabel="Managed FloodUQ deployment lifecycle" />
        </Section>

        <Section
          id="solutions"
          eyebrow="Who it serves"
          title={<>Built for teams comparing<br /><span>coastal flood scenarios.</span></>}
          intro={(
            <p>
              FloodUQ helps resilience agencies, infrastructure operators, engineering partners, and risk teams move
              from isolated model runs to comparable probabilistic evidence.
            </p>
          )}
          className="solutions-section"
        >
          <NumberedGrid items={useCases} columns={3} />
        </Section>

        <section className="demo-section" data-reveal>
          <div className="demo-section-inner">
            <div className="demo-copy">
              <p className="marketing-eyebrow">Portsmouth product demo</p>
              <h2>See calibrated flood uncertainty<br /><span>in action.</span></h2>
              <p>
                Use the Portsmouth deployment to submit a forcing scenario, follow the forecast, inspect calibrated
                maps and local uncertainty, and see how unfamiliar scenarios are identified for review.
              </p>
              <div className="hero-actions">
                <a className="primary-cta" href={PILOT_MAILTO}>
                  Request a pilot <ArrowUpRight size={17} aria-hidden="true" />
                </a>
                <a className="secondary-cta" href="/demo">
                  Explore the Portsmouth demo <ArrowRight size={17} aria-hidden="true" />
                </a>
              </div>
              <div className="demo-safety">
                <ShieldCheck size={18} aria-hidden="true" />
                <span>Google sign-in and an approved user account are required to submit GPU work.</span>
              </div>
            </div>
            <div className="terminal" aria-label="FloodUQ managed service workflow">
              <div className="terminal-bar"><span /><span /><span /><strong>flooduq | managed coastal intelligence</strong></div>
              <div className="terminal-body">
                <p><span className="terminal-prompt">01</span> submit a forcing scenario</p>
                <p className="terminal-output"><Check size={14} /> input quality and model familiarity assessed</p>
                <p><span className="terminal-prompt">02</span> generate calibrated flood evidence</p>
                <p className="terminal-output"><Check size={14} /> depth, probability, timing, extent, and uncertainty ready</p>
                <p><span className="terminal-prompt">03</span> review confidence and evidence gaps</p>
                <p className="terminal-output"><Check size={14} /> priority scenarios preserved for expert review</p>
                <p className="terminal-cursor">ready to compare <span>|</span></p>
              </div>
            </div>
          </div>
        </section>

        <section className="final-cta" data-reveal aria-labelledby="final-cta-title">
          <div>
            <p className="marketing-eyebrow">Start with a domain-specific pilot</p>
            <h2 id="final-cta-title">Bring your coastline, reference evidence, and study questions.</h2>
            <p>Together, we will define the domain, reference simulations, decision thresholds, validation plan, and service configuration.</p>
          </div>
          <a className="primary-cta" href={PILOT_MAILTO}>
            Request a pilot <ArrowUpRight size={17} aria-hidden="true" />
          </a>
        </section>

      </main>
      <ScrollChapterProgress />
      <MarketingFooter />
      <ScrollReveal />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }} />
    </>
  );
}
