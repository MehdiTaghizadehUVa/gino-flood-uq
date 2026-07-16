import Image from "next/image";
import caseStudyManifest from "../../public/marketing/portsmouth/manifest.json";
import { ArrowDown, ArrowRight, ArrowUpRight, Check, Play, ShieldCheck } from "lucide-react";
import { CalibrationGraphic } from "./components/CalibrationGraphic";
import { ComparisonMatrix } from "./components/ComparisonMatrix";
import { MarketingFooter } from "./components/MarketingFooter";
import { MarketingNav } from "./components/MarketingNav";
import { NumberedGrid } from "./components/NumberedGrid";
import { PortsmouthCaseStudy } from "./components/PortsmouthCaseStudy";
import { MicroLabel, Section } from "./components/Primitives";
import { ScrollReveal } from "./components/ScrollReveal";
import { ServiceLifecycle } from "./components/ServiceLifecycle";
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
  RESEARCH_DISCLAIMER,
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
    "A managed research service for rapid coastal scenario studies with calibrated probabilities, explainable uncertainty, and monitored model behavior.",
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
      <munder>
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
      <munder>
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
          <Image
            className="hero-media"
            src={caseStudyManifest.flagship.hero.src}
            alt=""
            fill
            priority
            sizes="100vw"
            aria-hidden="true"
          />
          <div className="hero-scrim" aria-hidden="true" />
          <div className="hero-content">
            <p className="hero-kicker">Managed probabilistic coastal flood modeling</p>
            <h1 id="hero-title" className="hero-wordmark" aria-label="FloodUQ">
              {"FLOODUQ".split("").map((letter, index) => <span key={`${letter}-${index}`}>{letter}</span>)}
            </h1>
            <p className="hero-statement">
              Know where flooding may go.<br />Know how certain the model is.
            </p>
            <p className="hero-copy">
              FloodUQ turns a domain-specific coastal model into rapid probability, timing, extent, and uncertainty
              products, delivered through a monitored research service.
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
              A central estimate can show one plausible outcome. Study teams also need threshold probability, timing,
              spread, and a clear signal when the model is operating outside familiar evidence.
            </p>
          )}
          className="problem-section"
        >
          <div className="comparison-grid">
            <figure className="comparison-figure">
              <Image
                src="/marketing/mean-depth.webp"
                alt="Ensemble-mean coastal water-depth map from the Portsmouth deployment"
                width={1400}
                height={1180}
                sizes="(max-width: 900px) 100vw, 50vw"
              />
              <figcaption>
                <MicroLabel>Central estimate</MicroLabel>
                <strong>Where the ensemble expects water</strong>
                <p>Useful for orientation, but unable to show how often a threshold is exceeded or where members disagree.</p>
              </figcaption>
            </figure>
            <figure className="comparison-figure featured">
              <Image
                src="/marketing/exceedance-probability.webp"
                alt="Calibrated coastal water-depth exceedance probability map from the Portsmouth deployment"
                width={1400}
                height={1180}
                sizes="(max-width: 900px) 100vw, 50vw"
              />
              <figcaption>
                <MicroLabel>Probability view</MicroLabel>
                <strong>Where a study threshold may be exceeded</strong>
                <p>Calibrated probabilities add frequency and confidence context while preserving the raw ensemble evidence.</p>
              </figcaption>
            </figure>
          </div>
          <div className="decision-questions">
            <NumberedGrid items={decisionQuestions} columns={3} />
          </div>
        </Section>

        <Section
          eyebrow="Service outcomes"
          title={<>Answers organized around<br /><span>the questions you ask.</span></>}
          intro={(
            <p>
              Compare forcing scenarios, inspect individual locations, and retain the scientific evidence behind every
              map and summary.
            </p>
          )}
          className="products-section"
        >
          <div className="product-gallery">
            {serviceOutcomes.map((outcome) => (
              <figure key={outcome.label}>
                <Image
                  src={outcome.src}
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
            <MicroLabel>Connected evidence</MicroLabel>
            <p>
              Cell-level traces, raw and calibrated products, input fingerprints, model versions, monitoring reports,
              and downloadable artifacts remain attached to the same run.
            </p>
          </div>
        </Section>

        <Section
          eyebrow="Scenario capacity"
          title={<>Explore alternatives without<br /><span>diffusion-scale sampling cost.</span></>}
          intro={(
            <p>
              FloodUQ is designed for rapid ensemble inference so a study can examine changing assumptions rather than
              stopping at one deterministic scenario.
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
          eyebrow="Calibrated confidence"
          title={<>Probability is corrected<br /><span>against held-out evidence.</span></>}
          intro={(
            <p>
              Raw ensemble exceedance is mapped through threshold-specific calibration curves fitted to held-out
              reference simulations. The correction and the original signal remain visible together.
            </p>
          )}
          className="calibration-section"
        >
          <div className="calibration-layout">
            <CalibrationGraphic />
            <div className="calibration-notes">
              {calibrationNotes.map((note) => (
                <article key={note.number}>
                  <span>{note.number}</span>
                  <h3>{note.title}</h3>
                  <p>{note.body}</p>
                </article>
              ))}
            </div>
          </div>
        </Section>

        <Section
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
          <NumberedGrid items={uncertaintySources} columns={2} />
          <div className="math-band">
            <div>
              <MicroLabel>Variance decomposition</MicroLabel>
              <p>
                High epistemic uncertainty can motivate additional model review or reference evidence. High
                aleatoric uncertainty indicates a broad retained range of plausible forecast members.
              </p>
            </div>
            <div
              className="math-expression"
              dangerouslySetInnerHTML={{ __html: varianceDecompositionMath }}
            />
          </div>
        </Section>

        <Section
          id="comparison"
          eyebrow="Why FloodUQ"
          title={<>A managed uncertainty system,<br /><span>not just another surrogate.</span></>}
          intro={(
            <p>
              Physics models remain the reference authority. FloodUQ complements them with rapid, calibrated ensemble
              products, explicit disagreement structure, and monitoring around every submitted scenario.
            </p>
          )}
          className="comparison-section"
        >
          <ComparisonMatrix methods={comparisonMethods} rows={comparisonRows} />
        </Section>

        <Section
          eyebrow="Model governance"
          title={<>The service knows when a scenario<br /><span>deserves more evidence.</span></>}
          intro={(
            <p>
              Monitoring does not prove model error and it does not retrain automatically. It creates a disciplined,
              expert-reviewed path from unfamiliar behavior to future high-fidelity evidence and versioned updates.
            </p>
          )}
          className="monitoring-section"
        >
          <ServiceLifecycle items={monitoringLoop} ariaLabel="FloodUQ monitoring and evidence-review lifecycle" />
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
          title={<>A repeatable path from your domain<br /><span>to a monitored research service.</span></>}
          intro={(
            <p>
              FloodUQ is not a universal pretrained map. Each coastline is onboarded through a defined data, training,
              calibration, validation, and governance workflow.
            </p>
          )}
          className="deployment-section"
        >
          <ServiceLifecycle items={deploymentSteps} ariaLabel="Managed FloodUQ deployment lifecycle" />
        </Section>

        <Section
          id="solutions"
          eyebrow="Who it serves"
          title={<>Built for coastal studies<br /><span>where uncertainty matters.</span></>}
          intro={(
            <p>
              The primary service supports resilience and infrastructure studies, with technical partnerships for
              engineering modelers and risk teams.
            </p>
          )}
          className="solutions-section"
        >
          <NumberedGrid items={useCases} columns={3} />
        </Section>

        <section className="demo-section" data-reveal>
          <div className="demo-section-inner">
            <div className="demo-copy">
              <p className="marketing-eyebrow">Portsmouth research demo</p>
              <h2>Inspect the workflow.<br /><span>Then scope your domain.</span></h2>
              <p>
                The gated demo presents the Portsmouth deployment: submit coastal stage and precipitation forcing,
                follow the GPU run, and inspect calibrated maps, disagreement diagnostics, monitoring, and artifacts.
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
                <span>Google sign-in and an approved research account are required to submit GPU work.</span>
              </div>
            </div>
            <div className="terminal" aria-label="FloodUQ managed service workflow">
              <div className="terminal-bar"><span /><span /><span /><strong>flooduq | managed research service</strong></div>
              <div className="terminal-body">
                <p><span className="terminal-prompt">01</span> validate domain forcing</p>
                <p className="terminal-output"><Check size={14} /> contract and reference screening complete</p>
                <p><span className="terminal-prompt">02</span> generate calibrated ensemble</p>
                <p className="terminal-output"><Check size={14} /> probability, timing, extent, and disagreement products</p>
                <p><span className="terminal-prompt">03</span> monitor forecast behavior</p>
                <p className="terminal-output"><Check size={14} /> familiarity diagnostics and evidence queue updated</p>
                <p className="terminal-cursor">ready <span>|</span></p>
              </div>
            </div>
          </div>
        </section>

        <section className="final-cta" data-reveal aria-labelledby="final-cta-title">
          <div>
            <p className="marketing-eyebrow">Start with a domain-specific pilot</p>
            <h2 id="final-cta-title">Bring your coastline, reference evidence, and study questions.</h2>
            <p>We will define the deployment contract, validation evidence, uncertainty products, and research limits together.</p>
          </div>
          <a className="primary-cta" href={PILOT_MAILTO}>
            Request a pilot <ArrowUpRight size={17} aria-hidden="true" />
          </a>
        </section>

        <p className="research-status"><strong>Research status.</strong> {RESEARCH_DISCLAIMER}</p>
      </main>
      <MarketingFooter />
      <ScrollReveal />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }} />
    </>
  );
}
