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
  plainLanguageGuide,
  COLLABORATION_MAILTO,
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
  name: "FloodUQ coastal flood probability and uncertainty modeling",
  serviceType: "Domain-specific coastal flood scenario modeling",
  description:
    "A managed service for comparing coastal flood scenarios, estimating calibrated flood probabilities and timing, and showing how closely plausible forecasts agree.",
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
            <p className="hero-kicker">Coastal flood scenarios with probability and uncertainty</p>
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
              Compare water-level and rainfall scenarios quickly, see the chance that water passes a chosen depth,
              understand why plausible forecasts differ, and identify where detailed simulation may add the most value.
            </p>
            <div className="hero-actions">
              <a className="primary-cta" href={COLLABORATION_MAILTO}>
                Discuss a collaboration <ArrowUpRight size={17} aria-hidden="true" />
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
              An expected-depth map shows where water may go and how deep it may become. A fuller picture also shows
              when water may arrive, the calibrated chance of crossing a selected depth, and where plausible forecasts
              remain close together or spread apart.
            </p>
          )}
          className="problem-section"
        >
          <DecisionRiskStory questions={decisionQuestions} />
          <div className="plain-language-guide">
            <MicroLabel>How we use these terms</MicroLabel>
            <NumberedGrid items={plainLanguageGuide} columns={4} />
          </div>
        </Section>

        <Section
          eyebrow="Service outcomes"
          title={<>See where and when flooding may develop,<br /><span>with uncertainty kept visible.</span></>}
          intro={(
            <p>
              Compare water-level and rainfall inputs, inspect individual locations, and trace every result back to the
              model and reference evidence that produced it.
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
              Review local forecast traces, compare raw model probabilities with calibrated probabilities, and trace
              each result to its scenario inputs, model version, calibration version, and review record.
            </p>
          </div>
        </Section>

        <Section
          eyebrow="Scenario capacity"
          title={<>Explore more alternatives<br /><span>within the same study cycle.</span></>}
          intro={(
            <p>
              FloodUQ generates a group of plausible maps directly, reducing the cost of testing changing assumptions
              instead of stopping at one flood map.
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
          eyebrow="Probability calibration"
          title={<>Turn raw ensemble probabilities into<br /><span>calibrated probabilities.</span></>}
          intro={(
            <p>
              A forecast group, often called an ensemble, estimates how often water passes a selected depth. Calibration
              maps that raw estimate using detailed simulations that were not used to train the forecast model. Its
              performance is then evaluated on held-out reference cases.
            </p>
          )}
          className="calibration-section"
        >
          <CalibrationStory notes={calibrationNotes} />
        </Section>

        <Section
          id="uncertainty"
          eyebrow="Why forecasts differ"
          title={<>See the range.<br /><span>Know what is driving it.</span></>}
          intro={(
            <p>
              FloodUQ separates model uncertainty (epistemic), which is disagreement between independently trained
              models, from outcome variability (aleatoric), which is the range of plausible results retained within one
              model.
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
              FloodUQ complements detailed physics simulations with fast groups of plausible forecasts, calibrated
              probabilities, clear explanations of forecast disagreement, and checks for unfamiliar scenarios.
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
              FloodUQ flags scenarios with unfamiliar inputs or unusually wide forecast differences for expert review,
              helping teams direct new detailed simulations toward the largest evidence gaps.
            </p>
          )}
          className="monitoring-section"
        >
          <MonitoringStory items={monitoringLoop} />
        </Section>

        <Section
          id="evidence"
          eyebrow="Portsmouth, Virginia / Deployment proof"
          title={<>From historical water levels and rainfall to<br /><span>calibrated flood probabilities.</span></>}
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
              Each deployment is built from the coastline's own terrain, water-level and rainfall scenarios, detailed
              reference simulations, evaluation criteria, and decision needs.
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
              from isolated model runs to comparable ranges, probabilities, and timing evidence.
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
              <h2>See flood probabilities and forecast agreement<br /><span>in action.</span></h2>
              <p>
                Use the Portsmouth deployment to submit water-level and rainfall inputs, follow the forecast, inspect
                expected-depth and probability maps, and see where plausible forecasts agree or differ.
              </p>
              <div className="hero-actions">
                <a className="primary-cta" href={COLLABORATION_MAILTO}>
                  Discuss a collaboration <ArrowUpRight size={17} aria-hidden="true" />
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
                <p><span className="terminal-prompt">01</span> submit water-level and rainfall inputs</p>
                <p className="terminal-output"><Check size={14} /> input quality and similarity to model-building evidence confirmed</p>
                <p><span className="terminal-prompt">02</span> generate flood depth and probability results</p>
                <p className="terminal-output"><Check size={14} /> depth, chance, timing, affected area, and forecast range ready</p>
                <p><span className="terminal-prompt">03</span> review forecast agreement and evidence gaps</p>
                <p className="terminal-output"><Check size={14} /> priority scenarios preserved for expert review</p>
                <p className="terminal-cursor">ready to compare <span>|</span></p>
              </div>
            </div>
          </div>
        </section>

        <section className="final-cta" data-reveal aria-labelledby="final-cta-title">
          <div>
            <p className="marketing-eyebrow">Collaborate on a domain-specific deployment</p>
            <h2 id="final-cta-title">Bring your coastline, reference evidence, and study questions.</h2>
            <p>Together, we will define the domain, detailed reference simulations, depth thresholds, evaluation plan, and service configuration.</p>
          </div>
          <a className="primary-cta" href={COLLABORATION_MAILTO}>
            Discuss a collaboration <ArrowUpRight size={17} aria-hidden="true" />
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
