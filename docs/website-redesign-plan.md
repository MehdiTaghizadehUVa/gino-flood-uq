# FloodUQ Website Redesign Plan

**Goal:** transform the gated research console at `fgn-lab.…` into a product-marketing site in the style of [inunda.ai](https://inunda.ai/), selling **FloodUQ** (calibrated uncertainty-quantified flood forecasting) as a product, with the existing run-submission console reframed as a gated **Demo**.

Status: proposal · Owner: Mehdi Taghizadeh · Scope: `apps/fgn-serving-frontend` + `deployment/fgn-serving` (Caddy/oauth2-proxy)

---

## 1. Executive summary

| | Today | Target |
|---|---|---|
| First impression | Login wall (oauth2-proxy) → console | Public cinematic marketing page |
| Audience | Approved researchers | Prospects, partners, reviewers → *then* approved demo users |
| Story | "Submit a run" | "Probabilistic flood intelligence, calibrated — try the demo" |
| Console | The whole site | One gated tab: **Demo** |
| Visual language | Light teal/slate dashboard | Apple-keynote dark: black, ice-blue accent, giant type |

The product story writes itself from what the system already does uniquely well — and differently from inunda.ai. Inunda sells **speed of a deterministic solver**. We sell **trustworthy uncertainty**: nested deep ensembles (3 checkpoints × 20 latents), decomposed epistemic/aleatoric variance, isotonic-calibrated exceedance probabilities, drift monitoring, HEC-RAS-referenced validation. "Not just a flood map — a flood map you can put error bars on."

---

## 2. Design language spec (extracted from inunda.ai, measured)

### 2.1 Palette (design tokens)

| Token | Value | Usage (measured on inunda.ai) |
|---|---|---|
| `--bg` | `#000000` | Page background (pure black) |
| `--text` | `#F5F5F7` | Primary text (Apple near-white) |
| `--text-secondary` | `#86868B` | Secondary/labels (Apple gray) |
| `--accent` | `#7FD6FF` | Ice-blue accent — headlines, eyebrows, data |
| `--accent-soft` | `#BFEAFF` | Light-blue tinted body text |
| `--accent-a33/-a21/-a13/-a11` | `rgba(127,214,255,α)` | Alpha ramp: borders, card tints, glows, chart fills |

Our adaptation: keep this exact system but consider shifting the accent slightly toward our existing cyan-depth ramp (`#19d9f2 → #0284c7`) so the marketing site and the demo's flood maps feel like one brand. Decision: **keep `#7FD6FF` family** — it *is* a water color and reads perfectly against black.

### 2.2 Typography (measured)

Font: inunda uses `SF Pro Display` (Apple-licensed — we cannot ship it). Use **Inter + Inter Display** via `next/font/google` (metrically closest free equivalent; also fixes our existing unloaded-Inter bug in `layout.tsx`).

| Role | Spec (from inunda.ai) | Our token |
|---|---|---|
| Hero wordmark | 184px/700, letter-spacing +52% (spread caps) | `--type-wordmark`, clamp(64px→184px) |
| Section headline | 62–68px/700, tracking −2% | `--type-h1`, clamp(36px→68px) |
| Sub-headline / statement | 46–56px/600, tracking −1%, often in accent | `--type-h2` |
| Feature statement | 50px/600 accent | `--type-statement` |
| Eyebrow label | 22px/600 UPPERCASE, tracking +42%, accent | `--type-eyebrow` |
| Card title | 25px/600 | `--type-card` |
| Micro label | 13px/600 UPPERCASE, tracking +30%, gray | `--type-micro` |
| Math/physics | Serif math stack (Cambria Math/STIX) | keep for UQ equations |
| Numeric/data | tabular-nums; JetBrains Mono for terminal | `--type-mono` |

### 2.3 Layout & motion grammar

- **Single-page scroll narrative** on `/` — 10–12 full-bleed sections, each: eyebrow → giant headline → content.
- **Numbered inventory pattern** (`01…06`) for pipeline/solutions sections.
- **Benchmark bars**: horizontal, labeled with source citations, "lower/higher is better" captions, IDEAL reference line.
- **Terminal mockup** section (they show a CLI; we show our REST/CSV submit or `curl` + the demo).
- **Motion:** scroll-triggered fade/translate reveals (IntersectionObserver + CSS transitions, `prefers-reduced-motion` respected). No heavy animation lib needed.
- **Hero canvas:** inunda uses a WebGL 3D flood scene. Phase-1 substitute: our real `calibrated_mean_wd_animation` GIF/MP4 rendered as a masked full-bleed loop with a black gradient overlay (zero new tech, real product output). Phase-3 option: deck.gl/three.js 3D terrain flyover.

---

## 3. Information architecture

```
PUBLIC (no auth)                        GATED (oauth2-proxy, unchanged policy)
──────────────────                      ─────────────────────────────────────
/                Marketing single-page  /demo             Console home (today's "/")
/#product        (anchor sections)      /demo/runs/[id]   Run detail (today /runs/[id])
/#performance                           /admin            Access control (unchanged)
/#how-it-works                          /admin/monitoring Drift monitoring (unchanged)
/#solutions                             /api/*            FastAPI (unchanged)
/contact         (or #contact anchor)
/login → /oauth2/sign_in?rd=/demo
```

Navigation (fixed translucent header): `FloodUQ ▪ Product · Performance · How it works · Solutions · —— · Demo (accent button)`. "Demo" is the single CTA that crosses the auth boundary.

**Route implementation:** Next.js route groups — `app/(marketing)/page.tsx` and `app/(demo)/demo/…`, each group with its own `layout.tsx` (marketing: dark tokens, no AppShell; demo: existing `AppShell`). Old paths `/runs/[id]` get permanent redirects to `/demo/runs/[id]` in `next.config` so bookmarked runs survive.

---

## 4. Page spec — marketing `/` (section by section)

Content mirrors inunda.ai's narrative arc but every claim is backed by an artifact our pipeline already produces.

| # | Section | Eyebrow / Headline | Content |
|---|---|---|---|
| 0 | **Nav** | — | Translucent black, blur backdrop, Demo CTA |
| 1 | **Hero** | — / `FLOODUQ` giant tracked wordmark | Tagline: "Probabilistic coastal flood forecasting with calibrated uncertainty." Sub: "Ensemble flood maps with error bars — in minutes, not days." Background: real calibrated mean-WD animation loop. SCROLL cue. |
| 2 | **The problem** | EVERY FORECAST IS WRONG / "The question is *how* wrong" | Single deterministic map vs our exceedance-probability map side-by-side. One sentence on why decision-makers need P(WD > 0.3 m), not one line. |
| 3 | **Nested ensembles** | THE METHOD / "60 futures per forecast" | Numbered 01–03: checkpoints (epistemic) × 02 latent samples (aleatoric) × 03 calibration. Serif math: `Var[H] = Var_θ E_z[H] + E_θ Var_z[H]`. This is our "governing physics" section. |
| 4 | **Calibration** | TRUSTWORTHY PROBABILITIES / "When we say 30%, we mean 30%" | Reliability diagram (already produced), isotonic calibration one-liner, PIT histogram. |
| 5 | **Validated performance** | VALIDATED / "Referenced against HEC-RAS" | inunda-style horizontal benchmark bars from our real eval: KGE 0.85, CRPS, hit-rate vs thresholds, per-event held-out storms (Ophelia '23, Isabel '03, Irene '11 figures exist). Citations to the paper. |
| 6 | **Speed** | GPU BENCHMARKS / "94-step ensemble forecast in ~10 minutes" | Real timings from run manifests (60-member, 5,904-cell mesh, single A100/4090). Honest scale note. |
| 7 | **Ecosystem** | THE PIPELINE / "Not just a model — a monitored system" | Numbered 01–06: Forcing ingest → Initial-condition library → FGNO ensemble → Calibration → Products (maps/animations/HDF5) → Drift monitoring & retraining candidates. Mirrors inunda's ecosystem section; ours is genuinely deeper on MLOps. |
| 8 | **Products** | MODEL OUTPUT / "Decision-ready products" | Gallery of real artifacts: exceedance maps, envelope/arrival-time maps, animation, cell inspector screenshot, forcing hydrograph. |
| 9 | **Demo CTA** | TRY IT / "Upload a hydrograph. Get a probabilistic flood map." | Terminal-style mock of the 3-step flow (upload CSV → run → products) + big `Request demo access` / `Open demo` buttons. |
| 10 | **Solutions** | WHAT WE OFFER / numbered 01–03 | Coastal forecast studies · Model UQ audits ("error bars for your existing model") · Custom domains/training. |
| 11 | **Footer** | Giant spread wordmark (inunda-style) | Contact email, UVA affiliation, paper link, **research disclaimer** (see §7.3), © |

Copy tone: short declarative sentences, numbers do the talking, zero adjectives without a citation.

---

## 5. Component architecture

### 5.1 New (marketing family — `app/(marketing)/components/`)

| Component | Purpose |
|---|---|
| `MarketingNav` | Fixed translucent nav + mobile sheet |
| `Section` | Full-bleed section wrapper: eyebrow, headline, reveal-on-scroll |
| `Eyebrow`, `Headline`, `Statement` | Type-system primitives (enforce scale) |
| `HeroWordmark` | Clamped giant tracked wordmark + media background |
| `BenchmarkBars` | Horizontal cited bars w/ ideal line (perf + speed sections) |
| `NumberedGrid` | 01–06 pipeline/solutions pattern |
| `MathBlock` | Serif math display |
| `TerminalMock` | Styled fake terminal for the demo-flow section |
| `ArtifactGallery` | Real product images w/ captions |
| `RevealObserver` | IntersectionObserver hook (respects reduced-motion) |
| `MarketingFooter` | Wordmark footer + disclaimer |

All styled-jsx + CSS custom properties; **no new runtime deps** (no Tailwind, no framer-motion — CSS transitions suffice). Static assets (benchmark JSON, gallery images) checked into `public/marketing/` at build time — the public site makes **zero API calls**.

### 5.2 Reused as-is (demo family — moved under `(demo)` group)

`AppShell` (nav gains "← flooduq.ai" home link), `PageHeader`, `MetricCard`, `StatusBadge`, `RunProgress`, `DataTable`, `ArtifactDrawer`, `SegmentedControl`, `InfoTip`, `CommandBar`, `Toolbar`, `Panel`, `EmptyState`, `FigureCard`, `MapFrame`, `InsightCaption`, `SectionHeader`, run-detail suite (`RunHeader`, `RunTabs`, `CellInspector`, `TimePlayer`), `sampleScenarios`.

### 5.3 Restyled

- `globals.css` → two token scopes: `[data-theme=marketing]` (black/ice-blue) and existing demo tokens. Demo gets a **dark-harmonized pass** (Phase 3) so crossing from marketing to demo isn't jarring: same accent family, dark surface `#0a0f14` instead of pure black (dashboards need elevation layers).
- `ResearchNotice` → also rendered on marketing footer + demo entry (compliance, §7.3).

---

## 6. Backend / infra changes (the real engineering)

### 6.1 Caddy auth-boundary split (`deployment/fgn-serving/Caddyfile`)

Today the catch-all `route` forward-auths **everything**. New routing:

```caddyfile
{$FGN_SITE_HOSTNAME} {
    handle /oauth2/*  { reverse_proxy oauth2-proxy:4180 … }        # unchanged

    route /api/*      { forward_auth oauth2-proxy … reverse_proxy api:8000 }   # unchanged

    @gated path /demo* /admin* /runs*
    route @gated {
        forward_auth oauth2-proxy:4180 { …401→sign_in redirect… }
        reverse_proxy frontend:3000
    }

    route {                       # everything else = PUBLIC marketing
        reverse_proxy frontend:3000
    }
}
```

Security notes:
- `/api/*` stays fully gated — the public page needs no API.
- Next.js internals: `/_next/*` assets must be public (they're shared by both trees; they contain no data, only code — acceptable, same as any SPA). Demo *data* only ever flows through `/api/*`, which stays gated. Verify no demo page pre-renders gated data into static HTML (all console data is client-fetched today — confirmed).
- Add basic rate-limit on the public route at Caddy or Cloudflare level.

### 6.2 Smoke tests (`deployment/fgn-serving/scripts/smoke_lab.sh`)

Extend: (a) `GET /` **without** auth returns 200 + contains `FLOODUQ`; (b) `GET /demo` without auth returns 302 → `/oauth2/sign_in`; (c) `GET /api/health` unauth → 302/401. The auth boundary becomes a tested invariant.

### 6.3 Domain (blocking for a real launch)

A marketing site on `fgn-lab.172.28.39.176.nip.io` with a self-signed cert undermines the entire exercise. Required: real domain (e.g. `flooduq.ai`/`.org`), Cloudflare DNS + tunnel already in place → free cert via DNS-01. Tracked as a Phase-4 gate. (Also fixes the browser trust warning permanently.)

### 6.4 CI/CD

No pipeline changes needed: same image, same workflows. The frontend CI job already builds the Next app; add `npm run lint` + a Playwright-less HTML smoke (grep marketing markers in `next build` output). Deploy remains CI → Images → Deploy Lab.

---

## 7. Cross-cutting concerns

### 7.1 SEO / metadata (new — site is public now)
`metadata` export per route group: title "FloodUQ — Probabilistic flood forecasting with calibrated uncertainty", OpenGraph card using an exceedance-map render, `robots.txt` allowing `/` and disallowing `/demo /admin /api`, sitemap with the single marketing URL. JSON-LD `SoftwareApplication` + `ScholarlyArticle` (paper citation).

### 7.2 Performance budget
Public page: LCP < 2.5 s on 4G. Hero animation ≤ 2 MB (re-encode GIF→AV1/H.264 `<video>` loop, `poster` for LCP), lazy-load below-fold galleries, zero blocking JS beyond Next runtime, fonts via `next/font` (self-hosted, `display: swap`). Lighthouse ≥ 90 perf/SEO/a11y as CI gate (manual initially).

### 7.3 Research-status disclaimer (non-negotiable)
The bundle ships `research_disclaimer: "Research only; not for emergency or operational decision use."` The marketing site may *sell the capability* but must not misrepresent readiness: persistent footer line + dedicated sentence in the Solutions section ("Research system — pilot engagements under research agreements"). This keeps the product framing honest and protects you institutionally (UVA affiliation displayed).

### 7.4 Accessibility
Black-bg sites fail a11y most often on gray-on-black: `#86868B` on `#000` = 5.1:1 (AA pass for normal text, keep ≥ 16px). Accent `#7FD6FF` on black = 12.5:1. Focus rings (accent, 2px), skip-link to main, reduced-motion path disables reveals + autoplaying video, all charts get text alternatives.

### 7.5 Analytics
Cloudflare Web Analytics (free, cookieless, no consent banner needed) on the public tree only.

---

## 8. Phased delivery

### Phase 0 — Foundations (0.5 day)
Design tokens (`[data-theme=marketing]`), `next/font` Inter/Inter Display/JetBrains Mono, route groups + layouts, redirects `/runs/* → /demo/runs/*`.
**Accept:** demo works unchanged at `/demo`; `/` renders empty marketing shell with tokens.

### Phase 1 — Marketing page (2–3 days)
Sections 0–5 + 11 (nav, hero, problem, method, calibration, performance, footer) with `Section`/`Eyebrow`/`Headline`/`BenchmarkBars`/`RevealObserver`. Static assets exported from real artifacts into `public/marketing/`.
**Accept:** visual parity with inunda-style reference at desktop/tablet/mobile breakpoints; Lighthouse ≥ 90; zero API calls on `/`.

### Phase 2 — Auth split + demo reframe (1 day)
Caddyfile split, smoke-test additions, `AppShell` home-link + "Demo" breadcrumb, ResearchNotice placement, deploy via normal pipeline, verify boundary invariants on the lab host.
**Accept:** anonymous `GET /` 200, `GET /demo` 302-to-login, `/api` gated; logged-in flow unchanged end-to-end (submit → run → products).

### Phase 3 — Full narrative + polish (2 days)
Sections 6–10 (speed, ecosystem, products gallery, demo CTA terminal, solutions), scroll reveals, hero video re-encode, demo dark-harmonization pass, OG images, JSON-LD.
**Accept:** full 11-section page; reduced-motion audit; a11y pass.

### Phase 4 — Launch gates (0.5 day + external)
Real domain + Cloudflare DNS-01 cert, analytics, `robots.txt`/sitemap, final content review (disclaimer wording), Lighthouse CI snapshot.
**Accept:** public URL with valid cert; demo access flow tested from a cold browser.

Total: **~6–7 engineering days** sequential; Phases 1 and 2 are independent and parallelizable.

### Explicit non-goals (v1)
WebGL 3D hero (Phase-5 candidate: deck.gl terrain + water), i18n switcher (inunda has EN/ES/中 — skip), CMS (copy lives in typed TSX constants), payments/self-serve signup (demo access remains admin-approved via existing `/admin` allowlist).

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Public exposure of previously-private surface | High | Caddy split reviewed + smoke-tested invariants (§6.2); `/api` untouched; rate-limit public route |
| "Product" framing vs research reality | Med | §7.3 disclaimer strategy; no operational claims; UVA affiliation explicit |
| nip.io domain undermines credibility | Med | Phase-4 real domain is a launch gate, not optional |
| Hero media tanks LCP | Med | `<video>` + poster + ≤2 MB budget; static hero fallback |
| SF Pro licensing | Low | Inter/Inter Display substitution decided upfront |
| Marketing/demo visual mismatch | Low | Shared accent family; Phase-3 dark-harmonization of demo |

---

## 10. Open decisions (need product-owner input)

1. **Name/wordmark:** `FLOODUQ` assumed — confirm final name before hero/OG assets.
2. **Accent color:** keep measured `#7FD6FF` (recommended) vs existing `#19d9f2` cyan ramp.
3. **Demo access CTA:** "Request access" mailto vs a small public form POSTing to a gated endpoint (form requires more backend; mailto ships day one).
4. **Real domain name** purchase + who owns DNS.
5. Whether Solutions section lists **pricing-style tiers** or stays engagement-based ("Contact us") — recommend the latter for a research system.
