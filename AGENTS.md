# Agent Instructions

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `MehdiTaghizadehUVa/gino-flood-uq`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default engineering-skill triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single root domain context plus repo-level ADRs. See `docs/agents/domain.md`.

## Project rules

- Treat the coastal FGN serving stack as research-only infrastructure, not operational flood guidance.
- Preserve the fixed-domain V1 scope unless the user explicitly asks for a broader deployment.
- Keep model-serving modules independent from FastAPI, Celery, OAuth, React, and SQL adapters.
- Prefer TDD vertical slices through public seams over tests of private tensor helpers.
