# 0002. Google OAuth, Internal TLS, And Path-Parametric Storage

## Status

Accepted

## Context

The service is gated research infrastructure for collaborators, not a public operational flood system. The lab PC will be reachable over VPN/LAN. Local disk capacity is a deployment constraint and full HDF5 artifacts can be large.

## Decision

Use Google OAuth through oauth2-proxy, enforce allowlist/admin/disclaimer policy in FastAPI, terminate HTTPS with Caddy internal TLS, and require `FGN_DATA_ROOT` for model bundles, artifacts, Postgres data, and backups.

### Guest Console Amendment (2026-07)

The exact `/demo` route is public so visitors can configure scenarios and validate forcing CSVs before deciding to use shared compute. Public API access is limited to model-bundle metadata, the forcing template, and non-persistent forcing validation. Creating or reading runs, downloading run artifacts, inspecting run monitoring, and accessing administration remain authenticated and continue to enforce owner/admin policy in FastAPI.

Authentication is requested only when a visitor launches a run. Private history and run-detail URLs remain unavailable anonymously and become visible after authentication. The browser may keep an unsubmitted draft in tab-scoped session storage across the OAuth redirect; the server does not persist guest uploads.

## Consequences

- Collaborators need VPN/LAN access and must trust the internal Caddy CA.
- Auth remains outside inference code; FastAPI receives trusted identity headers.
- Anonymous visitors can explore the scenario workflow without gaining access to private runs or GPU submission endpoints.
- Public forcing validation is non-persistent and limited to 2 MiB per CSV.
- Deployment fails early if `FGN_DATA_ROOT` is not configured.
- Storage can move to a new large disk without changing code.
