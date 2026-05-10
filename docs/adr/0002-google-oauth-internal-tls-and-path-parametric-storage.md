# 0002. Google OAuth, Internal TLS, And Path-Parametric Storage

## Status

Accepted

## Context

The service is gated research infrastructure for collaborators, not a public operational flood system. The lab PC will be reachable over VPN/LAN. Local disk capacity is a deployment constraint and full HDF5 artifacts can be large.

## Decision

Use Google OAuth through oauth2-proxy, enforce allowlist/admin/disclaimer policy in FastAPI, terminate HTTPS with Caddy internal TLS, and require `FGN_DATA_ROOT` for model bundles, artifacts, Postgres data, and backups.

## Consequences

- Collaborators need VPN/LAN access and must trust the internal Caddy CA.
- Auth remains outside inference code; FastAPI receives trusted identity headers.
- Deployment fails early if `FGN_DATA_ROOT` is not configured.
- Storage can move to a new large disk without changing code.
