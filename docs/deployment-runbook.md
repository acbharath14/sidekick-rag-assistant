# Meridian Logistics — Deployment Runbook (Tracking Service)

> Fictional documentation for demonstration purposes.

## Environments

| Environment | URL | Purpose |
|---|---|---|
| dev | https://tracking-dev.meridian.example | Engineer testing |
| staging | https://tracking-staging.meridian.example | Pre-prod validation |
| prod | https://tracking.meridian.example | Customer traffic |

## Deploy checklist

1. Merge to `main` only after CI is green (unit + contract tests).
2. Tag the release: `git tag tracking-vX.Y.Z`.
3. Deploy to staging first: `deploy tracking --env staging --tag vX.Y.Z`.
4. Run the smoke suite: `smoke --env staging --suite tracking`.
5. Soak for 30 minutes; watch the `http_5xx_rate` dashboard panel.
6. Promote to prod: `deploy tracking --env prod --tag vX.Y.Z`.
7. Verify with `GET /v1/health` on prod; expect `{"status":"ok"}`.

## Rollback

Rollbacks use the previous tag: `deploy tracking --env prod --tag vX.Y.(Z-1)`.
A rollback does not require a change ticket but must be announced in
`#logistics-ops`. Database migrations are forward-only; rolling back code
never rolls back schema.

## Feature flags

New endpoints ship behind flags defined in `flags/tracking.yaml`. Flags
default to off in prod and are flipped via the admin console after the
deploy soaks. Never remove a flag in the same release that introduces it.

## On-call

Primary on-call is paged via PagerDuty for `http_5xx_rate > 1%` sustained
5 minutes or p99 latency above 800 ms. The runbook for a full outage starts
at the troubleshooting guide's "API returns 5xx" section.
