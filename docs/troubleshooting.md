# Meridian Logistics — Troubleshooting Guide

> Fictional documentation for demonstration purposes.

## API returns 401 Unauthorized

Check the token expiry first: tokens lapse after 90 days and the API
responds with error code `token_expired`. Issue a new token in the admin
console and retry. If the token is fresh, verify the `Authorization` header
format — it must be `Bearer <token>`, not `Token <token>`.

## API returns 429 Too Many Requests

You hit the 100-requests-per-minute limit. Read the `Retry-After` header
and back off for that many seconds. Batch tracking calls with
`POST /v1/shipments/batch` (up to 50 IDs per call) instead of looping.

## API returns 5xx

1. Check the status page for an active incident.
2. Look at the `http_5xx_rate` dashboard panel; a spike confined to one
   availability zone suggests a bad deploy — roll back per the runbook.
3. If the spike is global, page the primary on-call.

## Shipment stuck in `exception`

`exception` means customs or weather intervened. Query
`GET /v1/shipments/{id}` and read the `exception_reason` field. For customs
holds, confirm the commercial invoice was attached at creation time —
missing paperwork is the most common cause. For weather, the ETA recalculates
automatically; do not create a duplicate shipment.

## Webhooks not arriving

Verify the callback URL is reachable from the public internet and responds
within 5 seconds. Check the `X-Meridian-Signature` verification on your side —
a signature mismatch means your endpoint rejects the payload before logging
it. Webhook deliveries retry with exponential backoff for 24 hours, then stop.

## Search returns no results

The tracking search index rebuilds every 15 minutes. Recently created
shipments may not appear until the next rebuild cycle. If older shipments are
missing, check that the shipment was created in the environment you are
querying — dev, staging, and prod have separate indexes.
