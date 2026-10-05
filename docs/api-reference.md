# Meridian Logistics — Shipment Tracking API Reference

> Fictional documentation for demonstration purposes. Any resemblance to real
> systems is coincidental.

## Authentication

All requests require a bearer token in the `Authorization` header:

```
Authorization: Bearer <api-token>
```

Tokens are issued per service account in the Meridian admin console and
expire after 90 days. Rotate tokens before expiry; the API returns
`401 Unauthorized` with error code `token_expired` when a token lapses.

## Create a shipment

`POST /v1/shipments`

Request body:

| Field | Type | Required | Description |
|---|---|---|---|
| `origin` | string | yes | Origin facility code, e.g. `ORD-04` |
| `destination` | string | yes | Destination facility code |
| `weight_kg` | number | yes | Billable weight, max 1200 kg per parcel |
| `priority` | string | no | `standard` (default) or `express` |

Response `201 Created`:

```json
{
  "shipment_id": "SHP-88213",
  "status": "label_created",
  "eta_days": 4
}
```

Express shipments are guaranteed within 2 business days; standard within 5.

## Track a shipment

`GET /v1/shipments/{shipment_id}`

Returns the current status, location history, and ETA. Statuses in order:
`label_created` → `in_transit` → `out_for_delivery` → `delivered`.
A shipment may also enter `exception` if customs or weather intervene.

## Webhooks

Register a callback URL with `POST /v1/webhooks`. Meridian POSTs a JSON
payload on every status transition and retries failed deliveries with
exponential backoff for up to 24 hours. Webhook payloads are signed with
HMAC-SHA256; verify the `X-Meridian-Signature` header before trusting them.

## Rate limits

100 requests per minute per token. Exceeding the limit returns `429` with a
`Retry-After` header in seconds.
