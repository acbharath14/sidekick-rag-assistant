# Meridian Logistics — Security Policy (Data Handling)

> Fictional documentation for demonstration purposes.

## Classification

Customer shipment data (names, addresses, contents) is **Confidential**.
Aggregated metrics (volumes, on-time rates) are **Internal**. API reference
documentation is **Public**.

## Storage

Confidential data must live in encrypted stores (AES-256 at rest, TLS 1.2+
in transit). Never copy customer addresses into tickets, chat, or dashboards.
Debug logs must redact the `recipient` object before writing.

## Access

Access follows least privilege: engineers get staging data by default; prod
access requires manager approval and expires after 7 days. Service accounts
used by the tracking API hold exactly one permission: `shipments:read` or
`shipments:write`, never both.

## Retention

Webhook delivery logs are kept 30 days, then purged. Shipment records are
kept 2 years for customs compliance, then anonymized. Backups inherit the
retention of the data they contain.

## Incident reporting

Suspected data exposure must be reported to `security@meridian.example`
within 1 hour of discovery. Do not attempt to contain a suspected breach by
deleting logs — preserve evidence and let the incident team direct the
response.

## Third parties

Vendors receiving Confidential data must sign the data processing addendum
and complete the annual security review. The approved vendor list lives in
the admin console; procurement outside that list needs CISO sign-off.
