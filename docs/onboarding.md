# Meridian Logistics — New Engineer Onboarding

> Fictional documentation for demonstration purposes.

## Week 1

- Get laptop + accounts (SSO covers the admin console, PagerDuty, CI).
- Read the API reference and the security policy's data-handling section —
  you will handle Confidential shipment data from day one.
- Shadow the on-call engineer for one incident review.

## Week 2

- Ship a docs fix to the troubleshooting guide (small, safe first PR).
- Complete the staging deploy walkthrough with your buddy: deploy the
  tracking service to staging, run the smoke suite, then roll it back.
- Request prod access only if your team needs it; it expires after 7 days.

## Week 3

- Take your first on-call shadow shift. Know the escalation path: primary
  on-call → team lead → VP engineering.
- Pick up a `good-first-issue` from the tracking service backlog.

## Buddy system

Every new engineer is paired with a buddy for the first month. Your buddy
reviews your first three PRs and walks you through one full deploy cycle
(dev → staging → prod) before you deploy solo.

## Definition of done

A task is done when: code merged, tests green, docs updated if behavior
changed, and the change is observable in staging. "Works on my machine" is
not done.
