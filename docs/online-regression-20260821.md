# Online regression ledger — 2026-08-21

This ledger records the shared online regression against the Research Agent
and Research Presentation Agent deployments. Secrets and public API keys are
intentionally excluded.

## Contract baseline

- Research Agent `/v1`: health, models, chat completions, and Responses all passed with valid authentication; invalid authentication returned `401`.
- Research Presentation Agent `/presentation/v1`: health, models, chat completions, and standard file upload passed; invalid authentication returned `401`.
- Presentation Agent does not expose `/v1/responses`; this is an intentional contract difference documented by that service.

## Bad cases tracked

- `BC-20260821-001`: a presentation request that explicitly asked for clarification only and said not to generate a PPT was routed into the PPT graph because the broad `PPT` keyword won over negative intent. The fix and regression test live in `research-presentation-agent`.
- `BC-20260821-002`: an online stage deck was delivered with a quality manifest issue saying the narrative did not end with a conclusion/next-step slide. The service preserves `content_ok=false`, `accepted_with_warnings=true`, and the issue in the manifest; acceptance must inspect that manifest rather than claim a fully clean deck.

## PPT acceptance observations

The audited online deck was a valid 6-slide PPTX with 6 unique visual pages
and substantive speaker notes on every slide. Package-level checks found no
missing slides or duplicate-only visuals. Render/overflow tooling could not be
started in the Windows shell because the bundled artifact renderer failed to
initialize; this limitation is recorded rather than treated as a clean visual
render assertion.
