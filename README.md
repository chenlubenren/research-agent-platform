# research-agent-platform

Research agent platform aligned to the PRD and tech spec, grounded by the vendored ARIS skill documents under `skills/`.

## Layout

- `router/`
- `state/`
- `graphs/`
- `connectors/`
- `memory/`
- `artifacts/`
- `ui/`

## Run

```bash
uv sync --no-editable
uv run uvicorn --app-dir src research_agent_platform.api:app --host 127.0.0.1 --port 8000
```

## OpenAI-compatible API

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

## Local chat and HITL

- Open `http://127.0.0.1:8000/chat`
- The page talks to the local research-agent workflow, not a plain echo proxy
- Human checkpoints are exceptional: the agent continues automatically unless the user explicitly asks it to wait or a material, unresolved choice requires the user. Approving accepts its recommended default; feedback can select another option.
- Research materials can be dragged onto the chat page or selected with the file picker. Uploads are stored under the current session workspace without starting a workflow.
- Files are written under `agent-workspace/local/<session_id>/` and exposed at `/workspace-files/...`
- Deployed clients can still pass `metadata.user_id` for identity metadata, but local artifact storage remains under `local/<session_id>/`.
- Task notes are session artifacts under `wiki/agent-notes/`; no model-generated research file is written to a separate project-level wiki.
- See `docs/workspace-layout.md` for the ARIS-style workspace layout

## Research workflow commands

- `/review` is the literature-review workflow, not peer review. It first writes a clarified, reproducible query protocol, then runs multi-query scholarly retrieval, relevance filtering, DOI/arXiv/title de-duplication, and a traceability gate. `bib/QUERY_YIELD.json` records per-query retained-paper metrics and `bib/REVIEW_DIRECTIONS.md` exposes evidence-grounded direction facets. A request to “写综述” additionally produces a thematic review draft; otherwise it hands evidence to `/idea`, `/plan`, or `/write`. Formal synthesis requires at least 10 admitted external or relevant local sources; otherwise it stops after the search and quality report instead of inventing a review.
- `/idea` generates candidate ideas, stress-tests novelty and feasibility, and writes the selected direction under `idea/FINAL_IDEA.md`. Wiki retrieval includes complete-paper coverage and a cross-paper evidence matrix. The deterministic novelty gate writes `Content/IDEA_NOVELTY_GATE.json` and marks insufficient evidence, direct Future Work, unsupported cross-paper differences, or missing falsifiability as `blocked_preliminary`; it does not create the experiment plan.
- `/plan` turns `FINAL_IDEA` or a directly supplied research objective into `plan/RESEARCH_BLUEPRINT.md`, `plan/EXPERIMENT_PLAN.md`, and `plan/EXECUTION_CHECKLIST.md`.
- `/write` freezes an attachment/session/workspace SourceSet, extracts stable evidence IDs, plans and drafts the paper, runs an independent self-review, produces an evidence-preserving revision, and writes deterministic citation/delivery reports before DOCX/PDF/TeX export.
- `/rebuttal` diagnoses an uploaded or generated manuscript when comments are absent; when reviewer comments are present, it maps each comment to paper evidence, drafts point-by-point replies, produces a revised manuscript and revision ledger, then verifies comment-ID coverage in `REBUTTAL_CLOSURE_REPORT.json`. Simulated peer review and deterministic pre-submission checks are internal stages; they are advisory quality controls, not journal decisions.
- `/code`, `/fig`, `/present`, and `/wiki` continue implementation planning, figure production, presentation generation, and persistent research memory. `/fig` freezes a FigureContract, builds a LayoutPlan, measures real text, and uses ELK before creating one shared DiagramRenderSpec. Data plots use reproducible Python; non-data figures default to editable Academic SVG; explicit structural editing uses native Draw.io; reference-image decomposition uses the optional Edit Banana sidecar. Every completed route emits SVG/PDF/PNG plus QA and a v3 delivery manifest.

Each command can run independently. When prior `/review` or `/idea` tasks exist in the same session, downstream commands prioritize their evidence map, research gaps, final idea, and research contract as handoff context.

`/review` uses relevant files under `*/uploads/` as local literature, then queries OpenAlex, Semantic Scholar, and arXiv with the brief's 4-6 Chinese/English topic variants; configured Web of Science and CNKI sources are added automatically. Admitted records receive stable IDs such as `[P001]`. When a provider exposes an explicit public PDF URL, the agent downloads and validates it into `bib/papers/`, subject to `REVIEW_DOWNLOAD_LIMIT`, `REVIEW_DOWNLOAD_MAX_MB`, and `REVIEW_DOWNLOAD_TIMEOUT_SECONDS`; `bib/LITERATURE_DOWNLOADS.json` records downloaded, unavailable, skipped, and failed items. A provider outage or unavailable full text is reported as a coverage limitation, never interpreted as a research gap, and no paywall is bypassed.

When a public PDF cannot be downloaded, `/review` also writes `bib/INSTITUTIONAL_ACCESS.md` and `bib/INSTITUTIONAL_ACCESS.json` with Tsinghua library gateway / off-campus access handoff links. Users must log in with their own institutional account. After downloading authorized PDFs, upload them with `target=bib`; PDF uploads to that target are stored in `bib/papers/` and can be reused by later `/review` runs as local literature.

Wiki reference expansion is disabled by default. Set `WIKI_REFERENCE_EXPANSION_ENABLED=true` only when direct-reference ingestion is intended. The opt-in path rejects loopback/private/link-local destinations after every redirect and enforces `WIKI_REFERENCE_SOURCE_LIMIT`, `WIKI_REFERENCE_LIMIT`, `WIKI_REFERENCE_DOWNLOAD_LIMIT`, `WIKI_REFERENCE_MAX_PDF_MB`, and `WIKI_REFERENCE_MAX_TOTAL_MB`. Metadata-only or failed references never count as full-text evidence.

For `/rebuttal`, upload the completed paper and optionally reviewer comments in the same session. Reviewer files whose names contain `review`, `reviewer`, `审稿`, or `评审` are routed to `rebuttal/uploads/`; paper files remain under `paper/uploads/`. With comments, the workflow performs point-by-point rebuttal and revision; without comments, it runs an advisory manuscript diagnosis. Explicit `--paper` and `--review` paths can override automatic selection; workspace fallback accepts only filenames clearly marked as final/accepted/终稿/定稿.

For `/write`, `--source attachments|selected|session|workspace` is optional. Automatic mode combines the latest upload batch with ranked paper, figure, plan, idea, bibliography, context, log, and code materials. `Content/PAPER_SOURCE_SELECTION.json` freezes the boundary; `paper/PAPER_EVIDENCE_MAP.json` stores extracted source/page/provenance records. The final exports use `paper/PAPER_REVISED.md`, not the first draft.

## Presentation workflow

- Presentation type and source scope are independent. Use `--type paper|stage` and `--source attachments|selected|session|workspace` when explicit control is needed.
- In automatic source mode, an explicit file/directory reference wins; otherwise the latest upload batch is used. The workflow searches the workspace only when requested or when no uploaded material exists.
- `/present --type paper --source attachments 论文汇报` uses only the latest uploaded PDF, Word, spreadsheet, image, or presentation batch.
- `/present --type stage --source workspace 阶段汇报` retrieves a limited, ranked set of relevant files from plans, figures, logs, notes, paper fragments, code, and other standard workspace directories.
- `/present --source selected \`paper/draft.pdf\` \`figures/result.png\`` freezes only the named files or directories as the task SourceSet.
- The workflow writes `presentation/SLIDES_OUTLINE.md`; it pauses only for a fully specified blocking choice that materially changes scope, cost, risk, claims, or an external commitment. Layout and wording preferences do not interrupt generation.
- Source selection and extracted assets are recorded in `Content/PRESENTATION_SOURCE_SELECTION.json`, `Content/PRESENTATION_SOURCE_INDEX.md`, and `Content/PRESENTATION_ASSETS.json`.
- PDF/Word/PPT images are extracted as original media. Reliable CSV/XLSX/Word tables become editable PowerPoint tables; PDF publication tables are retained as high-resolution original crops.
- `gpt-image-2` renders complete standalone narrative pages under `presentation/generated/`; original figures and tables are independently laid out on evidence pages and never mixed with Image-2 pages.
- `presentation/SPEAKER_NOTES.md` contains the generated talk script. The workflow embeds per-slide narration, timing, and transitions into native PowerPoint speaker notes and records the mapping in `Content/SPEAKER_NOTES.json`.
- `Content/SLIDE_SPECS.json` records the render policy and source asset used by each page; the final deck with speaker notes is written as `presentation/STAGE_REPORT.pptx` or `presentation/PAPER_TALK.pptx`.
- Choose a built-in template with `模板: stage-report` or `模板: paper-talk`. `PRESENTATION_TEMPLATE=auto` selects one from the requested mode.

## File uploads

- `POST /api/session/files` accepts multipart uploads with `files`, optional `session_id`, optional `user_id`, and `target`.
- Each upload response contains an `upload_batch_id`; `/present` uses the latest batch when source scope is automatic or `attachments`.
- `target=auto` routes reviewer-comment documents to `rebuttal/uploads/`, images to `figures/uploads/`, paper documents to `paper/uploads/`, presentations to `presentation/uploads/`, bibliography files to `bib/uploads/`, code to `code/uploads/`, and other files to `Content/uploads/`.
- A standard directory name can be passed as `target` to override automatic classification.
- Upload limits are configured with `UPLOAD_MAX_FILES` and `UPLOAD_MAX_FILE_MB`.

## Optional Seafile cloud workspace

- Set `CLOUD_SYNC_ENABLED=true` to mirror `agent-workspace/local/<session_id>/` to `SEAFILE_REMOTE_ROOT/<user_id>/<session_id>/`.
- Set `CLOUD_DELIVERY_REQUIRED=true` in local or deployed environments where a workflow must not report successful delivery until Seafile returns a share link.
- For Tsinghua Seafile, run `\.venv\Scripts\python.exe tools\configure_tsinghua_seafile.py` locally after changing any password previously sent through chat. The prompt hides the password, exchanges it for an API token, writes only the token to the ignored `.env`, and clears stored username/password fields.
- The Agent creates the remote session directory when the workspace is initialized, then uploads changed files after every workflow stage, checkpoint, final delivery, and user upload.
- `Content/CLOUD_SYNC.json` stores local file signatures so unchanged files are skipped on later syncs.
- Configure an existing library with `SEAFILE_REPO_ID`, or let the connector find/create `SEAFILE_REPO_NAME` when the account permits it.
- Prefer `SEAFILE_API_TOKEN`. `SEAFILE_USERNAME` and `SEAFILE_PASSWORD` are only a fallback for instances that support `/api2/auth-token/`; institutional single sign-on may require an API token or app-specific password.
- When `SEAFILE_SHARE_LINKS=true`, API responses expose `cloud_workspace.preview_url` and `cloud_workspace.download_url`. Workflow replies also include the same Tsinghua cloud URL after every synchronized artifact/checkpoint response.
- `POST /api/sessions/{session_id}/sync` and the sync icon in `/chat` trigger an immediate session upload and refresh the preview/download links.
- `GET /api/cloud/config` reports whether cloud sync is disabled, missing credentials, or ready for a real connection. The connector performs a one-way, incremental local-workspace-to-Seafile mirror; it does not automatically delete remote files or pull remote edits back into the local workspace.

## Streaming progress

- Workflow commands submitted from `/chat` return a task immediately and run in the background.
- The browser subscribes to `GET /api/tasks/{task_id}/events` via SSE and receives structured `snapshot`, `progress`, and `done` events. Polling remains the fallback when SSE is unavailable.
- Events contain route, stage, retrieval counts, artifact paths, cloud-sync status, checkpoint status, and errors. They are an auditable progress trace, not the model's hidden chain-of-thought.
- Ordinary questions submitted from the local chat page also use a lightweight `/chat` task, so the UI can show request analysis, model-call status, and answer completion in the same SSE panel. OpenAI-compatible `/v1/*` endpoints remain synchronous.
- The UI never receives hidden chain-of-thought. It receives only safe operational milestones and the final answer.
- Never commit `.env` or send the account password in chat. Real credentials remain in the local `.env`, which is excluded by `.gitignore`.

### Enable a real Seafile sync

The current checkout shows `disabled` because `.env` has no `CLOUD_SYNC_ENABLED` or Seafile credential. Run the local configuration helper in a terminal on the machine that runs the service:

```powershell
.\.venv\Scripts\python.exe tools\configure_tsinghua_seafile.py
```

It prompts for the account and hidden password, exchanges them for a Seafile API token, writes only that token to the ignored `.env`, and clears username/password fields. Restart the service after it succeeds. Verify `GET http://127.0.0.1:8000/api/cloud/config` reports `enabled: true` and `configured: true`; then create a session or click the cloud sync button. The service creates or reuses the `Research Agent` library, mirrors `agent-workspace/local/<session_id>/` to `research-agent/local/<session_id>/`, uploads changed files, and returns folder preview/download links.

This implementation is a one-way incremental mirror from local workspace to Seafile. It does not delete remote files or pull remote edits into the local workspace automatically.

## Deploy

```bash
uv sync --no-editable
uv run uvicorn --app-dir src research_agent_platform.api:app --host 0.0.0.0 --port 8000
```

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://YOUR_HOST:8000/v1")
```

## Configure ARIS grounding

ARIS skill documents are vendored into this repository under `skills/`, with shared templates under `templates/`. By default the platform resolves ARIS guidance from the project root, so the checkout is self-contained when shared.

Set `ARIS_REPO_ROOT` in `.env` only if you intentionally want to override the bundled skills with another ARIS-compatible directory.

## Literature providers

- `OpenAlex`, `Semantic Scholar`, and `arXiv` are enabled by default.
- `Web of Science` is enabled when `WOS_API_KEY` is set. The default endpoint is the Clarivate Starter API.
- `CNKI` is supported through an institutional or relay endpoint configured by `CNKI_SEARCH_ENDPOINT`.
- `CNKI` is intentionally adapter-based here; I did not add scraping because that is fragile and usually non-compliant.

## Upstream model

- If your relay exposes many model ids, set `UPSTREAM_MODEL` explicitly.
- If `UPSTREAM_MODEL` is empty, the platform now prefers chat-capable models such as `gpt-5.4-mini` instead of taking the first returned model blindly.
- Set `UPSTREAM_REVIEW_MODEL` to route isolated reviewer and meta-review calls separately. Reviewer failures are recorded as `needs_attention`/`REVIEW_UNAVAILABLE`, never converted into an empty clean review.
- `/fig` uses `IMAGE_MODEL`, which defaults to `gpt-image-2`, only for one optional text-free moodboard on mechanism/reference routes. Image generation is never the canonical source for labels, arrows, or scientific structure.
- Run `npm ci` to enable the locked ELK layout engine; the figure pipeline uses a deterministic grid fallback when Node/ELK is unavailable.
