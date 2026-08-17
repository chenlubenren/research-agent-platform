from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageDefinition:
    name: str
    title: str
    instruction: str
    artifact_path: str
    artifact_kind: str
    required_sections: list[str]
    skill_paths: list[str] = field(default_factory=list)
    model_role: str = "default"
    hitl: bool = False
    checkpoint_title: str = ""


@dataclass(frozen=True)
class WorkflowDefinition:
    command: str
    title: str
    description: str
    stage_definitions: list[StageDefinition]


def _skills(*names: str) -> list[str]:
    return [f"skills/{name}/SKILL.md" for name in names]


def workflow_registry() -> dict[str, WorkflowDefinition]:
    return {
        "/plan": WorkflowDefinition(
            command="/plan",
            title="Research Planning Workflow",
            description="Turn a selected idea or research objective into a claim-driven experiment and execution plan.",
            stage_definitions=[
                StageDefinition(
                    name="blueprint",
                    title="Research Blueprint",
                    instruction=(
                        "Read the selected idea and literature evidence available in the session workspace. If no prior /idea "
                        "artifact exists, treat the user's objective as the selected direction. Define the problem, hypotheses, "
                        "intended contribution, claim-to-evidence requirements, assumptions, dependencies, and failure criteria. "
                        "Do not generate alternative research topics or repeat a broad literature review."
                    ),
                    artifact_path="plan/RESEARCH_BLUEPRINT.md",
                    artifact_kind="plan",
                    required_sections=[
                        "Objective",
                        "Selected Direction",
                        "Research Hypotheses",
                        "Claim to Evidence Requirements",
                        "Dependencies and Assumptions",
                        "Failure Criteria",
                    ],
                    skill_paths=_skills(
                        "research-pipeline",
                        "research-refine-pipeline",
                        "research-refine",
                        "system-profile",
                    ),
                ),
                StageDefinition(
                    name="experiment_plan",
                    title="Experiment Plan",
                    instruction=(
                        "Convert the research blueprint into a claim-driven experiment design. Specify datasets, baselines, "
                        "metrics, controls, sanity checks, ablations, resource budget, launch order, and stop conditions. Reuse "
                        "the literature evidence produced by /review when available; clearly mark missing evidence instead of "
                        "inventing citations or results."
                    ),
                    artifact_path="plan/EXPERIMENT_PLAN.md",
                    artifact_kind="plan",
                    required_sections=[
                        "Claims Under Test",
                        "Datasets and Inputs",
                        "Baselines",
                        "Metrics",
                        "Must-Run Experiments",
                        "Ablations and Sanity Checks",
                        "Budget and Resources",
                        "Launch Order and Stop Conditions",
                    ],
                    skill_paths=_skills(
                        "experiment-plan",
                        "experiment-bridge",
                        "ablation-planner",
                        "experiment-audit",
                        "monitor-experiment",
                        "training-check",
                    ),
                ),
                StageDefinition(
                    name="execution_checklist",
                    title="Execution Checklist",
                    instruction=(
                        "Translate the blueprint and experiment plan into an operational checklist with milestones, deliverables, "
                        "owners, dependencies, result-capture requirements, necessary approval gates, and immediate next actions."
                    ),
                    artifact_path="plan/EXECUTION_CHECKLIST.md",
                    artifact_kind="plan",
                    required_sections=[
                        "Deliverables",
                        "Per-Stage Checklist",
                        "Approval Matrix",
                        "Immediate Next Actions",
                    ],
                    skill_paths=_skills(
                        "research-pipeline",
                        "experiment-bridge",
                        "experiment-plan",
                        "experiment-queue",
                        "monitor-experiment",
                        "research-wiki",
                    ),
                ),
            ],
        ),
        "/idea": WorkflowDefinition(
            command="/idea",
            title="Idea Discovery Workflow",
            description="Generate, challenge, verify, and select research ideas using literature evidence.",
            stage_definitions=[
                StageDefinition(
                    name="idea_candidates",
                    title="Idea Candidates",
                    instruction=(
                        "Use the user's objective and any /review evidence in the session to generate three distinct research "
                        "ideas after first reading the supplied Cross-Paper Evidence Matrix and Evidence Coverage. When no literature review exists, perform only the targeted novelty search supplied in context. "
                        "For each idea state the problem, mechanism, novelty thesis, expected contribution, feasibility, and main "
                        "risk, candidate type, supporting Paper IDs and Evidence IDs, author Future Work overlap, closest prior work, concrete difference, and novelty_status. "
                        "Generate one direct replication, one cross-paper combination, and one mechanism-or-question candidate. Do not write an experiment plan. Recommend one candidate and continue automatically. Use a blocking "
                        "Decision Required only when directions entail materially different scope, cost, risk, or external commitments "
                        "that cannot be resolved from evidence; ordinary topic preferences are not blocking."
                    ),
                    artifact_path="idea/IDEA_CANDIDATES.md",
                    artifact_kind="report",
                    required_sections=[
                        "Problem Frame",
                        "Candidate Ideas",
                        "Comparative Assessment",
                        "Recommended Candidate",
                        "Risks and Unknowns",
                        "Decision Required",
                    ],
                    skill_paths=_skills(
                        "platform-idea-generation",
                    ),
                    model_role="idea_generator",
                    hitl=True,
                    checkpoint_title="Topic Selection Approval",
                ),
                StageDefinition(
                    name="idea_verification",
                    title="Idea Verification",
                    instruction=(
                        "Act as an independent critic of the recommended or user-selected candidate. Test the novelty claim "
                        "against the supplied literature, identify the closest prior work, search for disconfirming evidence, "
                        "check whether it is merely author Future Work or an already implemented method, evaluate cross-paper support, "
                        "and state what remains uncertain. Do not expand this into a "
                        "full experiment plan."
                    ),
                    artifact_path="idea/IDEA_VERIFICATION.md",
                    artifact_kind="report",
                    required_sections=[
                        "Candidate Under Review",
                        "Closest Prior Work",
                        "Novelty Stress Test",
                        "Feasibility Stress Test",
                        "Disconfirming Evidence",
                        "Unresolved Questions",
                        "Verification Verdict",
                    ],
                    skill_paths=_skills(
                        "platform-idea-critique",
                    ),
                    model_role="idea_critic",
                ),
                StageDefinition(
                    name="final_idea",
                    title="Final Idea",
                    instruction=(
                        "Consolidate the locked candidate and verification findings into a professional, readable Chinese "
                        "research Idea for graduate and doctoral researchers. Explain why the problem matters, what the closest "
                        "evidence-backed gap is, how the proposed mechanism addresses it, what the dominant contribution is, and "
                        "how the idea could be falsified. Keep facts, evidence-backed inference, and unverified assumptions "
                        "clearly separated. Include a concise but concrete paper-style experiment section covering evidence-backed "
                        "datasets, baselines, metrics, experiment blocks, ablations, and failure criteria. Keep implementation "
                        "commands, exact runtime configuration, and the complete execution plan in /plan. If fewer than five complete readable papers are available, "
                        "or direct Future Work overlap or unsupported novelty remains, mark the document status as blocked_preliminary instead of claiming a new-paper contribution."
                    ),
                    artifact_path="idea/FINAL_IDEA.md",
                    artifact_kind="report",
                    required_sections=[
                        "摘要",
                        "1. 引言",
                        "2. 相关工作",
                        "3. 研究问题与核心假设",
                        "4. 方法思路",
                        "5. 实验方案",
                        "6. 预期贡献与可证伪预测",
                        "7. 局限、风险与不确定性",
                        "创新性判定",
                        "8. 结论与下一步",
                        "参考文献与证据",
                    ],
                    skill_paths=_skills(
                        "platform-idea-finalization",
                    ),
                    model_role="idea_finalizer",
                ),
            ],
        ),
        "/code": WorkflowDefinition(
            command="/code",
            title="Experiment Bridge Workflow",
            description="Turn an approved idea into implementation, validation, and handoff materials.",
            stage_definitions=[
                StageDefinition(
                    name="implementation_plan",
                    title="Implementation Plan",
                    instruction="Turn the approved idea into a concrete implementation brief: what to change, what to keep, which files to edit, how to configure the environment, which scripts to run, what data to touch, and how to validate success locally before any broader launch. Include explicit assumptions, dependencies, and a failure fallback. Choose a conservative launch default automatically. Use a blocking Decision Required only for unresolved cost, safety, data-access, or external-commitment choices that the agent cannot authorize.",
                    artifact_path="code/IMPLEMENTATION_PLAN.md",
                    artifact_kind="plan",
                    required_sections=[
                        "Objective and Scope",
                        "Repo Inputs",
                        "Files to Create or Modify",
                        "Execution Plan",
                        "Validation Plan",
                        "Launch Risks",
                        "Decision Required",
                    ],
                    skill_paths=_skills(
                        "experiment-bridge",
                        "run-experiment",
                        "experiment-queue",
                        "serverless-modal",
                        "vast-gpu",
                        "qzcli",
                        "meta-apply",
                        "system-profile",
                    ),
                    hitl=True,
                    checkpoint_title="Experiment Launch Approval",
                ),
                StageDefinition(
                    name="launch_runbook",
                    title="Launch Runbook",
                    instruction="Write the exact runbook for running the implementation end to end: commands, order, environment variables, logs, expected outputs, rollback points, retry rules, and how to capture results for later review. Make it usable by a human operator without extra context.",
                    artifact_path="code/LAUNCH_RUNBOOK.md",
                    artifact_kind="plan",
                    required_sections=[
                        "Prerequisites",
                        "Execution Commands",
                        "Logging and Outputs",
                        "Result Capture",
                        "Failure Handling",
                        "Post-Run Review",
                    ],
                    skill_paths=_skills(
                        "run-experiment",
                        "experiment-queue",
                        "monitor-experiment",
                        "training-check",
                        "serverless-modal",
                        "vast-gpu",
                        "feishu-notify",
                    ),
                ),
                StageDefinition(
                    name="local_collaboration",
                    title="Local Collaboration Guide",
                    instruction="Describe how the human and local tools should collaborate on code changes, file handoffs, logs, checkpoints, and experiment results. Make the file flow explicit so local testing and follow-up edits are straightforward.",
                    artifact_path="code/LOCAL_COLLAB.md",
                    artifact_kind="note",
                    required_sections=[
                        "Workspace Layout",
                        "Human Tasks",
                        "Agent Tasks",
                        "Expected File Handoffs",
                        "Verification Checklist",
                        "How to Resume",
                    ],
                    skill_paths=_skills(
                        "research-pipeline",
                        "research-wiki",
                        "wiki-enrich",
                        "monitor-experiment",
                        "system-profile",
                    ),
                ),
            ],
        ),
        "/fig": WorkflowDefinition(
            command="/fig",
            title="Figure Generation Workflow",
            description="Generate a finished research figure from explicit data or with gpt-image-2.",
            stage_definitions=[
                StageDefinition(
                    name="figure_inventory",
                    title="Figure Inventory",
                    instruction=(
                        "Identify the single highest-value final figure and its evidence dependencies. Inspect the supplied "
                        "workspace context. If explicit tabular data is available, require a precise code-rendered chart; "
                        "otherwise specify a scientific illustration suitable for gpt-image-2."
                    ),
                    artifact_path="figures/FIGURE_INVENTORY.md",
                    artifact_kind="report",
                    required_sections=[
                        "Required Figures",
                        "Data Dependencies",
                        "Narrative Purpose",
                        "Priority Order",
                        "Render Mode",
                    ],
                    skill_paths=_skills(
                        "paper-figure",
                        "figure-spec",
                        "figure-description",
                        "mermaid-diagram",
                        "paper-illustration-image2",
                    ),
                ),
                StageDefinition(
                    name="figure_briefs",
                    title="Figure Briefs",
                    instruction=(
                        "Write a production brief for the selected final figure. Declare exactly one render mode: `code` "
                        "for explicit numeric data or `image2` for a scientific illustration. For code mode name the "
                        "source file, columns, chart type, axes, units, and caption. For image2 mode define composition, "
                        "labels, visual hierarchy, and scientific constraints."
                    ),
                    artifact_path="figures/FIGURE_BRIEFS.md",
                    artifact_kind="plan",
                    required_sections=[
                        "Per-Figure Brief",
                        "Data Fields",
                        "Design Notes",
                        "Caption Drafts",
                        "Render Mode",
                    ],
                    skill_paths=_skills(
                        "paper-figure",
                        "figure-spec",
                        "figure-description",
                        "paper-illustration",
                        "paper-illustration-image2",
                        "render-html",
                    ),
                ),
            ],
        ),
        "/write": WorkflowDefinition(
            command="/write",
            title="Paper Writing Workflow",
            description="Freeze research evidence, draft and independently review the paper, revise it, and run delivery gates.",
            stage_definitions=[
                StageDefinition(
                    name="paper_evidence",
                    title="Paper Evidence Contract",
                    instruction=(
                        "Freeze the writing SourceSet and extract stable evidence records with source paths, pages, evidence "
                        "types, provenance, and excerpts. This stage is generated deterministically by the platform."
                    ),
                    artifact_path="paper/PAPER_EVIDENCE_MAP.json",
                    artifact_kind="manifest",
                    required_sections=[
                        "Evidence ID",
                        "Source Path",
                        "Page",
                        "Evidence Type",
                        "Provenance",
                        "Excerpt",
                    ],
                    skill_paths=_skills(
                        "paper-claim-audit",
                        "citation-audit",
                        "result-to-claim",
                    ),
                ),
                StageDefinition(
                    name="paper_plan",
                    title="Paper Plan",
                    instruction=(
                        "Produce an executable paper plan from the frozen evidence contract: target venue, core contribution, "
                        "outline, section responsibilities, paragraph jobs, claim-to-evidence IDs, missing evidence, figure/table "
                        "needs, terminology, and writing order. Distinguish experimental papers from review articles. Select a "
                        "defensible venue and story automatically. Use a blocking Decision Required only when mutually exclusive "
                        "claims or external submission commitments cannot be resolved from the frozen evidence."
                    ),
                    artifact_path="paper/PAPER_PLAN.md",
                    artifact_kind="plan",
                    required_sections=[
                        "Target Story",
                        "Submission Target",
                        "Section Outline",
                        "Section Responsibilities and Paragraph Jobs",
                        "Claim to Evidence Map",
                        "Terminology and Claim Boundaries",
                        "Writing Risks",
                        "Questions for Human Review",
                        "Decision Required",
                    ],
                    skill_paths=_skills(
                        "paper-plan",
                        "paper-writing",
                        "paper-write",
                        "writing-systems-papers",
                        "paper-claim-audit",
                        "citation-audit",
                        "result-to-claim",
                    ),
                    hitl=True,
                    checkpoint_title="Writing Outline Approval",
                ),
                StageDefinition(
                    name="narrative_report",
                    title="Narrative Report",
                    instruction=(
                        "Write a narrative handoff document that a coauthor can use directly: problem, method, evidence IDs, "
                        "results, contradictions, what is missing, limitations, and how the current work should be framed. "
                        "Separate demonstrated results from intended contributions and future work."
                    ),
                    artifact_path="paper/NARRATIVE_REPORT.md",
                    artifact_kind="report",
                    required_sections=[
                        "Problem Statement",
                        "Core Claim",
                        "Method Summary",
                        "Key Results",
                        "Evidence and Citation Boundaries",
                        "Limitations",
                        "Open Gaps",
                    ],
                    skill_paths=_skills(
                        "claims-drafting",
                        "result-to-claim",
                        "analyze-results",
                        "paper-claim-audit",
                        "research-review",
                        "citation-audit",
                    ),
                ),
                StageDefinition(
                    name="draft_sections",
                    title="Draft Sections",
                    instruction=(
                        "Write a complete first-pass paper aligned with the approved outline and frozen evidence. Use continuous "
                        "prose rather than outline bullets, keep one controlling idea per paragraph, preserve supplied equations, "
                        "numbers, tables, figures, and citations, and keep unsupported content as [AUTHOR INPUT NEEDED]. "
                        "Do not invent bibliography entries or evidence."
                    ),
                    artifact_path="paper/PAPER_DRAFT.md",
                    artifact_kind="report",
                    required_sections=[
                        "Title",
                        "Abstract",
                        "Introduction",
                        "Related Work",
                        "Method",
                        "Experiments",
                        "Limitations",
                        "Conclusion",
                        "References",
                        "Next Revision Steps",
                    ],
                    skill_paths=_skills(
                        "paper-writing",
                        "paper-write",
                        "writing-systems-papers",
                        "paper-claim-audit",
                        "citation-audit",
                        "overleaf-sync",
                        "paper-compile",
                        "claims-drafting",
                    ),
                ),
                StageDefinition(
                    name="paper_self_review",
                    title="Paper Self Review",
                    instruction=(
                        "Independently audit PAPER_DRAFT.md against PAPER_EVIDENCE_MAP.json and PAPER_PLAN.md. Assign stable "
                        "issue IDs and major/minor severity. Check argument, claim-evidence alignment, citation closure, coverage, "
                        "structure, terminology, paragraph flow, overclaiming, contradictions, limitations, figure/table "
                        "traceability, and venue requirements. Do not perform a new experiment or literature search."
                    ),
                    artifact_path="paper/PAPER_SELF_REVIEW.md",
                    artifact_kind="review",
                    required_sections=[
                        "Quality Scores",
                        "Major Issues",
                        "Minor Issues",
                        "Claim and Evidence Findings",
                        "Citation Findings",
                        "Structure and Venue Findings",
                        "Revision Actions",
                    ],
                    skill_paths=_skills(
                        "paper-claim-audit",
                        "citation-audit",
                        "research-review",
                        "integrity-forensics",
                        "experiment-audit",
                    ),
                ),
                StageDefinition(
                    name="paper_revision",
                    title="Paper Revision",
                    instruction=(
                        "Revise PAPER_DRAFT.md using every issue in PAPER_SELF_REVIEW.md. Make the smallest evidence-supported "
                        "change that resolves each issue, preserve facts, numbers, equations, citations, figure/table references, "
                        "and provenance, and retain unresolved scientific gaps as [AUTHOR INPUT NEEDED]. Return the complete "
                        "revised manuscript, not a change summary."
                    ),
                    artifact_path="paper/PAPER_REVISED.md",
                    artifact_kind="report",
                    required_sections=[
                        "Title",
                        "Abstract",
                        "Introduction",
                        "Related Work",
                        "Method",
                        "Experiments",
                        "Limitations",
                        "Conclusion",
                        "References",
                        "Unresolved Author Inputs",
                    ],
                    skill_paths=_skills(
                        "paper-writing",
                        "paper-write",
                        "paper-claim-audit",
                        "citation-audit",
                        "claims-drafting",
                    ),
                ),
            ],
        ),
        "/review": WorkflowDefinition(
            command="/review",
            title="Literature Review Workflow",
            description="Search, organize, and synthesize literature evidence for idea discovery and research planning.",
            stage_definitions=[
                StageDefinition(
                    name="research_brief",
                    title="Research Brief",
                    instruction=(
                        "Turn the user's request into a bounded, reproducible retrieval protocol before any external search. "
                        "Define the research question, review type, technical concept groups, time range, publication policy, "
                        "and explicit inclusion/exclusion criteria. Under Search Strategy, provide exactly 4-6 query variants "
                        "as standalone lines `Q1: ...` through `Q6: ...`: include the core topic, canonical English terminology, "
                        "domain aliases, a recent-review query, and a foundational-work query. Plan a recent/foundational split. "
                        "Do not claim retrieval coverage or provider success before the search has run."
                    ),
                    artifact_path="bib/RESEARCH_BRIEF.md",
                    artifact_kind="report",
                    required_sections=[
                        "Research Question",
                        "Scope",
                        "Concept Groups",
                        "Search Strategy",
                        "Foundational and Recent Split",
                        "Source Coverage",
                        "Inclusion and Exclusion Criteria",
                        "Retrieval Limitations",
                    ],
                    skill_paths=_skills(
                        "research-lit",
                        "prior-art-search",
                        "openalex",
                        "semantic-scholar",
                        "arxiv",
                        "deepxiv",
                        "comm-lit-review",
                    ),
                ),
                StageDefinition(
                    name="literature_synthesis",
                    title="Literature Synthesis",
                    instruction=(
                        "Use only the admitted retrieval records and extracted local literature. Start with a paper evidence table "
                        "covering stable ID or local path, year, venue/status, problem, method, data or scenario, key finding, "
                        "limitation, evidence depth, and relevance. Then synthesize by technical axis rather than search order, "
                        "separating foundational from recent work and formal publications from preprints. Compare methods, datasets, "
                        "baselines, metrics, contradictions, and deployment evidence. Every paper-level factual statement must cite "
                        "a supplied stable ID such as [P001], or an exact local source path with page when available. Distinguish "
                        "metadata/abstract evidence from full-text evidence. Provider failures are retrieval limitations, never domain facts."
                    ),
                    artifact_path="bib/LITERATURE_REVIEW.md",
                    artifact_kind="report",
                    required_sections=[
                        "Executive Summary",
                        "Paper Evidence Table",
                        "Research Landscape",
                        "Foundational and Recent Work",
                        "Methods and Datasets",
                        "Baselines and Metrics",
                        "Key Findings",
                        "Contradictions and Limitations",
                        "References",
                    ],
                    skill_paths=_skills(
                        "research-lit",
                        "comm-lit-review",
                        "research-review",
                        "citation-audit",
                    ),
                ),
                StageDefinition(
                    name="evidence_map",
                    title="Evidence Map",
                    instruction=(
                        "Convert the literature synthesis into an evidence map for downstream /idea and /plan workflows. For each "
                        "important claim or open question, list supporting evidence, opposing or weak evidence, methods, datasets, "
                        "baselines, metrics, confidence, evidence depth, and source pointers. Every claim row must contain admitted "
                        "stable IDs or exact local paths. Calibrate confidence from independent source count, agreement, and whether "
                        "the evidence is full text or metadata/abstract only; do not infer confidence from citation count alone."
                    ),
                    artifact_path="bib/EVIDENCE_MAP.md",
                    artifact_kind="report",
                    required_sections=[
                        "Claim Evidence Matrix",
                        "Methods",
                        "Datasets",
                        "Baselines",
                        "Metrics",
                        "Contradictory Evidence",
                        "Confidence and Source Pointers",
                    ],
                    skill_paths=_skills(
                        "research-review",
                        "citation-audit",
                        "result-to-claim",
                    ),
                ),
                StageDefinition(
                    name="research_gaps",
                    title="Research Gaps",
                    instruction=(
                        "Identify defensible research gaps, unresolved disputes, missing comparisons, under-tested assumptions, "
                        "and practical opportunities from the evidence map. Separate well-supported gaps from speculative "
                        "opportunities and provide explicit handoff guidance for /idea and /plan. Cite every supported gap with "
                        "admitted IDs or exact local paths and state the chain from observed evidence to gap. Distinguish evidence "
                        "of absence from absence of evidence. Provider outages, missing APIs, and sparse retrieval are coverage "
                        "limitations, not research gaps."
                    ),
                    artifact_path="bib/RESEARCH_GAPS.md",
                    artifact_kind="report",
                    required_sections=[
                        "Supported Research Gaps",
                        "Unresolved Disputes",
                        "Missing Evidence",
                        "Speculative Opportunities",
                        "Handoff to Idea",
                        "Handoff to Plan",
                    ],
                    skill_paths=_skills(
                        "research-lit",
                        "novelty-check",
                        "kill-argument",
                        "research-review",
                    ),
                ),
            ],
        ),
        "/download": WorkflowDefinition(
            command="/download",
            title="Paper Download Workflow",
            description="Freeze a source set, resolve public PDF targets, and download them into the workspace.",
            stage_definitions=[
                StageDefinition(
                    name="download_plan",
                    title="Download Plan",
                    instruction=(
                        "Freeze the download SourceSet before any retrieval. Identify the source files, paper titles, DOI "
                        "targets, URL targets, and query terms that will guide public PDF lookup. Do not claim that any "
                        "paper has been downloaded yet."
                    ),
                    artifact_path="bib/DOWNLOAD_PLAN.md",
                    artifact_kind="plan",
                    required_sections=[
                        "Source Boundary",
                        "Target Papers",
                        "Query Terms",
                        "Download Policy",
                        "Risk Controls",
                    ],
                    skill_paths=_skills(
                        "research-lit",
                        "prior-art-search",
                        "openalex",
                        "semantic-scholar",
                        "arxiv",
                        "deepxiv",
                        "comm-lit-review",
                    ),
                ),
                StageDefinition(
                    name="download_search",
                    title="Download Search",
                    instruction=(
                        "Use the frozen source set to locate public PDFs and candidate URLs. Prefer exact matches on title, "
                        "DOI, or source URLs. Record what was found, what is still missing, and whether a public PDF seems "
                        "available."
                    ),
                    artifact_path="bib/DOWNLOAD_SEARCH.md",
                    artifact_kind="report",
                    required_sections=[
                        "Query Plan",
                        "Source File Summary",
                        "Candidate Public PDFs",
                        "Unresolved Items",
                    ],
                    skill_paths=_skills(
                        "research-lit",
                        "prior-art-search",
                        "openalex",
                        "semantic-scholar",
                        "arxiv",
                        "deepxiv",
                        "citation-audit",
                    ),
                ),
                StageDefinition(
                    name="download_fetch",
                    title="Download Fetch",
                    instruction=(
                        "Download the resolved public PDFs into the workspace and summarize the outcome. Only report files "
                        "that were actually retrieved, and keep failed or unavailable items explicit."
                    ),
                    artifact_path="bib/DOWNLOAD_REPORT.md",
                    artifact_kind="report",
                    required_sections=[
                        "Downloaded PDFs",
                        "Failed Downloads",
                        "Invalid PDFs",
                        "No Public PDF",
                        "Manifest Summary",
                    ],
                    skill_paths=_skills(
                        "research-lit",
                        "prior-art-search",
                        "openalex",
                        "semantic-scholar",
                        "arxiv",
                        "deepxiv",
                        "citation-audit",
                    ),
                ),
            ],
        ),
        "/rebuttal": WorkflowDefinition(
            command="/rebuttal",
            title="Review Response and Rebuttal Workflow",
            description="Analyze a completed paper against reviewer comments, draft traceable responses, and plan the revision.",
            stage_definitions=[
                StageDefinition(
                    name="rebuttal_intake",
                    title="Rebuttal Intake",
                    instruction="Freeze and validate the completed-paper and reviewer-comment SourceSet. This stage is generated deterministically by the platform and must not infer missing inputs.",
                    artifact_path="rebuttal/REBUTTAL_INPUTS.md",
                    artifact_kind="manifest",
                    required_sections=[
                        "Validation",
                        "Completed Paper",
                        "Reviewer Comments",
                        "Required Analysis",
                    ],
                    skill_paths=_skills(
                        "research-review",
                        "rebuttal",
                        "integrity-forensics",
                        "paper-claim-audit",
                        "citation-audit",
                    ),
                ),
                StageDefinition(
                    name="review_to_paper_map",
                    title="Review to Paper Map",
                    instruction=(
                        "Parse every reviewer comment without merging distinct requests. Map each comment to the exact paper "
                        "section, current claim or wording, figure/table/evidence, and identified gap. Record whether the paper "
                        "already addresses the point, partially addresses it, contradicts it, or lacks evidence. Use stable "
                        "reviewer and comment IDs that later stages must preserve."
                    ),
                    artifact_path="rebuttal/REVIEW_TO_PAPER_MAP.md",
                    artifact_kind="review",
                    required_sections=[
                        "Reviewer and Comment Index",
                        "Comment to Section Map",
                        "Claim and Evidence Map",
                        "Figure and Table Map",
                        "Unaddressed Evidence Gaps",
                        "Traceability Checks",
                    ],
                    skill_paths=_skills(
                        "research-review",
                        "rebuttal",
                        "integrity-forensics",
                        "paper-claim-audit",
                        "citation-audit",
                        "experiment-audit",
                    ),
                ),
                StageDefinition(
                    name="response_strategy",
                    title="Response Strategy",
                    instruction=(
                        "Choose a response strategy for every mapped comment: accept, partially accept, clarify, respectfully "
                        "disagree with evidence, or add experiments/analysis. Recommend one default path and explain the evidence "
                        "needed. Choose the evidence-supported response automatically. Use a blocking Decision Required only when "
                        "mutually exclusive strategies change claims, experiment commitments, cost, or risk and cannot be resolved "
                        "from the paper and reviews."
                    ),
                    artifact_path="rebuttal/RESPONSE_STRATEGY.md",
                    artifact_kind="review",
                    required_sections=[
                        "Strategy by Comment",
                        "Accepted and Partially Accepted Points",
                        "Clarifications and Evidence-Based Disagreements",
                        "New Experiments or Analysis",
                        "Risks and Dependencies",
                        "Decision Required",
                    ],
                    skill_paths=_skills(
                        "rebuttal",
                        "kill-argument",
                        "experiment-audit",
                        "paper-claim-audit",
                        "citation-audit",
                    ),
                    hitl=True,
                    checkpoint_title="Rebuttal Strategy Decision",
                ),
                StageDefinition(
                    name="rebuttal_draft",
                    title="Rebuttal Draft",
                    instruction=(
                        "Write a reviewer-by-reviewer, comment-by-comment rebuttal using the stable IDs in the map. Quote or "
                        "faithfully restate each comment, answer it directly, cite the relevant current paper location and "
                        "evidence, and state the exact manuscript change or additional experiment. Never claim a revision or "
                        "result that does not exist; label planned changes as commitments."
                    ),
                    artifact_path="rebuttal/REBUTTAL_DRAFT.md",
                    artifact_kind="review",
                    required_sections=[
                        "Opening Summary",
                        "Reviewer-by-Reviewer Responses",
                        "Comment-by-Comment Responses",
                        "Paper Locations and Evidence",
                        "Committed Manuscript Changes",
                        "Committed Experiments or Analysis",
                        "Remaining Limitations",
                    ],
                    skill_paths=_skills(
                        "rebuttal",
                        "kill-argument",
                        "claims-drafting",
                        "research-review",
                        "paper-claim-audit",
                    ),
                ),
                StageDefinition(
                    name="revision_plan",
                    title="Revision Plan",
                    instruction=(
                        "Turn the review-to-paper map, selected response strategies, and rebuttal draft into an executable "
                        "revision plan. Identify the exact manuscript file and section, text/claim change, experiment or "
                        "analysis, figure/table update, owner, dependency, priority, and verification needed for every comment."
                    ),
                    artifact_path="rebuttal/REVISION_PLAN.md",
                    artifact_kind="plan",
                    required_sections=[
                        "Comment Coverage Matrix",
                        "Manuscript Section Changes",
                        "Experiment and Analysis Tasks",
                        "Figure and Table Updates",
                        "Owners",
                        "Dependencies",
                        "Priority",
                        "Verification Checklist",
                    ],
                    skill_paths=_skills(
                        "auto-review-loop",
                        "resubmit-pipeline",
                        "ablation-planner",
                        "experiment-audit",
                        "citation-audit",
                        "paper-claim-audit",
                        "result-to-claim",
                    ),
                ),
                StageDefinition(
                    name="revised_manuscript",
                    title="Revised Manuscript",
                    instruction=(
                        "Apply the response strategy and revision plan to the frozen completed paper. Return a complete revised "
                        "manuscript while preserving supported facts, numbers, citations, equations, figures, and tables. "
                        "Implement textual clarifications now. Do not fabricate promised experiments or results; mark unavailable "
                        "changes as [AUTHOR INPUT NEEDED: Rn.Cn ...]."
                    ),
                    artifact_path="paper/PAPER_REVISED_AFTER_REVIEW.md",
                    artifact_kind="report",
                    required_sections=[
                        "Title",
                        "Abstract",
                        "Introduction",
                        "Method",
                        "Experiments or Results",
                        "Limitations",
                        "Conclusion",
                        "References",
                        "Unresolved Author Inputs",
                    ],
                    skill_paths=_skills(
                        "rebuttal",
                        "paper-writing",
                        "paper-write",
                        "paper-claim-audit",
                        "citation-audit",
                    ),
                ),
                StageDefinition(
                    name="revision_ledger",
                    title="Revision Ledger",
                    instruction=(
                        "Compare the frozen paper, review-to-paper map, response strategy, rebuttal draft, revision plan, and "
                        "revised manuscript. Produce one row per stable comment ID. Every row must include Status exactly as "
                        "implemented, planned, or unresolved; paper location; original issue; response promise; actual manuscript "
                        "change; evidence; and author verification needed. Do not claim implemented when the revised manuscript "
                        "does not contain the change."
                    ),
                    artifact_path="rebuttal/REVISION_LEDGER.md",
                    artifact_kind="review",
                    required_sections=[
                        "Coverage Summary",
                        "Comment Revision Ledger",
                        "Implemented Changes",
                        "Planned Changes",
                        "Unresolved Changes",
                        "Author Verification",
                    ],
                    skill_paths=_skills(
                        "rebuttal",
                        "paper-claim-audit",
                        "citation-audit",
                        "integrity-forensics",
                        "experiment-audit",
                    ),
                ),
            ],
        ),
        "/present": WorkflowDefinition(
            command="/present",
            title="Presentation Workflow",
            description="Turn workspace evidence into an approved, image-rendered research deck.",
            stage_definitions=[
                StageDefinition(
                    name="slides_outline",
                    title="Slides Outline",
                    instruction=(
                        "Read only the frozen presentation SourceSet supplied in the context. Detect whether this is a stage report "
                        "or a paper talk, extract the problem, method, progress, evidence, limitations, and next steps, "
                        "then prepare a slide-by-slide story and page-level rendering plan. Every slide must specify Page Type, "
                        "Render Mode, Layout Hint, one claim, and source files. Use image2_full for cover, section, explanation, "
                        "synthesis, and conclusion pages; Image-2 will create the complete page and nothing may be overlaid later. "
                        "Use evidence only for a separate original-figure or original-table page and provide exactly one primary "
                        "asset_id whenever possible. Never assign a paper figure or table to the cover. Never mix Image-2 artwork "
                        "and an original evidence asset on the same page. Aim for roughly 60-75% image2_full pages and 25-40% "
                        "evidence pages, adapting to the material. Resolve narrative and design choices automatically. Use a "
                        "blocking Decision Required only when the alternatives materially change claims, audience, disclosure, "
                        "cost, or external commitments; layout and style preferences never block deck generation."
                    ),
                    artifact_path="presentation/SLIDES_OUTLINE.md",
                    artifact_kind="slides",
                    required_sections=[
                        "Talk Arc",
                        "Slide List",
                        "Key Evidence per Slide",
                        "Open Design Questions",
                        "Decision Required",
                    ],
                    skill_paths=_skills(
                        "paper-slides",
                        "paper-talk",
                        "paper-poster",
                        "paper-poster-html",
                        "slides-polish",
                    ),
                    hitl=True,
                    checkpoint_title="Presentation Outline Approval",
                ),
                StageDefinition(
                    name="slide_content",
                    title="Slide Content",
                    instruction=(
                        "Expand the approved outline into exact page-level content. Use one H2 heading per slide in "
                        "the form 'Slide N: Title'. Under each slide include Page Type, Render Mode, Layout Hint, Main Message, "
                        "On-Slide Text, Visual, Source Files, and Asset IDs. Render Mode must be exactly image2_full or evidence. "
                        "For image2_full, Asset IDs must be None because the generated image is the complete final slide. "
                        "For evidence, provide a valid original asset_id and describe a restrained full-page evidence layout; "
                        "do not request Image-2. Keep text concise, preserve exact numbers, and never invent evidence."
                    ),
                    artifact_path="presentation/SLIDE_CONTENT.md",
                    artifact_kind="slides",
                    required_sections=[
                        "Slide N: Title",
                        "Page Type",
                        "Render Mode",
                        "Layout Hint",
                        "Main Message",
                        "On-Slide Text",
                        "Visual",
                        "Source Files",
                        "Asset IDs",
                    ],
                    skill_paths=_skills(
                        "paper-slides",
                        "paper-talk",
                        "slides-polish",
                        "result-to-claim",
                        "paper-claim-audit",
                    ),
                ),
                StageDefinition(
                    name="speaker_notes",
                    title="Speaker Notes",
                    instruction=(
                        "Write presenter-ready speaker notes that exactly follow SLIDE_CONTENT.md. Under Per-Slide "
                        "Notes, create one 'Slide N: Title' section per main slide and provide a natural spoken script "
                        "that can be delivered directly, not just keywords or internal production instructions. Add a "
                        "suggested duration and a concise transition for every slide. Keep claims and numbers grounded "
                        "in the selected sources; do not invent author metadata or results."
                    ),
                    artifact_path="presentation/SPEAKER_NOTES.md",
                    artifact_kind="slides",
                    required_sections=[
                        "Per-Slide Notes",
                        "Timing",
                        "Transitions",
                        "Backup Slides",
                    ],
                    skill_paths=_skills(
                        "paper-talk",
                        "paper-slides",
                        "interview-cheatsheet",
                        "slides-polish",
                    ),
                ),
                StageDefinition(
                    name="qa_brief",
                    title="Q and A Brief",
                    instruction="Prepare anticipated questions, concise answers, and follow-up evidence pointers.",
                    artifact_path="presentation/QA_BRIEF.md",
                    artifact_kind="note",
                    required_sections=[
                        "Likely Questions",
                        "Recommended Answers",
                        "Evidence Pointers",
                    ],
                    skill_paths=_skills(
                        "interview-cheatsheet",
                        "kill-argument",
                        "research-review",
                        "paper-claim-audit",
                        "result-to-claim",
                    ),
                ),
            ],
        ),
        "/wiki": WorkflowDefinition(
            command="/wiki",
            title="Research Wiki Workflow",
            description="Consolidate reusable memory and workspace notes.",
            stage_definitions=[
                StageDefinition(
                    name="knowledge_digest",
                    title="Knowledge Digest",
                    instruction="Summarize the reusable knowledge from the current request into a concise memory note, including claims, evidence, failures, and next-use cues.",
                    artifact_path="wiki/KNOWLEDGE_DIGEST.md",
                    artifact_kind="wiki",
                    required_sections=[
                        "Reusable Knowledge",
                        "Claims and Evidence",
                        "Failures to Remember",
                        "Reuse Cues",
                    ],
                    skill_paths=_skills(
                        "platform-research-wiki",
                    ),
                ),
                StageDefinition(
                    name="memory_update",
                    title="Memory Update",
                    instruction="Write a concise update describing how this material should be stored in the local research wiki and how it should be reused in future tasks.",
                    artifact_path="wiki/MEMORY_UPDATE.md",
                    artifact_kind="wiki",
                    required_sections=[
                        "Suggested Nodes",
                        "Suggested Links",
                        "Future Retrieval Prompts",
                    ],
                    skill_paths=_skills(
                        "platform-research-wiki",
                    ),
                ),
            ],
        ),
    }
