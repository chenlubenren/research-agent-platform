"""Interactive scope-clarification stage for /review (round-0).

Pass 1 asks topic-adaptive clarifying questions and emits a code-normalized default
``## Clarified Scope`` plus a compliant blocking decision block, so the workflow
pauses. Pass 2 merges the user's free-text answers into the same ``## Clarified Scope``
structure and drops the blocking block so the workflow continues. Both the approve and
the correct paths therefore leave downstream stages a single, uniform scope heading.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

from .review_screening import parse_json_block

GenerateText = Callable[..., Awaitable[str]]

# The seven adaptive dimensions the LLM should merge/rewrite per topic.
CLARIFY_DIMENSIONS = (
    "研究对象与需要包含/排除的相近概念（概念边界）",
    "希望这些文献回答的核心问题",
    "时间范围，是否重点关注近期研究",
    "重点方法、技术路线、应用场景或对比对象（务必落到具体场景）",
    "语言、数据库、出版类型与文献质量要求",
    "必须纳入与应明确排除的情况",
    "（可选）更强调全面覆盖还是精选核心",
)

_FALLBACK_QUESTIONS = [
    "本次检索具体研究什么对象，哪些相近概念需要包含或排除？",
    "你希望通过这些文献回答哪些核心问题？",
    "文献需要覆盖什么时间范围，是否重点关注近期研究？",
    "需要重点关注哪些方法、技术路线、应用场景或对比对象？",
    "对语言、数据库、出版类型和文献质量有什么要求？",
    "哪些研究必须纳入，哪些情况应明确排除？",
]


def default_scope_card(topic: str) -> dict[str, str]:
    return {
        "研究主题": topic,
        "核心研究对象": topic,
        "核心研究问题": f"围绕「{topic}」的主要方法、进展与开放问题",
        "概念边界": "以主题核心概念为准，排除仅表面相关的邻近主题",
        "重点方法/技术路线": "覆盖该主题下的主流方法与代表性技术路线",
        "应用场景/研究情境": "该主题的典型研究场景（若不明确将在第一轮检索后再聚焦）",
        "对比对象": "同主题下的代表性方法之间的比较",
        "时间范围": "近 5 年为主，保留奠基性早期工作",
        "语言范围": "中英文",
        "文献类型": "正式会议/期刊为主，接受 arXiv 等预印本但优先正式发表",
        "重点数据库": "按领域自动路由（OpenAlex/Crossref/Semantic Scholar 等）",
        "纳入条件": "直接研究该主题、含方法描述或实验评估的工作",
        "排除条件": "仅泛泛提及、与核心概念无关或无方法/证据的工作",
        "检索策略": "两阶段混合（第一轮高召回，第二轮严格筛选）",
    }


def clarified_scope_markdown(scope_card: dict[str, str]) -> str:
    lines = ["## Clarified Scope", ""]
    lines.extend(f"- {key}: {value}" for key, value in scope_card.items())
    return "\n".join(lines) + "\n"


async def build_clarify_questions(
    topic: str, *, generate_text: GenerateText, model: str | None = None
) -> list[str]:
    system = (
        "You help a researcher converge a vague literature-review topic into a searchable scope. "
        "Using these dimensions as an internal checklist, write 3-6 SPECIFIC, easy-to-answer clarifying "
        "questions tailored to the topic (merge/rewrite dimensions; do not ask them verbatim; ensure the "
        "scenario/context is pinned down). Return ONLY a JSON array of question strings.\n"
        + "\n".join(f"- {dim}" for dim in CLARIFY_DIMENSIONS)
    )
    user = f"Topic: {topic}\n\nReturn the JSON array of clarifying questions now."
    try:
        raw = await generate_text(system_prompt=system, user_prompt=user, model=model, temperature=0.2)
    except Exception:
        return list(_FALLBACK_QUESTIONS)
    parsed = parse_json_block(raw)
    if isinstance(parsed, list):
        questions = [str(item).strip() for item in parsed if str(item).strip()]
        if questions:
            return questions[:6]
    return list(_FALLBACK_QUESTIONS)


def clarify_pass1_markdown(topic: str, questions: list[str], scope_card: dict[str, str]) -> str:
    questions = questions or list(_FALLBACK_QUESTIONS)
    lines = [
        "# Scope Clarification",
        "",
        f"- Topic: {topic}",
        "",
        "## Decision Required",
        "",
        "- Blocking: Yes",
        "- 决策原因: 需先确认本次检索的范围（研究对象/核心问题/概念边界/重点方法或场景/时间范围/纳入排除），"
        "以免检索偏离你的真实意图。",
        "- Recommended Default: 采用下方《默认检索范围卡》（近 5 年、中英文、两阶段混合检索）。",
        "- Option A: 采用默认检索范围卡，直接开始检索。",
        "- Option B: 在回复中修正下列澄清问题的任意项（时间/语言/方法/场景/子问题/纳入排除）。",
        "",
        "待澄清问题:",
    ]
    lines.extend(f"{index}. {question}" for index, question in enumerate(questions, start=1))
    lines.append("")
    lines.append(clarified_scope_markdown(scope_card))
    return "\n".join(lines) + "\n"


def _strip_blocking_sections(text: str) -> str:
    return re.sub(
        r"(?:^|\n)#{1,6}\s*(?:Decision Required|需要用户决策|待用户选择)\s*\n.*?(?=\n#{1,6}\s|\Z)",
        "\n",
        text or "",
        flags=re.I | re.S,
    )


def _prior_scope_excerpt(prior_text: str) -> str:
    match = re.search(r"(?:^|\n)##\s*Clarified Scope\s*\n(.*?)(?=\n#{1,6}\s|\Z)", prior_text or "", re.I | re.S)
    return match.group(1).strip() if match else ""


async def build_clarify_pass2(
    topic: str,
    *,
    feedback: str,
    prior_text: str,
    generate_text: GenerateText,
    model: str | None = None,
) -> str:
    """Merge the user's answers into a finalized ``## Clarified Scope`` (no blocking block)."""

    prior_scope = _prior_scope_excerpt(prior_text) or clarified_scope_markdown(default_scope_card(topic))
    header = f"# Scope Clarification\n\n- Topic: {topic}\n\n"
    system = (
        "You finalize a literature-review search scope. Merge the user's answers into the prior default "
        "scope and output a single markdown section titled exactly '## Clarified Scope' as a bullet list "
        "covering: 研究主题, 核心研究对象, 核心研究问题, 概念边界, 重点方法/技术路线, 应用场景/研究情境, "
        "对比对象, 时间范围, 语言范围, 文献类型, 重点数据库, 纳入条件, 排除条件, 检索策略. "
        "Be concrete and unambiguous so a reader can tell which papers should be found and which excluded. "
        "Return ONLY that markdown section."
    )
    user = (
        f"Topic: {topic}\n\nPrior default scope:\n{prior_scope}\n\n"
        f"User answers / corrections:\n{feedback.strip()}\n\nReturn the '## Clarified Scope' section now."
    )
    try:
        raw = await generate_text(system_prompt=system, user_prompt=user, model=model, temperature=0.2)
    except Exception:
        raw = ""
    raw = _strip_blocking_sections(raw).strip()
    match = re.search(r"(##\s*Clarified Scope\s*\n.*)$", raw, re.I | re.S)
    if match and match.group(1).strip():
        body = re.sub(r"^##\s*Clarified Scope\s*\n", "## Clarified Scope\n", match.group(1).strip(), flags=re.I)
        return header + body.strip() + "\n"
    # Deterministic fallback: keep the prior default and record the user's corrections verbatim.
    fallback = [
        header.rstrip(),
        "",
        "## Clarified Scope",
        "",
        prior_scope.strip(),
        "",
        "用户补充/修正:",
        feedback.strip() or "（无）",
    ]
    return "\n".join(fallback) + "\n"
