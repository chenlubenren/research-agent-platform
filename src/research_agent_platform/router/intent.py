from __future__ import annotations

import re
from dataclasses import dataclass

from ..upstream import generate_text


APPROVE_WORDS = {
    "approve",
    "approved",
    "go",
    "continue",
    "ok",
    "yes",
    "y",
    "同意",
    "批准",
    "继续",
}

STOP_WORDS = {"stop", "halt", "cancel", "停止", "取消"}

ALIASES = {
    "/research-pipeline": "/plan",
    "/pipeline": "/plan",
    "/idea-discovery": "/idea",
    "/experiment-bridge": "/code",
    "/paper-writing": "/write",
    "/literature-review": "/review",
    "/lit-review": "/review",
    "/paper-download": "/download",
    "/auto-review-loop": "/rebuttal",
    "/review-response": "/rebuttal",
    "/paper-slides": "/present",
    "/research-wiki": "/wiki",
}

COMMAND_DESCRIPTIONS = {
    "/review": "search, organize, and synthesize research literature and evidence",
    "/download": "search for papers, resolve public PDFs, and download them into the workspace",
    "/idea": "generate, challenge, verify, and select research ideas",
    "/plan": "turn a selected idea or objective into experiments and an execution plan",
    "/code": "turn the chosen plan into implementation and experiment execution materials",
    "/fig": "generate one evidence-grounded research figure with editable Python, Academic SVG, or Draw.io source",
    "/write": "create paper outlines, narrative reports, and draft sections",
    "/rebuttal": "analyze peer-review comments and produce rebuttal and revision materials",
    "/present": "prepare slides, poster, talk track, and Q&A materials",
    "/wiki": "update persistent research memory and reusable knowledge notes",
}

HEURISTICS = {
    "/review": [
        "literature review",
        "related work",
        "survey paper",
        "literature search",
        "文献综述",
        "文献调研",
        "找文献",
        "相关工作",
        "检索文献",
    ],
    "/download": [
        "download paper",
        "paper download",
        "pdf download",
        "download pdf",
        "download literature",
        "download papers",
        "下载论文",
        "下载pdf",
    ],
    "/idea": ["idea", "novelty", "research topic", "选题", "想法", "创新点", "研究方向"],
    "/plan": [
        "experiment plan",
        "research plan",
        "roadmap",
        "pipeline",
        "milestone",
        "实验方案",
        "研究计划",
        "规划",
        "路线图",
    ],
    "/code": ["experiment", "implement", "code", "reproduce", "run", "实验", "实现", "复现"],
    "/fig": ["figure", "plot", "diagram", "chart", "图", "图表", "流程图"],
    "/write": ["paper", "draft", "write", "manuscript", "论文", "写作", "草稿"],
    "/rebuttal": [
        "rebuttal",
        "reviewer comments",
        "peer review",
        "revision letter",
        "审稿意见",
        "审稿回复",
        "返修",
        "回复审稿",
    ],
    "/present": [
        "slides",
        "poster",
        "talk",
        "presentation",
        "汇报",
        "答辩",
        "ppt",
        "论文汇报",
        "论文答辩",
        "研究汇报",
        "阶段汇报",
    ],
    "/wiki": ["wiki", "memory", "knowledge base", "知识库", "记忆", "归档"],
}

CHAT_CUES = {
    "hello",
    "hi",
    "hey",
    "你好",
    "您好",
    "在吗",
    "谢谢",
    "thanks",
    "how are you",
    "who are you",
}

# Questions are deliberately handled before keyword heuristics.  Terms such
# as “论文”, “图” and “汇报” occur frequently in explanatory questions, but
# those questions must not silently start a file-producing workflow.
QUESTION_PREFIXES = (
    "请问",
    "什么",
    "为何",
    "为什么",
    "怎么",
    "如何",
    "是否",
    "能否",
    "能不能",
    "可不可以",
    "有没有",
    "哪些",
    "哪个",
    "多少",
    "什么时候",
    "在哪里",
    "在哪",
    "介绍",
    "解释",
    "说明",
    "what",
    "why",
    "how",
    "can",
    "could",
    "is",
    "are",
    "do",
    "does",
    "where",
    "when",
    "which",
)
QUESTION_PHRASES = (
    "是什么",
    "什么意思",
    "有什么区别",
    "包括哪些",
    "包括什么",
    "有哪些",
    "能做什么",
    "如何理解",
    "怎么用",
    "为什么",
    "可以吗",
    "能吗",
    "行吗",
    "好吗",
)
INFORMATIONAL_PHRASES = (
    "想知道",
    "想了解",
    "知道",
    "了解",
    "咨询",
    "请教",
    "解释一下",
    "介绍一下",
    "说明一下",
    "说明",
    "告诉我",
    "帮我理解",
    "如何",
    "区别",
    "概念",
    "定义",
    "建议",
    "推荐",
    "步骤",
    "方法",
    "教程",
    "流程",
    "用法",
    "原理",
    "结构",
    "组成",
    "规范",
    "要求",
    "注意事项",
)
QUESTION_TOKENS = (
    "什么",
    "怎么",
    "如何",
    "为什么",
    "为何",
    "是否",
    "能否",
    "能不能",
    "可不可以",
    "有没有",
    "哪些",
    "哪个",
    "多少",
    "什么时候",
    "哪里",
    "哪儿",
    "what",
    "why",
    "how",
    "whether",
    "which",
    "where",
    "when",
)
EXECUTION_PHRASES = (
    "请做",
    "请生成",
    "请制作",
    "请创建",
    "请写",
    "请撰写",
    "请画",
    "请绘制",
    "请整理",
    "请制定",
    "请规划",
    "请设计",
    "请实现",
    "请运行",
    "请执行",
    "请下载",
    "请检索",
    "请搜索",
    "请分析",
    "请回复",
    "请更新",
    "请归档",
    "请导出",
    "帮我生成",
    "帮我做",
    "帮我弄",
    "帮我搞",
    "帮我制作",
    "帮我创建",
    "帮我写",
    "帮我撰写",
    "帮我画",
    "帮我绘制",
    "帮我整理",
    "帮我制定",
    "帮我规划",
    "帮我设计",
    "帮我实现",
    "帮我运行",
    "帮我下载",
    "帮我检索",
    "帮我搜索",
    "帮我分析",
    "帮我回复",
    "我要生成",
    "我要做",
    "我要弄",
    "我要搞",
    "我要制作",
    "我要创建",
    "我想生成",
    "我想制作",
    "我想做",
    "我想弄",
    "我想搞",
    "我需要生成",
    "我需要做",
    "我需要弄",
    "我需要搞",
    "给我做",
    "给我弄",
    "给我搞",
    "给我一份",
    "给我一个",
    "给我一张",
    "来一份",
    "来一个",
    "来一张",
    "我需要制作",
    "generate",
    "create",
    "produce",
    "make",
    "write",
    "draft",
    "draw",
    "build",
    "download",
    "search",
    "analyze",
    "prepare",
)
DIRECT_EXECUTION_PREFIXES = (
    "能不能帮我",
    "能否帮我",
    "可不可以帮我",
    "可以帮我",
    "方便帮我",
)
EXECUTION_TARGET_CUES = (
    "ppt", "汇报", "答辩", "图", "图表", "图片", "论文", "文献", "综述",
    "审稿", "回复", "方案", "计划", "选题", "实验", "代码", "知识库", "记忆",
    "slides", "presentation", "figure", "paper", "review",
)
EXECUTION_VERBS = (
    "做",
    "弄",
    "搞",
    "生成",
    "制作",
    "创建",
    "撰写",
    "起草",
    "写一份",
    "写一篇",
    "画一张",
    "绘制",
    "整理",
    "提出",
    "制定",
    "规划",
    "设计",
    "实现",
    "执行",
    "运行",
    "复现",
    "下载",
    "检索",
    "搜索",
    "查找",
    "找文献",
    "分析",
    "回复",
    "处理",
    "更新",
    "归档",
    "导出",
    "generate",
    "create",
    "produce",
    "make",
    "write",
    "draft",
    "draw",
    "build",
    "download",
    "search",
    "analyze",
    "prepare",
    "implement",
    "run",
)


@dataclass
class RouteDecision:
    command: str
    source: str
    reason: str
    workflow_mode: str = ""
    skill_bundle: tuple[str, ...] = ()


def _publication_mode(command: str, message: str) -> tuple[str, tuple[str, ...]]:
    lowered = re.sub(r"^\s*/[\w-]+\b", "", message, flags=re.I).lower()
    if command == "/review":
        if any(term in lowered for term in ("写综述", "综述文章", "literature review", "narrative review", "survey paper", "review article", "synthesize", "synthesis", "compare authorities")) or ("综述" in lowered and any(term in lowered for term in ("写", "生成", "起草", "draft"))):
            return "literature_review_writing", ("paper-init", "paper-style-learn", "paper-literature-review", "paper-draft", "paper-final-check")
        return "literature_evidence", ("paper-init", "paper-literature-review")
    if command == "/write":
        if any(term in lowered for term in ("改论文", "修改稿件", "润色", "improve manuscript", "revise manuscript", "polish")):
            return "manuscript_improvement", ("paper-init", "paper-revise", "paper-final-check")
        return "research_materials_writing", ("paper-init", "paper-draft", "paper-review", "paper-final-check")
    if command == "/rebuttal":
        if any(term in lowered for term in ("审稿意见", "reviewer comment", "rebuttal", "返修", "回复审稿")):
            return "rebuttal_revision", ("paper-review", "paper-revise-from-review", "paper-final-check")
        return "manuscript_diagnosis", ("paper-init", "paper-review")
    return "", ()


def is_approval_message(message: str) -> bool:
    return message.strip().lower() in APPROVE_WORDS


def is_stop_message(message: str) -> bool:
    return message.strip().lower() in STOP_WORDS


def _looks_like_chat(message: str) -> bool:
    stripped = message.strip().lower()
    if not stripped:
        return True
    if stripped in CHAT_CUES:
        return True
    return len(stripped) <= 24 and any(cue in stripped for cue in CHAT_CUES)


async def route_message(message: str, context: str = "") -> RouteDecision | None:
    stripped = message.strip()
    explicit = explicit_route(message)
    if explicit is not None:
        return explicit

    # These high-signal non-execution cases stay in text chat without a
    # provider round-trip.  The LangGraph node still owns the overall gate;
    # this guard prevents a keyword such as “PPT” from ever creating a task.
    if is_informational_question(stripped):
        return None

    if _looks_like_chat(stripped):
        return None

    if context.strip() and _looks_like_session_followup_question(stripped):
        return None

    if _looks_like_artifact_location_question(stripped):
        return None

    # A bare artifact keyword (for example, “PPT”) is ordinary chat.  Only
    # send messages with at least a spoken request cue to the semantic model.
    if not has_execution_intent(stripped) and not any(
        cue in "".join(stripped.lower().split())
        for cue in ("帮我", "我要", "我想", "我需要", "给我", "来个", "整一个", "弄个")
    ):
        return None

    # Let the language model make the primary semantic decision.  The local
    # keyword route below remains a deterministic fallback for provider
    # outages, malformed classifier output, and tests with a stub upstream.
    options = "\n".join(f"{name}: {description}" for name, description in COMMAND_DESCRIPTIONS.items())
    classifier_prompt = (
        "Classify the user's request into exactly one command from the list below, or return chat.\n"
        "Return chat for questions, explanations, capability checks, definitions, comparisons, or how-to advice. "
        "Only select a command when the user explicitly asks you to perform the workflow or create/download an artifact. "
        "Spoken requests such as 做个PPT, 帮我做一份汇报, 给我弄一张图, or 来一个论文汇报 are execution requests; "
        "phrases such as 是什么, 怎么做, 如何使用, or 介绍一下 are informational questions.\n"
        "Return only one token: a command token or chat.\n\n"
        f"{options}\n\nUser request:\n{message}"
    )
    if context.strip():
        classifier_prompt += f"\n\nSession context:\n{context.strip()}"
    try:
        raw = await generate_text(
            system_prompt="You are a strict intent router. Output one token only.",
            user_prompt=classifier_prompt,
            temperature=0,
        )
    except Exception:
        # Routing must never make the chat endpoint unavailable.  If the
        # classifier is unavailable, continue to the deterministic fallback
        # below rather than losing a clearly executable request.
        raw = ""
    first_line = raw.strip().splitlines()[0].strip() if raw else ""
    command_token = first_line.strip("`'\".,:; ")
    # Be tolerant of a provider adding a short label around the token while
    # still requiring an allow-listed workflow command.
    command = ALIASES.get(command_token, command_token)
    if command.lower() == "chat":
        command = ""
    if command not in COMMAND_DESCRIPTIONS:
        match = re.search(r"/(?:review|download|idea|plan|code|fig|write|rebuttal|present|wiki)\b", raw or "", re.I)
        command = ALIASES.get(match.group(0).lower(), match.group(0).lower()) if match else ""
    if command and not is_informational_question(stripped) and has_execution_intent(stripped):
        workflow_mode, skill_bundle = _publication_mode(command, stripped)
        return RouteDecision(
            command=command,
            source="implicit_llm",
            reason=f"llm classifier -> {command}",
            workflow_mode=workflow_mode,
            skill_bundle=skill_bundle,
        )

    # Informational questions are always kept in text chat, even if a noisy
    # classifier guessed a workflow from a keyword such as “PPT” or “图”.
    if is_informational_question(stripped):
        return None

    # Deterministic fallback for an unavailable or non-conforming upstream.
    lowered = stripped.lower()
    best_command = ""
    best_score = (0, 0)
    for fallback_command, keywords in HEURISTICS.items():
        matches = [keyword for keyword in keywords if keyword.lower() in lowered]
        score = (len(matches), sum(len(keyword) for keyword in matches))
        if score > best_score:
            best_command = fallback_command
            best_score = score
    if best_command and best_score[0] > 0 and has_execution_intent(stripped):
        return RouteDecision(
            command=best_command,
            source="implicit_heuristic",
            reason=f"keyword fallback matches={best_score[0]} specificity={best_score[1]}",
            workflow_mode=_publication_mode(best_command, stripped)[0],
            skill_bundle=_publication_mode(best_command, stripped)[1],
        )

    return None


def is_informational_question(message: str) -> bool:
    """Return True when a message asks for information instead of execution.

    This is intentionally conservative: an explicit slash command always wins
    in ``route_message``.  For natural-language input, an unresolved question
    is kept in chat so the model can explain the workflow without creating
    files.  A polite request such as “请生成 PPT，可以吗？” is also treated as
    a question and waits for an unambiguous command in a later turn.
    """

    stripped = message.strip()
    if not stripped or explicit_route(stripped) is not None:
        return False
    normalized = "".join(stripped.lower().split())
    if not normalized:
        return False
    if _looks_like_direct_execution_request(normalized):
        return False
    if any(
        suffix in normalized
        for suffix in ("的步骤", "的方法", "的教程", "的流程", "的用法")
    ):
        return True
    if normalized.endswith(("?", "？")):
        return True
    if normalized.endswith(("吗", "么", "呢")):
        return True
    if normalized.startswith(QUESTION_PREFIXES):
        return True
    if any(phrase in normalized for phrase in EXECUTION_PHRASES):
        # An explicit execution phrase wins over an embedded question token
        # in the requested content (for example, “请生成 PPT，说明什么是 AI”).
        # A trailing question mark still remains ambiguous and is kept in chat.
        return False
    if any(token in normalized for token in QUESTION_TOKENS):
        return True
    if any(phrase in normalized for phrase in QUESTION_PHRASES):
        return True
    return any(phrase in normalized for phrase in INFORMATIONAL_PHRASES)


def has_execution_intent(message: str) -> bool:
    """Return whether natural-language input contains an execution request.

    Keyword matches such as “PPT”, “图” or “论文” are not sufficient to start
    a workflow.  A natural-language route needs at least one explicit action
    phrase/verb; slash commands remain an explicit user override.
    """

    normalized = "".join(message.lower().split())
    if not normalized:
        return False
    if explicit_route(message) is not None:
        return True
    return any(term in normalized for term in EXECUTION_PHRASES + EXECUTION_VERBS)


def _looks_like_direct_execution_request(normalized_message: str) -> bool:
    """Recognize polite spoken requests with a concrete artifact target."""

    return any(prefix in normalized_message for prefix in DIRECT_EXECUTION_PREFIXES) and any(
        cue in normalized_message for cue in EXECUTION_TARGET_CUES
    )


def _looks_like_artifact_location_question(message: str) -> bool:
    normalized = "".join(message.lower().split())
    if not normalized:
        return False
    location_terms = (
        "where",
        "saved",
        "save",
        "output",
        "download",
        "link",
        "path",
        "file",
        "folder",
        "directory",
        "artifact",
        "在哪里",
        "在哪",
        "哪里",
        "哪儿",
        "输出到",
        "保存到",
        "放到",
        "存到",
        "路径",
        "位置",
        "下载",
        "链接",
        "文件",
        "文件夹",
        "目录",
        "产物",
    )
    artifact_terms = (
        "figure",
        "image",
        "plot",
        "chart",
        "diagram",
        "artifact",
        "output",
        "paper",
        "docx",
        "pdf",
        "ppt",
        "png",
        "svg",
        "图",
        "图片",
        "图像",
        "图表",
        "示意图",
        "流程图",
        "产物",
        "论文",
        "文档",
        "文件",
        "结果",
    )
    followup_terms = (
        "要求的",
        "刚才",
        "上个",
        "上一",
        "前面",
        "生成的",
        "输出的",
        "保存的",
        "that",
        "the",
        "last",
        "previous",
        "generated",
    )
    has_location = any(term in normalized for term in location_terms)
    has_artifact = any(term in normalized for term in artifact_terms)
    has_followup = any(term in normalized for term in followup_terms)
    return has_location and has_artifact and (has_followup or len(normalized) <= 80)


def _looks_like_session_followup_question(message: str) -> bool:
    normalized = "".join(message.lower().split())
    if not normalized:
        return False
    followup_terms = (
        "那张",
        "那篇",
        "那个",
        "这个",
        "它",
        "刚才",
        "刚刚",
        "前面",
        "前文",
        "上一个",
        "上张",
        "上一",
        "继续",
        "接着",
        "还能",
        "还在",
        "哪里",
        "哪儿",
        "呢",
        "吗",
        "吧",
        "still",
        "again",
        "previous",
        "that",
        "this",
        "it",
    )
    return any(term in normalized for term in followup_terms)


def explicit_route(message: str) -> RouteDecision | None:
    stripped = message.strip()
    first_token = stripped.split(maxsplit=1)[0] if stripped else ""
    if not first_token.startswith("/"):
        return None
    command = ALIASES.get(first_token, first_token)
    if command not in COMMAND_DESCRIPTIONS:
        return None
    workflow_mode, skill_bundle = _publication_mode(command, stripped)
    return RouteDecision(command=command, source="explicit", reason=f"explicit command {first_token}", workflow_mode=workflow_mode, skill_bundle=skill_bundle)
