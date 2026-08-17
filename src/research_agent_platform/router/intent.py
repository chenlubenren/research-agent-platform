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
    "/present": ["slides", "poster", "talk", "presentation", "汇报", "答辩", "ppt"],
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

    if _looks_like_chat(stripped):
        return None

    if context.strip() and _looks_like_session_followup_question(stripped):
        return None

    if _looks_like_artifact_location_question(stripped):
        return None

    lowered = stripped.lower()
    best_command = ""
    best_score = (0, 0)
    for command, keywords in HEURISTICS.items():
        matches = [keyword for keyword in keywords if keyword.lower() in lowered]
        score = (len(matches), sum(len(keyword) for keyword in matches))
        if score > best_score:
            best_command = command
            best_score = score
    if best_command and best_score[0] > 0:
        return RouteDecision(
            command=best_command,
            source="implicit_heuristic",
            reason=f"keyword heuristic matches={best_score[0]} specificity={best_score[1]}",
            workflow_mode=_publication_mode(best_command, stripped)[0],
            skill_bundle=_publication_mode(best_command, stripped)[1],
        )

    options = "\n".join(f"{name}: {description}" for name, description in COMMAND_DESCRIPTIONS.items())
    classifier_prompt = (
        "Classify the user's request into exactly one command from the list below, or return chat if this is casual conversation.\n"
        "Return only one token: a command token or chat.\n\n"
        f"{options}\n\nUser request:\n{message}"
    )
    if context.strip():
        classifier_prompt += f"\n\nSession context:\n{context.strip()}"
    raw = await generate_text(
        system_prompt="You are a strict intent router. Output one token only.",
        user_prompt=classifier_prompt,
        temperature=0,
    )
    command = raw.strip().splitlines()[0].strip()
    if command.lower() == "chat":
        return None
    command = ALIASES.get(command, command)
    if command not in COMMAND_DESCRIPTIONS:
        return None
    workflow_mode, skill_bundle = _publication_mode(command, stripped)
    return RouteDecision(command=command, source="implicit_llm", reason=f"llm classifier -> {command}", workflow_mode=workflow_mode, skill_bundle=skill_bundle)


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
