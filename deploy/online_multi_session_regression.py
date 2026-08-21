"""Run a bounded multi-session, multi-turn regression against both deployments.

The script deliberately uses ordinary conversation turns rather than long PPT
generation for the full matrix.  A small sampled set can be enabled for stream
and file-contract checks.  It writes JSONL (one record per session) and a JSON
summary, never logging API keys or response headers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx


def load_env(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def assistant_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


@dataclass
class TurnRecord:
    round: int
    prompt: str
    status: int | None
    elapsed_ms: int
    response_bytes: int
    response_object: str
    text_chars: int
    has_session_marker: bool
    has_error: bool
    error: str = ""
    response_text: str = ""
    artifacts: list[dict[str, str]] | None = None


@dataclass
class SessionRecord:
    service: str
    session_id: str
    rounds_requested: int
    turns: list[TurnRecord]
    continuity_ok: bool
    completed: bool
    error: str = ""


def _prompt_rounds(service: str, rounds: int) -> list[str]:
    if service == "presentation":
        base = [
            "你好。请只用一句话说明你能处理哪些科研汇报任务，不要生成文件。",
            "记住我的主题是可解释机器学习。请只回答：你记住的主题是什么？不要生成PPT。",
            "基于刚才的主题，给出3个制作高水平学术汇报前需要确认的问题；只列问题，不生成PPT。",
            "我强调只做规划，不生成PPT。请把第一个问题改写得更具体。",
        ]
    else:
        base = [
            "你好。请只用一句话说明你能处理哪些科研任务，不要启动研究工作流。",
            "记住我的主题是可解释机器学习。请只回答：你记住的主题是什么？不要创建任务。",
            "基于刚才的主题，给出3个研究设计澄清问题；只列问题，不执行任务。",
            "我强调只做规划，不执行任务。请把第一个问题改写得更具体。",
        ]
    return base[: max(1, min(rounds, len(base)))]


async def run_session(
    client: httpx.AsyncClient,
    *,
    service: str,
    base_url: str,
    api_key: str,
    model: str,
    session_index: int,
    rounds: int,
    semaphore: asyncio.Semaphore,
) -> SessionRecord:
    session_id = f"multi-regression-{service}-{session_index:04d}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages: list[dict[str, str]] = []
    turns: list[TurnRecord] = []
    continuity_ok = True
    async with semaphore:
        for round_number, prompt in enumerate(_prompt_rounds(service, rounds), start=1):
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": model,
                "stream": False,
                "messages": messages,
                "metadata": {"session_id": session_id, "regression": "multi-session-20260822"},
            }
            started = time.perf_counter()
            try:
                response = await client.post(
                    f"{base_url.rstrip('/')}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                try:
                    body = response.json()
                except Exception:
                    body = {}
                text = assistant_text(body) if isinstance(body, dict) else ""
                marker = body.get("x_agent_task") or body.get("research_presentation") or {}
                has_session_marker = bool(
                    (isinstance(marker, dict) and (marker.get("session_id") or marker.get("run_id")))
                    or (isinstance(body, dict) and str(body.get("id") or "").startswith("chatcmpl-"))
                )
                has_error = response.status_code >= 400 or bool(body.get("error"))
                turns.append(
                    TurnRecord(
                        round=round_number,
                        prompt=prompt,
                        status=response.status_code,
                        elapsed_ms=elapsed_ms,
                        response_bytes=len(response.content),
                        response_object=str(body.get("object") or ""),
                        text_chars=len(text),
                        has_session_marker=has_session_marker,
                        has_error=has_error,
                        error=str(body.get("error") or "")[:300],
                        response_text=text[:12000],
                        artifacts=[
                            {
                                key: str(item.get(key) or "")[:500]
                                for key in ("name", "path", "kind", "label")
                                if key in item
                            }
                            for item in ((body.get("research_presentation") or {}).get("artifacts") or [])
                            if isinstance(item, dict)
                        ][:40],
                    )
                )
                if response.status_code != 200 or not text or has_error or not has_session_marker:
                    continuity_ok = False
                messages.append({"role": "assistant", "content": text})
            except Exception as exc:  # network and timeout errors are recorded per turn
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                turns.append(
                    TurnRecord(
                        round=round_number,
                        prompt=prompt,
                        status=None,
                        elapsed_ms=elapsed_ms,
                        response_bytes=0,
                        response_object="",
                        text_chars=0,
                        has_session_marker=False,
                        has_error=True,
                        error=f"{exc.__class__.__name__}: {exc}"[:300],
                        response_text="",
                        artifacts=[],
                    )
                )
                continuity_ok = False
                messages.append({"role": "assistant", "content": ""})
    return SessionRecord(
        service=service,
        session_id=session_id,
        rounds_requested=rounds,
        turns=turns,
        continuity_ok=continuity_ok and len(turns) == rounds,
        completed=len(turns) == rounds,
    )


async def run_service(
    service: str,
    base_url: str,
    env_path: str,
    model: str,
    count: int,
    rounds: int,
    concurrency: int,
) -> list[SessionRecord]:
    env = load_env(env_path)
    timeout = httpx.Timeout(90.0, connect=10.0)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        tasks = [
            run_session(
                client,
                service=service,
                base_url=base_url,
                api_key=env.get("PUBLIC_API_KEY", ""),
                model=model,
                session_index=index,
                rounds=rounds,
                semaphore=semaphore,
            )
            for index in range(1, count + 1)
        ]
        return list(await asyncio.gather(*tasks))


def summarize(records: list[SessionRecord]) -> dict[str, Any]:
    turn_rows = [turn for record in records for turn in record.turns]
    latencies = [turn.elapsed_ms for turn in turn_rows]
    by_service: dict[str, Any] = {}
    for service in sorted({record.service for record in records}):
        subset = [record for record in records if record.service == service]
        turns = [turn for record in subset for turn in record.turns]
        values = sorted(turn.elapsed_ms for turn in turns)
        by_service[service] = {
            "sessions": len(subset),
            "sessions_completed": sum(record.completed for record in subset),
            "sessions_continuity_ok": sum(record.continuity_ok for record in subset),
            "turns": len(turns),
            "turns_http_200": sum(turn.status == 200 for turn in turns),
            "turns_with_error": sum(turn.has_error for turn in turns),
            "latency_ms": {
                "min": min(values) if values else None,
                "median": int(statistics.median(values)) if values else None,
                "p95": values[max(0, int(len(values) * 0.95) - 1)] if values else None,
                "max": max(values) if values else None,
            },
        }
    return {
        "sessions": len(records),
        "sessions_completed": sum(record.completed for record in records),
        "sessions_continuity_ok": sum(record.continuity_ok for record in records),
        "turns": len(turn_rows),
        "turns_http_200": sum(turn.status == 200 for turn in turn_rows),
        "turns_with_error": sum(turn.has_error for turn in turn_rows),
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "median": int(statistics.median(latencies)) if latencies else None,
            "p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "by_service": by_service,
    }


async def main_async(args: argparse.Namespace) -> None:
    records: list[SessionRecord] = []
    if not args.only or args.only == "platform":
        records.extend(
            await run_service(
                "platform",
                args.platform_base,
                args.platform_env,
                "research-agent-platform",
                args.sessions,
                args.rounds,
                args.concurrency,
            )
        )
    if not args.only or args.only == "presentation":
        records.extend(
            await run_service(
                "presentation",
                args.presentation_base,
                args.presentation_env,
                "research-presentation-agent",
                args.sessions,
                args.rounds,
                args.concurrency,
            )
        )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    summary_path = output.with_suffix(".summary.json")
    summary = summarize(records)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    review_root = output.parent / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    good_kept: dict[str, int] = {}
    bad_count = 0
    for record in records:
        is_bad = not record.completed or not record.continuity_ok or any(turn.has_error for turn in record.turns)
        service_root = review_root / record.service
        if is_bad:
            bad_count += 1
        elif good_kept.get(record.service, 0) >= 3:
            continue
        service_root.mkdir(parents=True, exist_ok=True)
        review_path = service_root / f"{record.session_id}.json"
        review_path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
        if not is_bad:
            good_kept[record.service] = good_kept.get(record.service, 0) + 1
    summary["review_artifacts"] = {
        "directory": str(review_root),
        "good_sessions_kept_per_service": good_kept,
        "bad_sessions_kept": bad_count,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--only", choices=("platform", "presentation"), default="")
    parser.add_argument("--platform-base", default="http://123.57.91.184")
    parser.add_argument("--presentation-base", default="http://123.57.91.184/presentation")
    parser.add_argument("--platform-env", default="research-agent-platform/.env")
    parser.add_argument("--presentation-env", default="research-presentation-agent/.env")
    parser.add_argument("--output", default="output/online-multi-session-regression.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
