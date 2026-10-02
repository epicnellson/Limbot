"""``python -m app.eval`` — run the synthetic evaluation catalogue against live providers.

Requires provider credentials (GROQ_API_KEY and/or GEMINI_API_KEY) and is happy without
Postgres or Qdrant: the capability cases those back are skipped and reported as such, so a bare
CI run exercises identity, safety and conversation behaviour while a fully provisioned host also
covers tools and retrieval.

Exit codes: 0 passed the pass-rate gate, 1 the gate failed, 2 misconfiguration.

A Markdown report is written to ``EVAL_REPORT.md`` (``--report`` to move or disable it) and
appended to $GITHUB_STEP_SUMMARY when running in CI, both carrying the pass rate, the average
latency per provider and the request distribution across tiers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence

import httpx

from app.config import Settings
from app.conversation.store import ConversationStore
from app.core.embeddings import Embedder
from app.core.qdrant import VectorStore
from app.db.pool import Database
from app.db.students import StudentRepository
from app.eval.cases import EVAL_CASES, EvalCase
from app.eval.runner import EvalRunner, render_markdown, render_table
from app.llm.pipeline import LLMPipeline
from app.rag.pipeline import RetrievalPipeline
from app.services.answer import Answer, AnswerService
from app.tools.student import build_student_registry

DEFAULT_REPORT_PATH = "EVAL_REPORT.md"


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.eval", description=__doc__)
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.8,
        help="gate; the run exits 1 when the executed pass rate is below this (default 0.8)",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=["none", "tools", "retrieval", "conversation"],
        help="only run cases from this capability; repeatable, default all",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help=f"where to write the Markdown report (default {DEFAULT_REPORT_PATH}, empty to skip)",
    )
    return parser.parse_args(argv)


def _step_summary(report_text: str) -> None:
    """Append to $GITHUB_STEP_SUMMARY when running as a GitHub Actions job step."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(report_text)


def _write_report(path: str, markdown: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown)


async def _run(settings: Settings, args: argparse.Namespace, http: httpx.AsyncClient) -> int:
    pipeline = LLMPipeline(settings)
    conversations = ConversationStore(settings)

    retrieval: RetrievalPipeline | None = None
    if settings.vector_store_enabled:
        vector_store = VectorStore(settings)
        retrieval = RetrievalPipeline(settings, Embedder(settings), vector_store)

    tools = None
    if settings.tools_enabled:
        database = await _open_database(settings)
        if database is not None:
            tools = build_student_registry(
                StudentRepository(database),
                timeout_seconds=settings.ai_tool_call_timeout_seconds,
            )

    answers = AnswerService(
        settings,
        pipeline,
        retrieval=retrieval,
        tools=tools,
        conversations=conversations,
    )

    capabilities = {"none", "conversation"}
    if tools is not None:
        capabilities.add("tools")
    if retrieval is not None:
        capabilities.add("retrieval")
    if args.group:
        capabilities &= set(args.group)

    async def answer_case(case: EvalCase) -> Answer:
        if case.prior:
            conversations.reset(case.context.wa_id)
            for question, reply in case.prior:
                conversations.record(case.context.wa_id, question, reply)
        return await answers.answer(case.question, case.context, http=http)

    runner = EvalRunner(
        answer_case,
        capabilities=tuple(
            capability
            for capability in ("none", "conversation", "tools", "retrieval")
            if capability in capabilities
        ),
    )
    report = await runner.run(EVAL_CASES)
    table = render_table(report)
    markdown = render_markdown(report)

    pass_gate = args.min_pass_rate is not None
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(table)
        for result in report.results:
            if not result.skipped and result.failures:
                # A real gate needs failures on stderr, where they cannot be lost.
                print(f"FAIL {result.case.name}: {'; '.join(result.failures)}", file=sys.stderr)
    _step_summary(markdown)
    if args.report:
        _write_report(args.report, markdown)
        print(f"report written to {args.report}", file=sys.stderr)
    if not pass_gate:
        return 0
    return 0 if report.meets(args.min_pass_rate) else 1


async def _open_database(settings: Settings) -> Database | None:
    from app.runtime import _open_database as open_database

    return await open_database(settings)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv or sys.argv[1:])
    settings = Settings()
    if not settings.ai_enabled:
        print("AI_ENABLED is off; nothing to evaluate. All cases would fail.", file=sys.stderr)
        return 2
    if not settings.configured_providers:
        print(
            "no provider is configured: set GROQ_API_KEY and/or GEMINI_API_KEY (or "
            "TIER3_ENABLED) before running the evaluation suite.",
            file=sys.stderr,
        )
        return 2

    async def entry() -> int:
        async with httpx.AsyncClient() as http:
            return await _run(settings, args, http)

    return asyncio.run(entry())


if __name__ == "__main__":
    raise SystemExit(main())
