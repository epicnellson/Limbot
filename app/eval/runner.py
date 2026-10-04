"""Deterministic grading and report aggregation for the evaluation catalogue."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app import metrics
from app.eval.cases import EVAL_CASES, Capability, EvalCase
from app.services.answer import Answer

AnswerCase = Callable[[EvalCase], Awaitable[Answer]]

ALL_CAPABILITIES: tuple[Capability, ...] = ("none", "tools", "retrieval", "conversation")


def grade(case: EvalCase, answer: Answer) -> tuple[str, ...]:
    """Return the reasons this reply failed the rubric, or () when it passed.

    The checks are deliberately boolean and phrase-based so the same suite runs against a real
    provider in CI and a stub in unit tests without either side tuning to the other.
    """
    if not answer.ok:
        return (f"no usable answer ({answer.error or 'empty text'})",)

    failures: list[str] = []
    text = (answer.text or "").lower()
    if case.must_include and not any(term in text for term in case.must_include):
        failures.append(f"expected one of {list(case.must_include)} in the reply")
    if case.must_exclude and any(term in text for term in case.must_exclude):
        failures.append(f"reply contained a forbidden phrase ({list(case.must_exclude)})")
    if case.max_words is not None:
        words = len(answer.text.split())
        if words > case.max_words:
            failures.append(f"{words} words exceeds the {case.max_words} word limit")
    if case.expect_tools and not set(answer.used_tools) & set(case.expect_tools):
        failures.append(
            f"did not call any of {list(case.expect_tools)} (used {list(answer.used_tools)})"
        )
    if case.expect_no_tools and answer.used_tools:
        failures.append(f"called tools when it should not: {list(answer.used_tools)}")
    return tuple(failures)


@dataclass(frozen=True, slots=True)
class EvalResult:
    case: EvalCase
    answer: Answer
    skipped: bool = False
    failures: tuple[str, ...] = ()
    latency_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.skipped or not self.failures


@dataclass(frozen=True, slots=True)
class ProviderStat:
    """How much of a run one provider carried, and how slowly it answered."""

    provider: str
    requests: int
    total_ms: float
    share: float

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.requests if self.requests else 0.0


@dataclass(slots=True)
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def executed(self) -> int:
        return sum(1 for result in self.results if not result.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for result in self.results if result.skipped)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed and not result.skipped)

    @property
    def failures(self) -> int:
        return self.executed - self.passed

    @property
    def pass_rate(self) -> float:
        if self.executed == 0:
            return 0.0
        return self.passed / self.executed

    @property
    def providers(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for result in self.results:
            if not result.skipped:
                counts[result.answer.provider] += 1
        return dict(sorted(counts.items()))

    def provider_stats(self) -> tuple[ProviderStat, ...]:
        """Requests carried and average latency per provider, busiest first.

        A run where the primary tier fell back shows up here as two providers sharing the load
        rather than one provider answering everything, which is the signal that a tier is
        degraded in a way the pass rate alone hides.
        """
        counts: Counter[str] = Counter()
        # A Counter counts, so its value type would be int and it is the wrong container for a sum.
        # A plain dict keeps latency as float: mypy does not widen an int-valued Counter from how it
        # is later used, and int(...) here would have silently truncated every provider's latency to
        # whole milliseconds.
        totals: dict[str, float] = {}
        for result in self.results:
            if result.skipped:
                continue
            provider = result.answer.provider
            counts[provider] += 1
            totals[provider] = totals.get(provider, 0.0) + result.latency_ms
        executed = sum(counts.values())
        return tuple(
            ProviderStat(
                provider=name,
                requests=count,
                total_ms=totals[name],
                share=(count / executed) if executed else 0.0,
            )
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )

    def meets(self, min_pass_rate: float) -> bool:
        return self.executed > 0 and self.pass_rate >= min_pass_rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": self.executed + self.skipped,
            "executed": self.executed,
            "skipped": self.skipped,
            "passed": self.passed,
            "failed": self.failures,
            "pass_rate": round(self.pass_rate, 4),
            "duration_ms": round(self.duration_ms, 1),
            "providers": self.providers,
            "results": [
                {
                    "case": result.case.name,
                    "group": result.case.group,
                    "capability": result.case.capability,
                    "skipped": result.skipped,
                    "passed": result.passed,
                    "failures": list(result.failures),
                    "provider": None if result.skipped else result.answer.provider,
                    "latency_ms": round(result.latency_ms, 1),
                }
                for result in self.results
            ],
        }


class EvalRunner:
    """Run the catalogue through an answer callable and grade every reply.

    ``func`` receives each :class:`EvalCase` and returns an :class:`Answer`; the runner never
    talks to providers or stores itself, so the same code drives a live provider in production
    and a stub in tests. Any exception raised by ``func`` becomes a failed case instead of
    aborting the suite, which is what a provider outage should look like: many failures and a
    clear provider distribution, not a crash.
    """

    def __init__(
        self,
        func: AnswerCase,
        *,
        capabilities: tuple[Capability, ...] = ("none",),
    ) -> None:
        self._func = func
        self._capabilities = set(capabilities)

    async def run(
        self,
        cases: tuple[EvalCase, ...] | list[EvalCase] | None = None,
        *,
        report_metrics: bool = True,
    ) -> EvalReport:
        selected = tuple(cases) if cases is not None else EVAL_CASES
        results: list[EvalResult] = []
        started = time.perf_counter()
        for case in selected:
            if case.capability not in self._capabilities:
                results.append(
                    EvalResult(case=case, answer=Answer(text="", error="skipped"), skipped=True)
                )
                if report_metrics:
                    metrics.EVALS.labels(case=case.name, result="skipped", provider="none").inc()
                continue
            began = time.perf_counter()
            try:
                answer = await self._func(case)
            except Exception as exc:
                answer = Answer(text="", error=f"{type(exc).__name__}: {exc}")
            latency = (time.perf_counter() - began) * 1000
            failures = grade(case, answer)
            results.append(
                EvalResult(case=case, answer=answer, failures=failures, latency_ms=latency)
            )
            if report_metrics:
                metrics.EVALS.labels(
                    case=case.name,
                    result="passed" if not failures else "failed",
                    provider=answer.provider,
                ).inc()
        return EvalReport(results=results, duration_ms=(time.perf_counter() - started) * 1000)


def render_table(report: EvalReport) -> str:
    """A human-readable report for the terminal and for CI step summaries."""
    lines = [f"{'case':<22} {'group':<12} {'cap':<11} {'result':<7} provider", "-" * 78]
    for result in report.results:
        status = "skip" if result.skipped else ("PASS" if result.passed else "FAIL")
        provider = "-" if result.skipped else result.answer.provider
        lines.append(
            f"{result.case.name:<22} {result.case.group:<12} {result.case.capability:<11} "
            f"{status:<7} {provider}"
        )
    lines.append("-" * 78)
    lines.append(
        f"passed {report.passed}/{report.executed} "
        f"({report.pass_rate:.0%}), skipped {report.skipped}, "
        f"providers: {', '.join(f'{p}:{n}' for p, n in report.providers.items()) or 'none'}, "
        f"{report.duration_ms:.0f} ms"
    )
    if report.failures:
        lines.append("failures:")
        for result in report.results:
            if not result.skipped and result.failures:
                lines.append(f"  {result.case.name}: {'; '.join(result.failures)}")
    return "\n".join(lines)


def render_markdown(report: EvalReport) -> str:
    """The report written to EVAL_REPORT.md and appended to a CI job summary.

    Three things a reader needs and the terminal table does not give them: the pass rate as a
    headline, the average latency per provider, and how the requests were distributed across
    tiers, which is how a silent fallback shows up.
    """
    lines = [
        "# Limbot evaluation report",
        "",
        (
            f"**Pass rate {report.pass_rate:.0%}** — {report.passed}/{report.executed} executed "
            f"passed, {report.skipped} skipped, {report.duration_ms / 1000:.1f} s total."
        ),
        "",
        "## Request distribution",
        "",
    ]

    stats = report.provider_stats()
    if not stats:
        lines.append("No case reached a provider, so there is no distribution to report.")
    else:
        lines.append("| provider | requests | share | avg latency |")
        lines.append("| --- | ---: | ---: | ---: |")
        for stat in stats:
            lines.append(
                f"| {stat.provider} | {stat.requests} | {stat.share:.0%} | "
                f"{stat.average_ms:.0f} ms |"
            )

    lines += [
        "",
        "## Cases",
        "",
        "| case | group | capability | result | provider | latency |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for result in report.results:
        status = "skip" if result.skipped else ("pass" if result.passed else "FAIL")
        provider = "-" if result.skipped else result.answer.provider
        latency = "-" if result.skipped else f"{result.latency_ms:.0f} ms"
        lines.append(
            f"| `{result.case.name}` | {result.case.group} | {result.case.capability} | "
            f"{status} | {provider} | {latency} |"
        )

    failing = [result for result in report.results if not result.skipped and result.failures]
    lines += ["", "## Failures", ""]
    if not failing:
        lines.append("None.")
    else:
        for result in failing:
            lines.append(f"- `{result.case.name}`: {'; '.join(result.failures)}")

    return "\n".join(lines) + "\n"
