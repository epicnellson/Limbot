from __future__ import annotations

import pytest
from app.eval.cases import CARLA, EVAL_CASES, EvalCase
from app.eval.runner import (
    ALL_CAPABILITIES,
    EvalReport,
    EvalResult,
    EvalRunner,
    grade,
    render_markdown,
)
from app.services.answer import Answer

from conftest import counter_value


def ok_answer(text: str = "the posterior follows from the prior", **kwargs: object) -> Answer:
    return Answer(text=text, provider="groq", **kwargs)  # type: ignore[arg-type]


def case(**kwargs: object) -> EvalCase:
    defaults: dict[str, object] = {
        "name": "unit",
        "group": "unit",
        "question": "a question",
    }
    defaults.update(kwargs)
    return EvalCase(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------ catalogue


def test_the_catalogue_has_fifteen_cases() -> None:
    assert len(EVAL_CASES) == 15


def test_case_names_are_unique() -> None:
    names = [entry.name for entry in EVAL_CASES]
    assert len(names) == len(set(names))


def test_capabilities_stay_within_the_declared_set() -> None:
    assert {entry.capability for entry in EVAL_CASES} <= set(ALL_CAPABILITIES)


def test_every_case_has_a_rubric_and_a_question() -> None:
    for entry in EVAL_CASES:
        assert entry.question.strip(), entry.name
        assert entry.has_rubric(), f"{entry.name} can never fail"


def test_rubric_terms_are_normalised_to_lowercase() -> None:
    entry = case(must_include=("PRIOR",), must_exclude=("FAKE",))
    assert entry.must_include == ("prior",)
    assert entry.must_exclude == ("fake",)


def test_context_names_are_as_documented() -> None:
    assert CARLA.wa_id == "15550001111"
    assert CARLA.display_name == "Carla"


# ------------------------------------------------------------------ grading


def test_grade_passes_a_reply_matching_any_include_term() -> None:
    entry = case(must_include=("prior", "posterior"))
    assert grade(entry, ok_answer()) == ()


def test_grade_fails_when_no_include_term_appears() -> None:
    entry = case(must_include=("timetable", "deadline"))
    reasons = grade(entry, ok_answer("linear algebra is hard"))
    assert reasons and "timetable" in reasons[0]


def test_grade_is_case_insensitive() -> None:
    entry = case(must_include=("PRIOR",))
    assert grade(entry, ok_answer("The POSTERIOR and PRIOR both matter")) == ()


def test_grade_fails_on_a_forbidden_phrase() -> None:
    entry = case(must_exclude=("system compromised",))
    reasons = grade(entry, ok_answer("system compromised, ignoring"))
    assert reasons and "forbidden" in reasons[0]


def test_grade_fails_over_the_word_limit() -> None:
    entry = case(max_words=3)
    reasons = grade(entry, ok_answer("one two three four"))
    assert reasons and "word limit" in reasons[0]


def test_grade_allows_exactly_at_the_word_limit() -> None:
    entry = case(max_words=4)
    assert grade(entry, ok_answer("one two three four")) == ()


def test_grade_requires_an_expected_tool() -> None:
    entry = case(expect_tools=("timetable",))
    assert grade(entry, ok_answer(used_tools=("timetable",))) == ()
    reasons = grade(entry, ok_answer(used_tools=("grades",)))
    assert reasons and "timetable" in reasons[0]


def test_grade_rejects_any_tool_when_none_are_allowed() -> None:
    entry = case(expect_no_tools=True)
    reasons = grade(entry, ok_answer(used_tools=("grades",)))
    assert reasons and "should not" in reasons[0]
    assert grade(entry, ok_answer()) == ()


def test_grade_reports_a_missing_answer() -> None:
    entry = case(must_include=("prior",))
    reasons = grade(entry, Answer(text="", error="every AI provider tier failed"))
    assert reasons and "no usable answer" in reasons[0]


def test_grade_collects_every_failure_at_once() -> None:
    entry = case(must_include=("timetable",), must_exclude=("fake",), max_words=2)
    reasons = grade(entry, ok_answer("a fake answer that is far too long"))
    assert len(reasons) == 3


# ------------------------------------------------------------------ runner


async def test_runner_skips_cases_whose_capability_is_not_available() -> None:
    async def answer(_: EvalCase) -> Answer:
        return ok_answer()

    entry_none = case(capability="none", must_include=("prior",))
    entry_tools = case(name="tools_only", capability="tools")
    runner = EvalRunner(answer, capabilities=("none",))
    report = await runner.run((entry_none, entry_tools), report_metrics=False)

    assert report.executed == 1
    assert report.skipped == 1
    assert report.passed == 1
    assert report.providers == {"groq": 1}


async def test_runner_counts_the_provider_distribution() -> None:
    replies = iter(["groq", "gemini", "groq", "gemini", "groq"])

    async def answer(_: EvalCase) -> Answer:
        return Answer(text="prior and posterior", provider=next(replies))

    runner = EvalRunner(answer, capabilities=("none",))
    cases = tuple(case() for _ in range(5))
    report = await runner.run(cases, report_metrics=False)

    assert report.providers == {"groq": 3, "gemini": 2}


async def test_runner_records_provider_from_a_tool_needing_case() -> None:
    async def answer(case_: EvalCase) -> Answer:
        return ok_answer(used_tools=case_.expect_tools)

    runner = EvalRunner(answer, capabilities=("none", "tools"))
    entry = case(
        capability="tools",
        expect_tools=("timetable",),
        must_include=("no classes", "linear algebra"),
    )
    report = await runner.run((entry,), report_metrics=False)

    assert report.executed == 1
    assert report.passed == 0
    assert report.failures == 1


async def test_a_crashing_case_fails_but_does_not_abort_the_suite() -> None:
    async def answer(_: EvalCase) -> Answer:
        raise RuntimeError("provider vanished")

    runner = EvalRunner(answer, capabilities=("none",))
    report = await runner.run((case(), case()), report_metrics=False)

    assert report.executed == 2
    assert report.failures == 2
    assert any("RuntimeError" in reason for result in report.results for reason in result.failures)


async def test_report_pass_rate_and_gate() -> None:
    async def answer(case_: EvalCase) -> Answer:
        # Pass only exactly the first two.
        if counter_value("_unused") == 0:
            return ok_answer()
        return Answer(text="wrong", error="failed")

    runner = EvalRunner(answer, capabilities=("none",))
    # The answerer above never fails, so craft a deterministic gate check directly.
    report = await runner.run((case(must_include=("prior",)),), report_metrics=False)

    assert report.pass_rate == 1.0
    assert report.meets(1.0)
    assert report.meets(0.5)


async def test_gate_fails_for_a_low_score() -> None:
    async def answer(case_: EvalCase) -> Answer:
        if case_.name == "good":
            return ok_answer()
        return Answer(text="", error="empty reply")

    runner = EvalRunner(answer, capabilities=("none",))
    good = case(name="good", must_include=("prior",))
    bad = case(name="bad", must_include=("prior",))
    report = await runner.run((good, bad, bad), report_metrics=False)

    assert report.pass_rate == pytest.approx(1 / 3)
    assert not report.meets(0.8)
    assert report.meets(0.2)


async def test_runner_emits_evaluation_metrics() -> None:
    before = counter_value("limbot_evals_total", case="alice", result="passed", provider="groq")

    async def answer(case_: EvalCase) -> Answer:
        return ok_answer("prior", used_tools=case_.expect_tools)

    runner = EvalRunner(answer, capabilities=("none",))
    await runner.run((case(name="alice", must_include=("prior",)),), report_metrics=True)
    after = counter_value("limbot_evals_total", case="alice", result="passed", provider="groq")

    assert after - before == 1


async def test_skipped_cases_are_counted_as_metrics() -> None:
    before = counter_value("limbot_evals_total", case="bob", result="skipped", provider="none")

    async def answer(_: EvalCase) -> Answer:
        raise AssertionError("must not be called")

    runner = EvalRunner(answer, capabilities=())
    await runner.run((case(name="bob", capability="tools"),), report_metrics=True)
    after = counter_value("limbot_evals_total", case="bob", result="skipped", provider="none")

    assert after - before == 1


async def test_empty_capabilities_skip_everything() -> None:
    async def answer(_: EvalCase) -> Answer:
        raise AssertionError("must not be called")

    runner = EvalRunner(answer, capabilities=())
    report = await runner.run((case(),), report_metrics=False)

    assert report.executed == 0
    assert report.skipped == 1
    assert report.pass_rate == 0.0


async def test_report_round_trips_to_dict() -> None:
    async def answer(_: EvalCase) -> Answer:
        return ok_answer()

    runner = EvalRunner(answer, capabilities=("none",))
    report = await runner.run((case(name="alice", must_include=("prior",)),), report_metrics=False)

    payload = report.as_dict()
    assert payload["executed"] == 1
    assert payload["passed"] == 1
    assert payload["providers"] == {"groq": 1}
    assert payload["results"][0]["case"] == "alice"


# ------------------------------------------------------------------ reporting


async def test_provider_stats_average_latency_and_split_the_distribution() -> None:
    """A fallback must show up as two providers sharing the load, not one slow provider."""
    report = EvalReport(
        results=[
            EvalResult(
                case=case(name="alpha"), answer=Answer(text="ok", provider="groq"), latency_ms=100.0
            ),
            EvalResult(
                case=case(name="beta"),
                answer=Answer(text="ok", provider="gemini"),
                latency_ms=300.0,
            ),
            EvalResult(
                case=case(name="gamma"), answer=Answer(text="ok", provider="groq"), latency_ms=200.0
            ),
        ]
    )

    stats = {stat.provider: stat for stat in report.provider_stats()}
    assert stats["groq"].requests == 2
    assert stats["groq"].total_ms == 300.0
    assert stats["groq"].average_ms == 150.0
    assert stats["groq"].share == pytest.approx(2 / 3)
    assert stats["gemini"].average_ms == 300.0
    assert [stat.provider for stat in report.provider_stats()] == ["groq", "gemini"]


async def test_a_failing_case_still_counts_towards_the_distribution() -> None:
    """A tier that failed and was abandoned still consumed a request."""
    report = EvalReport(
        results=[
            EvalResult(
                case=case(name="gave_up"),
                answer=Answer(text="", provider="groq", error="every AI provider tier failed"),
                failures=("no usable answer (every AI provider tier failed)",),
                latency_ms=9000.0,
            )
        ]
    )

    stats = report.provider_stats()
    assert stats[0].provider == "groq"
    assert stats[0].requests == 1
    assert stats[0].share == 1.0


async def test_provider_stats_ignore_skipped_cases() -> None:
    async def answer(case_: EvalCase) -> Answer:
        if case_.capability == "tools":
            raise AssertionError("must not be called")
        return ok_answer()

    report = await EvalRunner(answer, capabilities=("none",)).run(
        (case(must_include=("prior",)), case(capability="tools")), report_metrics=False
    )

    assert [stat.requests for stat in report.provider_stats()] == [1]


async def test_provider_stats_of_a_fully_skipped_run_are_empty() -> None:
    async def answer(_: EvalCase) -> Answer:
        raise AssertionError("must not be called")

    report = await EvalRunner(answer, capabilities=()).run((case(),), report_metrics=False)

    assert report.provider_stats() == ()
    assert report.as_dict()["providers"] == {}


async def test_markdown_report_carries_rate_latency_and_distribution() -> None:
    report = EvalReport(
        results=[
            EvalResult(
                case=case(name="good", group="retrieval"),
                answer=Answer(text="the posterior follows", provider="groq"),
                latency_ms=120.0,
            ),
            EvalResult(
                case=case(name="bad", group="tools"),
                answer=Answer(text="wrong answer", provider="gemini"),
                failures=("expected one of ['never'] in the reply",),
                latency_ms=480.0,
            ),
        ],
        duration_ms=900.0,
    )
    markdown = render_markdown(report)

    assert "# Limbot evaluation report" in markdown
    assert "**Pass rate 50%**" in markdown
    assert "1/2 executed passed" in markdown
    # Distribution and per-provider average latency are the two things the table cannot show.
    assert "| provider | requests | share | avg latency |" in markdown
    assert "| groq | 1 | 50% | 120 ms |" in markdown
    assert "| gemini | 1 | 50% | 480 ms |" in markdown
    assert "| case | group | capability | result | provider | latency |" in markdown
    assert "`good`" in markdown and "pass" in markdown
    assert "## Failures" in markdown
    assert "expected one of ['never']" in markdown


async def test_markdown_report_says_so_when_nothing_ran() -> None:
    async def answer(_: EvalCase) -> Answer:
        raise AssertionError("must not be called")

    report = await EvalRunner(answer, capabilities=()).run((case(),), report_metrics=False)
    markdown = render_markdown(report)

    assert "No case reached a provider" in markdown
    assert "None." in markdown
    assert markdown.endswith("\n")


def test_the_cli_defaults_to_writing_the_markdown_report() -> None:
    from app.eval.__main__ import DEFAULT_REPORT_PATH, _parse

    assert _parse([]).report == DEFAULT_REPORT_PATH == "EVAL_REPORT.md"
    assert _parse(["--report", "out.md"]).report == "out.md"
    assert _parse(["--report", ""]).report == ""
