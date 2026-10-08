from __future__ import annotations

from dataclasses import replace

import httpx
import pytest
from app.config import Settings
from app.conversation.store import ConversationStore
from app.core.circuit import CircuitState
from app.llm.base import (
    LLMProvider,
    ProviderConfigError,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
)
from app.llm.pipeline import LLMPipeline, NoProviderAvailable
from app.llm.types import ChatMessage, LLMRequest, LLMResponse, ToolCall, ToolSpec
from app.rag.pipeline import RetrievalResult, RetrievedChunk
from app.services.answer import Answer, AnswerService
from app.tools.context import ToolContext
from app.tools.registry import ToolDefinition, ToolRegistry

from conftest import _base_settings

TIMETABLE = ToolSpec(
    name="get_timetable",
    description="Read the signed-in student's timetable.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)
DEADLINES = ToolSpec(
    name="get_upcoming_deadlines",
    description="Read the signed-in student's deadlines.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)

STUDENT = ToolContext(wa_id="15550001111", display_name="Ada")


class ScriptedProvider(LLMProvider):
    """Replays a fixed list of responses, so a whole conversation can be asserted exactly."""

    def __init__(
        self,
        script: list[LLMResponse],
        *,
        name: str = "scripted",
        tier: int = 1,
        error: ProviderError | None = None,
    ) -> None:
        super().__init__(f"{name}-model", name=name, tier=tier)
        self._script = list(script)
        self._error = error
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest, *, http: httpx.AsyncClient) -> LLMResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if not self._script:
            raise AssertionError("the provider was called more often than the script allows")
        return replace(self._script.pop(0), provider=self.name, model=self.model)


class FakeRetrieval:
    """Stands in for the vector store path so no embedder is needed."""

    def __init__(self, result: RetrievalResult) -> None:
        self._result = result
        self.queries: list[str] = []

    async def retrieve(self, question: str) -> RetrievalResult:
        self.queries.append(question)
        return self._result

    def describe(self) -> dict[str, object]:
        return {"enabled": True}


def says(value: str, provider: str = "scripted") -> LLMResponse:
    return LLMResponse(text=value, provider=provider, model="scripted-model")


def calls(*names: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=tuple(
            ToolCall(id=f"call_{index}", name=name, arguments={})
            for index, name in enumerate(names)
        ),
        provider="scripted",
        model="scripted-model",
    )


async def rows(context: ToolContext, arguments: dict[str, object]) -> dict[str, object]:
    return {"rows": [{"subject": "Linear algebra", "room": "B204"}], "wa_id": context.wa_id}


async def broken(context: ToolContext, arguments: dict[str, object]) -> dict[str, object]:
    raise RuntimeError("connection reset")


def make_service(
    script: list[LLMResponse],
    *,
    tools: ToolRegistry | None = None,
    retrieval: FakeRetrieval | None = None,
    conversations: ConversationStore | None = None,
    settings: Settings | None = None,
) -> tuple[AnswerService, ScriptedProvider]:
    resolved = settings or _base_settings(ai_enabled=True)
    provider = ScriptedProvider(script)
    service = AnswerService(
        resolved,
        LLMPipeline(resolved, providers=[provider]),
        retrieval=retrieval,  # type: ignore[arg-type]
        tools=tools,
        conversations=conversations,
    )
    return service, provider


def registry(*definitions: tuple[ToolSpec, object], timeout: float = 10.0) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(spec=spec, handler=handler, timeout_seconds=timeout)  # type: ignore[arg-type]
            for spec, handler in definitions
        ]
    )


def mock_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None))


async def test_a_plain_answer_comes_back_without_any_tool_being_available() -> None:
    service, provider = make_service([says("Linear algebra is at 10:00 in B204.")])

    async with mock_client() as http:
        answer = await service.answer("what is on today?", STUDENT, http=http)

    assert answer.text == "Linear algebra is at 10:00 in B204."
    assert answer.provider == "scripted"
    assert answer.used_tools == ()
    assert answer.tool_rounds == 0
    assert answer.ok
    assert provider.requests[0].tools == ()


async def test_without_tools_the_prompt_says_not_to_emit_a_tool_call() -> None:
    service, provider = make_service([says("No timetable is available.")])

    async with mock_client() as http:
        await service.answer("what is on today?", STUDENT, http=http)

    system = provider.requests[0].messages[0].content
    assert "No database tools are available" in system
    assert "do not emit a tool call" in system


async def test_with_tools_the_prompt_does_not_forbid_tool_calls() -> None:
    tools = registry((TIMETABLE, rows))
    service, provider = make_service([calls("get_timetable"), says("here it is")], tools=tools)

    async with mock_client() as http:
        await service.answer("what is on today?", STUDENT, http=http)

    system = provider.requests[0].messages[0].content
    assert "No database tools are available" not in system


async def test_the_prompt_carries_the_system_rules_and_the_student_name() -> None:
    service, provider = make_service([says("ok")])

    async with mock_client() as http:
        await service.answer("when is the exam?", STUDENT, http=http)

    messages = provider.requests[0].messages
    assert messages[0].role == "system"
    assert "Limbot" in messages[0].content
    assert "Ada" in messages[0].content
    assert messages[-1] == ChatMessage.user("when is the exam?")


async def test_a_sender_without_a_name_gets_no_name_in_the_prompt() -> None:
    service, provider = make_service([says("ok")])
    anonymous = ToolContext(wa_id="15550009999")

    async with mock_client() as http:
        await service.answer("when is the exam?", anonymous, http=http)

    assert provider.requests[0].messages[-1].content == "when is the exam?"


async def test_retrieved_material_is_given_to_the_model_with_its_source() -> None:
    retrieval = FakeRetrieval(
        RetrievalResult(
            chunks=(
                RetrievedChunk(
                    text="P(A|B) = P(B|A)P(A)/P(B)",
                    score=0.91,
                    title="Lecture 3",
                    source="notes.pdf",
                ),
            ),
            context="[1] Lecture 3\nP(A|B) = P(B|A)P(A)/P(B)",
            reason="hit",
            sources=("notes.pdf",),
        )
    )
    service, provider = make_service([says("That is Bayes' theorem.")], retrieval=retrieval)

    async with mock_client() as http:
        answer = await service.answer("what is bayes?", STUDENT, http=http)

    assert "P(A|B)" in provider.requests[0].messages[0].content
    assert answer.sources == ("notes.pdf",)
    assert answer.retrieved == 1
    assert retrieval.queries == ["what is bayes?"]


async def test_when_nothing_passes_the_threshold_the_model_is_told_to_say_so() -> None:
    retrieval = FakeRetrieval(RetrievalResult(reason="below_threshold", considered=4))
    service, provider = make_service([says("I could not find that.")], retrieval=retrieval)

    async with mock_client() as http:
        answer = await service.answer("who is the dean?", STUDENT, http=http)

    system = provider.requests[0].messages[0].content
    assert "could not find" in system
    assert "your own knowledge" in system
    assert "Course material retrieved" not in system
    assert answer.retrieved == 0
    assert answer.sources == ()


async def test_a_retrieval_failure_still_produces_an_answer() -> None:
    retrieval = FakeRetrieval(RetrievalResult(reason="error"))
    service, _provider = make_service([says("Here is what I know.")], retrieval=retrieval)

    async with mock_client() as http:
        answer = await service.answer("what is bayes?", STUDENT, http=http)

    assert answer.text == "Here is what I know."


async def test_a_tool_call_is_dispatched_for_the_verified_sender_and_its_rows_returned() -> None:
    tools = registry((TIMETABLE, rows))
    service, provider = make_service(
        [calls("get_timetable"), says("Linear algebra at 10:00 in B204.")], tools=tools
    )

    async with mock_client() as http:
        answer = await service.answer("what is on today?", STUDENT, http=http)

    assert answer.text == "Linear algebra at 10:00 in B204."
    assert answer.used_tools == ("get_timetable",)
    assert answer.tool_rounds == 1
    assert not answer.degraded

    follow_up = provider.requests[1].messages
    assert follow_up[-2].role == "assistant"
    assert follow_up[-2].tool_calls[0].name == "get_timetable"
    assert follow_up[-1].role == "tool"
    assert follow_up[-1].tool_call_id == "call_0"
    assert "B204" in follow_up[-1].content


async def test_several_tool_calls_in_one_response_are_all_answered() -> None:
    tools = registry((TIMETABLE, rows), (DEADLINES, rows))
    service, provider = make_service(
        [calls("get_timetable", "get_upcoming_deadlines"), says("Your week, briefly.")], tools=tools
    )

    async with mock_client() as http:
        answer = await service.answer("what is my week?", STUDENT, http=http)

    results = [m for m in provider.requests[1].messages if m.role == "tool"]
    assert [m.tool_call_id for m in results] == ["call_0", "call_1"]
    assert answer.used_tools == ("get_timetable", "get_upcoming_deadlines")


async def test_more_tool_calls_than_the_cap_are_truncated() -> None:
    specs = tuple(
        ToolSpec(
            name=f"tool_{index}",
            description="d",
            parameters={"type": "object", "properties": {}},
        )
        for index in range(9)
    )
    tools = ToolRegistry(
        [ToolDefinition(spec=spec, handler=rows) for spec in specs]  # type: ignore[arg-type]
    )
    script = [calls(*[spec.name for spec in specs]), says("done")]
    service, provider = make_service(script, tools=tools)

    async with mock_client() as http:
        answer = await service.answer("run everything", STUDENT, http=http)

    results = [m for m in provider.requests[1].messages if m.role == "tool"]
    assert len(results) == 4
    assert answer.text == "done"


async def test_a_failing_tool_is_reported_back_to_the_model_instead_of_raising() -> None:
    tools = registry((TIMETABLE, broken))
    service, provider = make_service(
        [calls("get_timetable"), says("I could not reach the records.")], tools=tools
    )

    async with mock_client() as http:
        answer = await service.answer("what is on today?", STUDENT, http=http)

    assert answer.text == "I could not reach the records."
    assert answer.used_tools == ("get_timetable",)
    assert "failed" in provider.requests[1].messages[-1].content


async def test_an_unknown_tool_name_is_told_to_the_model_with_the_available_ones() -> None:
    tools = registry((TIMETABLE, rows))
    service, provider = make_service(
        [calls("delete_student"), says("I cannot do that.")], tools=tools
    )

    async with mock_client() as http:
        answer = await service.answer("drop my grades", STUDENT, http=http)

    content = provider.requests[1].messages[-1].content
    assert "unknown tool" in content
    assert "get_timetable" in content
    assert answer.text == "I cannot do that."


async def test_arguments_that_break_the_schema_are_refused_before_the_handler_runs() -> None:
    invoked = False

    async def tracking(context: ToolContext, arguments: dict[str, object]) -> dict[str, object]:
        nonlocal invoked
        invoked = True
        return {"rows": []}

    spec = ToolSpec(
        name="get_timetable",
        description="timetable",
        parameters={
            "type": "object",
            "properties": {"day": {"enum": ["monday", "tuesday"]}},
            "additionalProperties": False,
        },
    )
    tools = ToolRegistry([ToolDefinition(spec=spec, handler=tracking)])  # type: ignore[arg-type]
    bad_call = LLMResponse(
        tool_calls=(ToolCall(id="call_0", name="get_timetable", arguments={"day": "funday"}),),
        provider="scripted",
    )
    service, provider = make_service([bad_call, says("Sorry.")], tools=tools)

    async with mock_client() as http:
        await service.answer("what is on today?", STUDENT, http=http)

    assert invoked is False
    assert "invalid arguments" in provider.requests[1].messages[-1].content


async def test_the_loop_stops_after_the_configured_number_of_tool_rounds() -> None:
    tools = registry((TIMETABLE, rows))
    script = [calls("get_timetable") for _ in range(4)]
    service, provider = make_service(script, tools=tools)

    async with mock_client() as http:
        answer = await service.answer("again?", STUDENT, http=http)

    # Three model calls run the tools. On the fourth the tool calls are dropped, so a model that
    # keeps asking cannot hold the turn open: the student gets a reply either way.
    assert len(provider.requests) == 4
    assert answer.tool_rounds == 3
    assert answer.degraded is True
    assert answer.used_tools == ("get_timetable",)
    # The fourth call repeats the three tool results and asks for an answer, without the tool
    # call the model asked for, so the transcript stays well formed.
    final = provider.requests[3].messages
    assert final[-1].role == "tool"
    assert all(message.role != "tool" or message.content for message in final)


async def test_a_model_that_stops_asking_for_tools_is_not_marked_degraded() -> None:
    tools = registry((TIMETABLE, rows))
    service, _provider = make_service([calls("get_timetable"), says("here it is")], tools=tools)

    async with mock_client() as http:
        answer = await service.answer("today?", STUDENT, http=http)

    assert answer.degraded is False
    assert answer.tool_rounds == 1


async def test_an_empty_reply_is_replaced_with_something_sendable() -> None:
    service, _provider = make_service([says("   ")])

    async with mock_client() as http:
        answer = await service.answer("when?", STUDENT, http=http)

    assert answer.text
    assert answer.ok
    assert answer.degraded is True


async def test_the_exchange_is_carried_into_the_next_question() -> None:
    service, provider = make_service([says("first answer"), says("second answer")])

    async with mock_client() as http:
        await service.answer("first question", STUDENT, http=http)
        await service.answer("second question", STUDENT, http=http)

    carried = provider.requests[1].messages
    assert [m.content for m in carried[1:-1]] == ["first question", "first answer"]
    assert carried[-1].content == "second question"


async def test_one_senders_history_is_never_shown_to_another() -> None:
    service, provider = make_service([says("mine"), says("theirs")])
    other = ToolContext(wa_id="15550002222", display_name="Grace")

    async with mock_client() as http:
        await service.answer("my question", STUDENT, http=http)
        await service.answer("their question", other, http=http)

    second = provider.requests[1].messages
    assert [m.content for m in second[1:]] == ["their question"]


async def test_clearing_forgets_the_conversation() -> None:
    service, provider = make_service([says("mine"), says("fresh")])

    async with mock_client() as http:
        await service.answer("my question", STUDENT, http=http)
        service.reset(STUDENT.wa_id)
        await service.answer("my question again", STUDENT, http=http)

    assert [m.content for m in provider.requests[1].messages][1:] == ["my question again"]


async def test_a_pipeline_with_no_provider_reports_that_it_cannot_answer() -> None:
    settings = _base_settings(ai_enabled=True)
    service = AnswerService(settings, LLMPipeline(settings, providers=[]))

    assert not service.enabled
    async with mock_client() as http:
        answer = await service.answer("hello", STUDENT, http=http)

    assert answer == Answer(text="", error="no AI provider is configured")


async def test_ai_being_switched_off_disables_the_service_even_with_a_provider() -> None:
    settings = _base_settings(ai_enabled=False)
    provider = ScriptedProvider([says("never called")])
    service = AnswerService(settings, LLMPipeline(settings, providers=[provider]))

    assert not service.enabled
    async with mock_client() as http:
        await service.answer("hello", STUDENT, http=http)

    assert provider.requests == []


async def test_a_transient_failure_in_every_tier_ends_the_turn_without_a_reply() -> None:
    settings = _base_settings(ai_enabled=True)
    provider = ScriptedProvider(
        [], error=ProviderServerError("503 from provider", provider="scripted")
    )
    service = AnswerService(settings, LLMPipeline(settings, providers=[provider]))

    async with mock_client() as http:
        answer = await service.answer("hello", STUDENT, http=http)

    assert answer.text == ""
    assert answer.error == "every AI provider tier failed"
    assert not answer.ok


async def test_a_rate_limited_tier_falls_through_to_the_backup_and_records_the_failure() -> None:
    settings = _base_settings(ai_enabled=True)
    throttled = ScriptedProvider(
        [],
        name="groq",
        tier=1,
        error=ProviderRateLimitError("429", provider="groq", status_code=429),
    )
    backup = ScriptedProvider([says("answered by the backup")], name="gemini", tier=2)
    pipeline = LLMPipeline(settings, providers=[throttled, backup])
    service = AnswerService(settings, pipeline)

    async with mock_client() as http:
        answer = await service.answer("hello", STUDENT, http=http)

    assert answer.text == "answered by the backup"
    assert answer.provider == "gemini"
    assert pipeline.breakers["groq"].snapshot().consecutive_failures == 1
    assert pipeline.breakers["gemini"].state is CircuitState.CLOSED


async def test_a_rejected_request_is_not_replayed_on_the_next_tier() -> None:
    settings = _base_settings(ai_enabled=True)
    rejected = ScriptedProvider(
        [], name="groq", tier=1, error=ProviderConfigError("401 bad key", provider="groq")
    )
    backup = ScriptedProvider([says("should not be reached")], name="gemini", tier=2)
    service = AnswerService(settings, LLMPipeline(settings, providers=[rejected, backup]))

    async with mock_client() as http:
        with pytest.raises(ProviderConfigError):
            await service.answer("hello", STUDENT, http=http)

    assert backup.requests == []


async def test_the_no_provider_error_carries_every_attempt_for_the_logs() -> None:
    settings = _base_settings(ai_enabled=True)
    provider = ScriptedProvider([], error=ProviderServerError("503", provider="scripted"))
    pipeline = LLMPipeline(settings, providers=[provider])

    async with mock_client() as http:
        with pytest.raises(NoProviderAvailable) as raised:
            await pipeline.complete(LLMRequest(messages=()), http=http)

    attempts = raised.value.attempts
    assert [attempt.as_dict()["result"] for attempt in attempts] == ["transient_error"]
    assert attempts[0].as_dict()["provider"] == "scripted"


async def test_a_conversation_store_can_be_supplied_and_is_reused() -> None:
    conversations = ConversationStore(_base_settings())
    service, _provider = make_service([says("hi")], conversations=conversations)

    async with mock_client() as http:
        await service.answer("hello", STUDENT, http=http)

    assert conversations.history(STUDENT.wa_id)
