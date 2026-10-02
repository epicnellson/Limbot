from __future__ import annotations

from app.config import Settings
from app.conversation.store import ConversationStore

from conftest import _base_settings


def make_store(
    *,
    max_messages: int = 16,
    ttl: int = 1800,
    max_conversations: int = 500,
) -> tuple[ConversationStore, list[float]]:
    """A store on a hand-cranked clock, so TTL behaviour is tested without sleeping."""
    now = [1000.0]
    store = ConversationStore(
        _base_settings(
            conversation_max_messages=max_messages,
            conversation_ttl_seconds=ttl,
            conversation_max_conversations=max_conversations,
        ),
        clock=lambda: now[0],
    )
    return store, now


def test_history_is_empty_until_something_is_recorded() -> None:
    store, _now = make_store()

    assert store.history("15550001111") == ()
    assert len(store) == 0


def test_a_recorded_exchange_comes_back_as_two_messages() -> None:
    store, _now = make_store()

    store.record("15550001111", "what is on today?", "Linear algebra at 10:00 in B204.")

    messages = store.history("15550001111")
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "what is on today?"
    assert messages[1].content == "Linear algebra at 10:00 in B204."


def test_conversations_are_kept_per_sender() -> None:
    store, _now = make_store()

    store.record("15550001111", "mine", "mine")
    store.record("15550002222", "theirs", "theirs")

    assert [m.content for m in store.history("15550001111")] == ["mine", "mine"]
    assert [m.content for m in store.history("15550002222")] == ["theirs", "theirs"]


def test_the_oldest_exchange_is_dropped_once_the_window_is_full() -> None:
    store, _now = make_store(max_messages=4)

    for index in range(4):
        store.record("15550001111", f"q{index}", f"a{index}")

    contents = [m.content for m in store.history("15550001111")]
    assert contents == ["q2", "a2", "q3", "a3"]


def test_a_history_is_emptied_once_it_goes_quiet() -> None:
    store, now = make_store(ttl=60)

    store.record("15550001111", "q", "a")
    now[0] += 59
    assert len(store.history("15550001111")) == 2

    now[0] += 2
    assert store.history("15550001111") == ()
    assert len(store) == 0


def test_expired_senders_are_evicted_even_without_a_read() -> None:
    store, now = make_store(ttl=60)

    store.record("15550001111", "q", "a")
    store.record("15550002222", "q", "a")
    assert len(store) == 2

    now[0] += 120
    store.record("15550003333", "q", "a")

    assert len(store) == 1
    assert store.history("15550001111") == ()


def test_the_number_of_tracked_senders_is_capped() -> None:
    store, _now = make_store(max_conversations=2)

    for index in range(5):
        store.record(f"1555000{index:04d}", "q", "a")

    assert len(store) == 2


def test_evicting_a_sender_drops_the_least_recently_active_one() -> None:
    store, _now = make_store(max_conversations=2)

    store.record("15550001111", "q", "a")
    store.record("15550002222", "q", "a")
    store.record("15550001111", "q2", "a2")  # refreshes the first sender
    store.record("15550003333", "q", "a")  # pushes the second sender out

    assert len(store) == 2
    assert store.history("15550001111")
    assert store.history("15550002222") == ()


def test_tools_used_is_deduplicated_and_ordered_by_first_use() -> None:
    store, _now = make_store()

    store.record("15550001111", "q", "a", used_tools=("list_courses",))
    store.record("15550001111", "q", "a", used_tools=("get_timetable", "list_courses"))

    assert store.tools_used("15550001111") == ("list_courses", "get_timetable")


def test_blank_questions_and_answers_are_not_carried_forward() -> None:
    store, _now = make_store()

    store.record("15550001111", "  ", "a")

    assert [m.role for m in store.history("15550001111")] == ["assistant"]


def test_clearing_forgets_a_sender() -> None:
    store, _now = make_store()

    store.record("15550001111", "q", "a")
    store.clear("15550001111")

    assert store.history("15550001111") == ()
    assert len(store) == 0


def test_the_window_is_derived_from_the_message_budget() -> None:
    settings = _base_settings(conversation_max_messages=10)
    store = ConversationStore(settings)

    for index in range(6):
        store.record("15550001111", f"q{index}", f"a{index}")

    # Ten messages means five question and answer pairs, so two pairs are too many.
    assert len(store.history("15550001111")) == 10


def test_the_store_defaults_to_the_process_clock() -> None:
    store = ConversationStore(_base_settings())

    store.record("15550001111", "q", "a")

    assert store.history("15550001111")


def test_settings_without_a_conversation_budget_still_work() -> None:
    settings: Settings = _base_settings(conversation_max_messages=0)

    store = ConversationStore(settings)
    store.record("15550001111", "q", "a")

    assert store.history("15550001111") == ()
