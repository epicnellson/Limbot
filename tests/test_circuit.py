from __future__ import annotations

import pytest
from app.core.circuit import CircuitBreaker, CircuitState


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_breaker(clock: FakeClock, *, threshold: int = 3, recovery: float = 30.0) -> CircuitBreaker:
    return CircuitBreaker(
        name="test", failure_threshold=threshold, recovery_seconds=recovery, clock=clock
    )


def test_starts_closed_and_allows_requests() -> None:
    breaker = make_breaker(FakeClock())

    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request() is True
    assert breaker.snapshot().consecutive_failures == 0


def test_opens_after_reaching_the_threshold() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=3)

    for _ in range(2):
        assert breaker.allow_request() is True
        breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED

    breaker.allow_request()
    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    assert breaker.allow_request() is False


def test_success_resets_the_failure_count() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=3)

    breaker.allow_request()
    breaker.record_failure()
    breaker.allow_request()
    breaker.record_failure()
    breaker.allow_request()
    breaker.record_success()
    breaker.allow_request()
    breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.snapshot().consecutive_failures == 1


def test_half_open_admits_a_limited_number_of_probes() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=1, recovery=10.0)

    breaker.allow_request()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    clock.advance(10.0)
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request() is True
    assert breaker.allow_request() is False

    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_failed_probe_reopens_the_circuit() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=1, recovery=10.0)

    breaker.allow_request()
    breaker.record_failure()
    clock.advance(10.0)
    breaker.allow_request()
    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    assert breaker.snapshot().half_open_in_flight == 0

    clock.advance(9.0)
    assert breaker.allow_request() is False
    clock.advance(1.0)
    assert breaker.allow_request() is True


def test_repeated_failures_in_half_open_do_not_double_count() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=1, recovery=5.0)

    breaker.allow_request()
    breaker.record_failure()
    clock.advance(5.0)
    breaker.allow_request()
    breaker.record_failure()
    assert breaker.snapshot().half_open_in_flight == 0


def test_snapshot_reports_seconds_until_probe() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=1, recovery=30.0)

    breaker.allow_request()
    breaker.record_failure()
    clock.advance(20.0)

    assert breaker.snapshot().seconds_until_probe == 10.0
    assert breaker.snapshot().as_dict()["state"] == "open"


def test_reset_returns_to_closed() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=1)

    breaker.allow_request()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    breaker.reset()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request() is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failure_threshold": 0},
        {"recovery_seconds": -1.0},
        {"half_open_calls": 0},
    ],
)
def test_rejects_nonsense_thresholds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(name="test", **kwargs)  # type: ignore[arg-type]
