from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from app import metrics

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted through a circuit that is not accepting requests."""

    def __init__(self, name: str, state: CircuitState) -> None:
        super().__init__(f"circuit {name!r} is {state.value}")
        self.name = name
        self.state = state


@dataclass(frozen=True, slots=True)
class CircuitSnapshot:
    name: str
    state: CircuitState
    consecutive_failures: int
    half_open_in_flight: int
    opened_at: float | None
    seconds_until_probe: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "half_open_in_flight": self.half_open_in_flight,
            "seconds_until_probe": (
                round(self.seconds_until_probe, 1) if self.seconds_until_probe is not None else None
            ),
        }


class CircuitBreaker:
    """Per provider circuit breaker: closed, open, then a limited half open probe.

    The point is to stop hammering a provider that is down. Each tier gets its own instance,
    so a Groq outage never spends latency probing Gemini and never opens Gemini's circuit.

    Every method is synchronous and short, so it is safe on a single event loop; a lock is
    taken anyway so a future thread pool or a sync call site cannot corrupt the counters.
    """

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
        half_open_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_seconds < 0:
            raise ValueError("recovery_seconds must be >= 0")
        if half_open_calls < 1:
            raise ValueError("half_open_calls must be >= 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.half_open_calls = half_open_calls
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._half_open_in_flight = 0
        self._opened_at: float | None = None
        self._emit_state(CircuitState.CLOSED)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._refresh()

    def allow_request(self) -> bool:
        """Reserve capacity for one call. Call :meth:`record_success` or :meth:`record_failure`."""
        with self._lock:
            state = self._refresh()
            if state is CircuitState.CLOSED:
                return True
            if state is CircuitState.HALF_OPEN and self._half_open_in_flight < self.half_open_calls:
                self._half_open_in_flight += 1
                return True
            metrics.CIRCUIT_REJECTIONS.labels(provider=self.name).inc()
            return False

    def record_success(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._failures = 0
                self._transition(CircuitState.CLOSED)
            elif self._state is CircuitState.CLOSED:
                self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            metrics.CIRCUIT_FAILURES.labels(provider=self.name).inc()
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._transition(CircuitState.OPEN)
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._transition(CircuitState.OPEN)

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._half_open_in_flight = 0
            self._transition(CircuitState.CLOSED)

    def snapshot(self) -> CircuitSnapshot:
        with self._lock:
            state = self._refresh()
            wait: float | None = None
            if state is CircuitState.OPEN and self._opened_at is not None:
                wait = max(0.0, self._recovery_deadline() - self._clock())
            elif state is CircuitState.HALF_OPEN:
                wait = 0.0
            return CircuitSnapshot(
                name=self.name,
                state=state,
                consecutive_failures=self._failures,
                half_open_in_flight=self._half_open_in_flight,
                opened_at=self._opened_at,
                seconds_until_probe=wait,
            )

    def _refresh(self) -> CircuitState:
        if self._state is CircuitState.OPEN and self._clock() >= self._recovery_deadline():
            self._transition(CircuitState.HALF_OPEN)
        return self._state

    def _recovery_deadline(self) -> float:
        return (self._opened_at or 0.0) + self.recovery_seconds

    def _transition(self, target: CircuitState) -> None:
        if target is self._state:
            return
        previous = self._state
        self._state = target
        if target is CircuitState.OPEN:
            self._opened_at = self._clock()
        elif target is CircuitState.CLOSED:
            self._opened_at = None
            self._failures = 0
        logger.warning(
            "circuit breaker transition",
            extra={"context": {"provider": self.name, "from": previous.value, "to": target.value}},
        )
        metrics.CIRCUIT_TRANSITIONS.labels(provider=self.name, to_state=target.value).inc()
        self._emit_state(target)

    def _emit_state(self, state: CircuitState) -> None:
        for candidate in CircuitState:
            metrics.CIRCUIT_STATE.labels(provider=self.name, state=candidate.value).set(
                1 if candidate is state else 0
            )
