"""Synthetic evaluation harness for the answering pipeline.

Run from the repository root with real provider credentials:

    python -m app.eval --min-pass-rate 0.8

The suite exits non-zero when the pass rate drops below the gate, which is what a GitHub
Actions job uses to block a prompt change that regresses behaviour.
"""

from app.eval.cases import EVAL_CASES, Capability, EvalCase
from app.eval.runner import EvalReport, EvalResult, EvalRunner, grade, render_table

__all__ = [
    "EVAL_CASES",
    "Capability",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "grade",
    "render_table",
]
