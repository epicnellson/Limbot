from __future__ import annotations

from dataclasses import dataclass
from typing import Any

JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}



class SchemaError(ValueError):
    """Arguments did not match the declared schema."""


@dataclass(frozen=True, slots=True)
class Problem:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[Problem]:
    """Check arguments against the JSON schema subset the tool registry uses.

    Deliberately small: the tools declare flat objects of strings, numbers and enums, and a
    hand written checker is easier to audit than a general validator. The one behaviour that
    matters most is refusing unknown properties, because an unexpected key is either a model
    mistake or an attempt to smuggle in a field the tool never intended to accept.
    """
    problems: list[Problem] = []
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return [Problem("$", "schema properties must be an object")]

    required = schema.get("required") or []
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in arguments:
                problems.append(Problem(name, "is required"))

    for name, value in arguments.items():
        declared = properties.get(name)
        if declared is None:
            if schema.get("additionalProperties", True) is False:
                problems.append(Problem(name, "is not an accepted argument"))
            continue
        problems.extend(_check(name, declared, value))

    return problems


def _check(path: str, declared: Any, value: Any) -> list[Problem]:
    if not isinstance(declared, dict):
        return []
    expected = declared.get("type")
    if isinstance(expected, str):
        python_type = JSON_TYPES.get(expected)
        if python_type is None:
            return []
        if not _matches(python_type, value):
            return [Problem(path, f"must be of type {expected}")]

    problems: list[Problem] = []
    allowed = declared.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        options = ", ".join(str(option) for option in allowed)
        problems.append(Problem(path, f"must be one of: {options}"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = declared.get("minimum")
        maximum = declared.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            problems.append(Problem(path, f"must be at least {minimum}"))
        if isinstance(maximum, (int, float)) and value > maximum:
            problems.append(Problem(path, f"must be at most {maximum}"))

    if isinstance(value, list):
        item_schema = declared.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                problems.extend(_check(f"{path}[{index}]", item_schema, item))

    if isinstance(value, dict) and isinstance(declared.get("properties"), dict):
        for problem in validate_arguments(declared, value):
            problems.append(Problem(f"{path}.{problem.path}", problem.message))

    return problems


def _matches(python_type: type | tuple[type, ...], value: Any) -> bool:
    if python_type is bool:
        return isinstance(value, bool)
    if python_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(python_type, tuple):
        return isinstance(value, python_type) and not isinstance(value, bool)
    return isinstance(value, python_type)


def require_valid(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the arguments, or raise :class:`SchemaError` listing every problem."""
    if not isinstance(arguments, dict):
        raise SchemaError("arguments must be a JSON object")
    problems = validate_arguments(schema, arguments)
    if problems:
        raise SchemaError("; ".join(str(problem) for problem in problems))
    return arguments
