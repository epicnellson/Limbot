"""Tool calling exposed to the model, and the student tools built on the database."""

from app.tools.context import ToolContext
from app.tools.registry import (
    ToolDefinition,
    ToolFailure,
    ToolRegistry,
    ToolResult,
)
from app.tools.schema import Problem, SchemaError, require_valid, validate_arguments
from app.tools.student import StudentTools, build_student_registry
from app.tools.tool_engine import (
    DEFAULT_DB_PATH,
    TOOL_NAMES,
    TOOLS,
    check_assignment_deadlines,
    check_student_timetable,
    close_tool_db,
    execute_tool,
    init_mock_db,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "TOOLS",
    "TOOL_NAMES",
    "Problem",
    "SchemaError",
    "StudentTools",
    "ToolContext",
    "ToolDefinition",
    "ToolFailure",
    "ToolRegistry",
    "ToolResult",
    "build_student_registry",
    "check_assignment_deadlines",
    "check_student_timetable",
    "close_tool_db",
    "execute_tool",
    "init_mock_db",
    "require_valid",
    "validate_arguments",
]
