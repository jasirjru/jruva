"""Base Structured Tool Abstraction & Secure Execution Contract."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolContext(BaseModel):
    """Execution context and caller identity passed to tool executions."""
    user_id: str = "guest"
    session_id: Optional[str] = None
    is_authenticated: bool = False


class ToolResult(BaseModel):
    """Structured response contract for all tool executions."""
    tool_name: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class BaseTool(ABC):
    """Abstract base class for type-safe, sandboxed tool definitions."""

    name: str
    description: str
    input_model: Type[BaseModel]
    timeout_seconds: float = 5.0

    @abstractmethod
    async def run(self, input_data: Any, context: ToolContext) -> Any:
        """Inner execution logic to be implemented by each specific tool."""
        pass

    async def execute(self, raw_input: Dict[str, Any], context: Optional[ToolContext] = None) -> ToolResult:
        """Safely execute tool with validation, timeouts, and error isolation."""
        import time
        start = time.perf_counter()
        ctx = context or ToolContext()

        # 1. Validate Input
        try:
            validated_input = self.input_model(**raw_input)
        except Exception as val_err:
            elapsed = (time.perf_counter() - start) * 1000.0
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Input validation error: {str(val_err)}",
                execution_time_ms=round(elapsed, 2),
            )

        # 2. Execute with strict timeout
        try:
            result_data = await asyncio.wait_for(
                self.run(validated_input, ctx),
                timeout=self.timeout_seconds,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=result_data,
                execution_time_ms=round(elapsed, 2),
            )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000.0
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Execution timed out after {self.timeout_seconds}s",
                execution_time_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.warning(f"Tool {self.name} failed during execution: {exc}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Tool error: {str(exc)}",
                execution_time_ms=round(elapsed, 2),
            )
