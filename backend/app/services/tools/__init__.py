"""Structured Tool Registry & Dispatcher for Jruva Assistant."""

import logging
from typing import Any, Dict, List, Optional

from app.services.tools.base import BaseTool, ToolContext, ToolResult
from app.services.tools.builtin import (
    CalculatorTool,
    CodeAnalysisTool,
    DocumentLookupTool,
    WebSearchTool,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages available tools and executes calls with isolation and telemetry."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.register(CalculatorTool())
        self.register(WebSearchTool())
        self.register(DocumentLookupTool())
        self.register(CodeAnalysisTool())

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return catalog of available tools with JSON input schemas."""
        catalog = []
        for name, tool in self._tools.items():
            schema = tool.input_model.model_json_schema()
            catalog.append({
                "name": name,
                "description": tool.description,
                "timeout_seconds": tool.timeout_seconds,
                "input_schema": schema,
            })
        return catalog

    async def execute_tool(
        self,
        name: str,
        raw_input: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is not registered or unavailable.",
            )
        return await tool.execute(raw_input=raw_input, context=context)


tool_registry = ToolRegistry()

__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    "CalculatorTool",
    "WebSearchTool",
    "DocumentLookupTool",
    "CodeAnalysisTool",
]
