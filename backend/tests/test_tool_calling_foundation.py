"""Tests for Structured Safe Tool Calling Foundation."""

import asyncio
import pytest
from pydantic import BaseModel

from app.services.tools import tool_registry, BaseTool, ToolContext, ToolResult
from app.services.tools.builtin import CalculatorTool, CodeAnalysisTool, DocumentLookupTool


@pytest.mark.asyncio
async def test_calculator_tool_valid_math():
    calc = CalculatorTool()
    res = await calc.execute({"expression": "sqrt(144) + 2 * 3.5"})
    assert res.success is True
    assert res.output["result"] == 19.0


@pytest.mark.asyncio
async def test_calculator_tool_arbitrary_code_injection_blocked():
    calc = CalculatorTool()
    malicious_expressions = [
        "__import__('os').system('whoami')",
        "eval('1 + 1')",
        "exec('import sys')",
        "open('secret.txt').read()",
        "globals()",
    ]
    for expr in malicious_expressions:
        res = await calc.execute({"expression": expr})
        assert res.success is False, f"Exploit was not blocked: {expr}"
        assert res.error is not None


@pytest.mark.asyncio
async def test_calculator_tool_dos_exponent_blocked():
    calc = CalculatorTool()
    res = await calc.execute({"expression": "2 ** 10001"})
    assert res.success is False
    assert "Exponent too large" in res.error


@pytest.mark.asyncio
async def test_code_analysis_tool():
    tool = CodeAnalysisTool()
    python_code = """
import os
import math

class DataProcessor:
    def process(self, x: int) -> int:
        return x * 2

def standalone_helper():
    pass
"""
    res = await tool.execute({"code": python_code})
    assert res.success is True
    out = res.output
    assert out["syntax_valid"] is True
    assert "DataProcessor" in out["classes"]
    assert "process" in out["functions"]
    assert "standalone_helper" in out["functions"]
    assert "os" in out["imports"]
    assert "math" in out["imports"]


@pytest.mark.asyncio
async def test_tool_execution_timeout():
    class SlowToolInput(BaseModel):
        delay: float

    class SlowTool(BaseTool):
        name = "slow_test_tool"
        description = "Simulates slow execution"
        input_model = SlowToolInput
        timeout_seconds = 0.1

        async def run(self, input_data: SlowToolInput, context: ToolContext):
            await asyncio.sleep(input_data.delay)
            return "finished"

    tool = SlowTool()
    res = await tool.execute({"delay": 0.5})
    assert res.success is False
    assert "timed out" in res.error.lower()


@pytest.mark.asyncio
async def test_tool_registry_catalog_and_dispatch():
    tools = tool_registry.list_tools()
    assert len(tools) >= 4
    tool_names = [t["name"] for t in tools]
    assert "calculator" in tool_names
    assert "web_search" in tool_names
    assert "document_lookup" in tool_names
    assert "code_analysis" in tool_names

    # Unknown tool
    res = await tool_registry.execute_tool("nonexistent_tool", {})
    assert res.success is False
    assert "not registered" in res.error
