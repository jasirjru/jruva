"""Dedicated REST API Endpoints for Deep Coding Intelligence & Safe Sandbox Execution."""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.security import EXPENSIVE_ENDPOINT_LIMIT, limiter
from app.services.coding.code_intelligence_service import code_intelligence_service

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalyzeCodeRequest(BaseModel):
    code: str = Field(..., max_length=20000, description="Source code to analyze")
    language: str = Field("python", description="Programming language (default: python)")


class ExecuteCodeRequest(BaseModel):
    code: str = Field(..., max_length=20000, description="Python source code to execute in safe sandbox")
    timeout_seconds: float = Field(0.5, ge=0.05, le=5.0, description="Max execution timeout in seconds")


class TestCaseRequest(BaseModel):
    func_name: str = Field("solution", description="Function name to synthesize test cases for")
    arg_types: Optional[List[str]] = Field(None, description="Optional parameter type hints")


class TemplateRequest(BaseModel):
    pattern: str = Field("binary_search", description="Algorithmic pattern key (e.g. binary_search, sliding_window, dijkstra)")
    language: str = Field("python", description="Target language (python, typescript)")


class RunTestsRequest(BaseModel):
    code: str = Field(..., max_length=20000, description="Python source code containing function implementation")
    tests: List[Dict[str, Any]] = Field(..., description="Array of test cases with 'name', 'call', and 'expected'")
    timeout_per_test: float = Field(0.5, ge=0.05, le=2.0, description="Max timeout per test case")


@router.post("/analyze", summary="Analyze code complexity & detect code smells")
@limiter.limit(EXPENSIVE_ENDPOINT_LIMIT)
async def analyze_code_endpoint(request: Request, req: AnalyzeCodeRequest):
    """Perform AST parsing, Big-O complexity calculation, and code smell analysis."""
    if req.language.lower() != "python":
        pattern_info = code_intelligence_service.classify_algorithmic_pattern(req.code)
        return {
            "valid": True,
            "language": req.language,
            "estimated_time_complexity": pattern_info.get("canonical_time", "O(N)"),
            "estimated_space_complexity": pattern_info.get("canonical_space", "O(1)"),
            "algorithmic_pattern": pattern_info,
            "summary": f"Polyglot code analysis ({req.language}): Pattern={pattern_info['pattern']}",
        }

    analysis = code_intelligence_service.analyze_python_code(req.code)
    return analysis


@router.post("/execute", summary="Execute code safely in an instruction-limited sandbox")
@limiter.limit(EXPENSIVE_ENDPOINT_LIMIT)
async def execute_code_endpoint(request: Request, req: ExecuteCodeRequest):
    """Execute Python code in an instruction-counted (150k steps) isolated sandbox."""
    res = code_intelligence_service.execute_safe_sandbox(req.code, timeout_seconds=req.timeout_seconds)
    return res


@router.post("/test-cases", summary="Synthesize canonical boundary and edge test cases")
async def synthesize_test_cases_endpoint(req: TestCaseRequest):
    """Generate canonical edge test cases for the given function name."""
    cases = code_intelligence_service.synthesize_boundary_test_cases(req.func_name, req.arg_types)
    return {
        "function_name": req.func_name,
        "total_cases": len(cases),
        "cases": cases,
    }


@router.post("/templates", summary="Get production-ready algorithmic template")
async def get_template_endpoint(req: TemplateRequest):
    """Retrieve canonical, type-annotated algorithmic pattern template."""
    tmpl = code_intelligence_service.get_algorithmic_template(req.pattern, req.language)
    return tmpl


@router.post("/run-tests", summary="Run test assertions against code in safe sandbox")
@limiter.limit(EXPENSIVE_ENDPOINT_LIMIT)
async def run_tests_endpoint(request: Request, req: RunTestsRequest):
    """Execute code against test assertions and report pass/fail status."""
    report = code_intelligence_service.run_test_suite(req.code, req.tests, timeout_per_test=req.timeout_per_test)
    return report
