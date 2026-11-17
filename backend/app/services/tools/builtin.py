"""Built-In Safe Tools for Jruva Assistant."""

import ast
import math
import operator
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext

# ─── 1. Safe Calculator Tool (Pure AST Evaluation) ───────────────────────────

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "pi": math.pi,
    "e": math.e,
}


def safe_eval_ast(node: ast.AST) -> float:
    """Evaluate an AST expression node using strictly whitelisted math operators."""
    if isinstance(node, ast.Expression):
        return safe_eval_ast(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Forbidden operator: {op_type.__name__}")
        left = safe_eval_ast(node.left)
        right = safe_eval_ast(node.right)
        if op_type == ast.Pow and right > 1000:
            raise ValueError("Exponent too large (denial-of-service guard)")
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Forbidden unary operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](safe_eval_ast(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name not in SAFE_FUNCTIONS:
                raise ValueError(f"Forbidden function call: '{func_name}'")
            func = SAFE_FUNCTIONS[func_name]
            args = [safe_eval_ast(arg) for arg in node.args]
            return float(func(*args))
        raise ValueError("Dynamic or nested function calls are forbidden")
    elif isinstance(node, ast.Name):
        name = node.id.lower()
        if name in SAFE_FUNCTIONS and isinstance(SAFE_FUNCTIONS[name], (int, float)):
            return float(SAFE_FUNCTIONS[name])
        raise ValueError(f"Forbidden variable reference: '{name}'")
    else:
        raise ValueError(f"Forbidden AST syntax: {type(node).__name__}")


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Mathematical expression (e.g. 'sqrt(144) + 2 * 3.5')")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Safe deterministic math evaluator for arithmetic, roots, logarithms, and trigonometry."
    input_model = CalculatorInput
    timeout_seconds = 2.0

    async def run(self, input_data: CalculatorInput, context: ToolContext) -> Dict[str, Any]:
        expr = input_data.expression.strip()
        if not expr:
            raise ValueError("Empty mathematical expression")
        if len(expr) > 200:
            raise ValueError("Expression exceeds maximum length of 200 characters")

        tree = ast.parse(expr, mode="eval")
        result = safe_eval_ast(tree)
        return {"expression": expr, "result": result}


# ─── 2. Web Retrieval Tool ───────────────────────────────────────────────────

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search engine query string")
    max_results: int = Field(default=3, le=5, ge=1)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Retrieves real-time organic web search snippets via DuckDuckGo without external API keys."
    input_model = WebSearchInput
    timeout_seconds = 8.0

    async def run(self, input_data: WebSearchInput, context: ToolContext) -> Dict[str, Any]:
        from app.services.search.web_search import web_search_service
        citations = await web_search_service.search(input_data.query)
        results = [
            {"title": c.title, "url": c.url, "snippet": c.snippet}
            for c in citations[:input_data.max_results]
        ]
        return {"query": input_data.query, "results": results, "count": len(results)}


# ─── 3. Document Lookup Tool ─────────────────────────────────────────────────

class DocumentLookupInput(BaseModel):
    query: str = Field(..., description="Keyword or topic to search across user's knowledge documents")
    project_id: Optional[str] = None


class DocumentLookupTool(BaseTool):
    name = "document_lookup"
    description = "Searches user-uploaded knowledge library documents with strict tenant isolation."
    input_model = DocumentLookupInput
    timeout_seconds = 5.0

    async def run(self, input_data: DocumentLookupInput, context: ToolContext) -> Dict[str, Any]:
        from app.services.rag.qdrant_vector_store import QdrantVectorStore
        from app.services.workspace_store import workspace_store

        from app.services.tenant_access import active_account
        if not active_account(context.user_id):
            raise PermissionError("Verified account required for document lookup")
        effective_user = context.user_id
        store = QdrantVectorStore()
        chunks = store.search_hybrid(
            "user_knowledge_base",
            query=input_data.query,
            limit=5,
            owner_id=effective_user,
        )
        passages = [
            {
                "id": c.get("id"),
                "file_id": c.get("file_id"),
                "filename": c.get("doc_name"),
                "page": c.get("page", 1),
                "text": c.get("text"),
                "score": round(c.get("rrf_score", c.get("dense_score", 0.0)), 4),
            }
            for c in chunks
        ]
        files = workspace_store.list_library_files(user_id=effective_user, project_id=input_data.project_id)
        query_lower = input_data.query.lower()
        matched_files = [
            {"id": f["id"], "filename": f["filename"], "file_type": f["file_type"], "status": f["status"]}
            for f in files
            if query_lower in f["filename"].lower() or query_lower in f.get("file_type", "").lower()
        ]
        return {
            "user_id": effective_user,
            "query": input_data.query,
            "matched_passages": passages,
            "matched_documents": matched_files,
            "total_passages": len(passages),
        }


# ─── 4. Static Code Intelligence Tool ────────────────────────────────────────

class CodeAnalysisInput(BaseModel):
    code: str = Field(..., description="Python source code to analyze")


class CodeAnalysisTool(BaseTool):
    name = "code_analysis"
    description = "Static AST analysis computing cyclomatic metrics, function signatures, and imports without code execution."
    input_model = CodeAnalysisInput
    timeout_seconds = 3.0

    async def run(self, input_data: CodeAnalysisInput, context: ToolContext) -> Dict[str, Any]:
        code = input_data.code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"syntax_valid": False, "error": str(e)}

        functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        imports: List[str] = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports.extend(alias.name for alias in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                imports.append(n.module)

        return {
            "syntax_valid": True,
            "lines_of_code": len(code.splitlines()),
            "functions": functions,
            "classes": classes,
            "imports": list(set(imports)),
        }
