"""Deep Coding Intelligence, AST Static Analysis & Sandboxed Execution Service.

Provides deterministic algorithmic complexity profiling, AST static validation,
boundary test case synthesis, and safe sandboxed execution evaluation.
"""
import ast
import io
import json
import logging
import math
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

SAFE_BUILTINS = {
    "__build_class__": __build_class__,
    "__name__": "__main__",
    "__doc__": None,
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "classmethod": classmethod,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "property": property,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "setattr": setattr,
    "sorted": sorted,
    "staticmethod": staticmethod,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "math": math,
    "json": json,
    # Standard exceptions
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "AttributeError": AttributeError,
    "BaseException": BaseException,
    "Exception": Exception,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "LookupError": LookupError,
    "NameError": NameError,
    "OverflowError": OverflowError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor calculating loops, branches, and recursion to estimate Big-O complexity."""

    def __init__(self):
        self.loop_depth = 0
        self.max_loop_depth = 0
        self.loop_count = 0
        self.branch_count = 0
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.current_function = None
        self.has_recursion = False
        self.has_infinite_loop_risk = False
        self.call_targets: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev_fn = self.current_function
        self.current_function = node.name
        self.functions[node.name] = {
            "name": node.name,
            "args": [a.arg for a in node.args.args],
            "loops": 0,
            "branches": 0,
            "calls": [],
        }
        self.generic_visit(node)
        self.current_function = prev_fn

    def visit_For(self, node: ast.For):
        self.loop_count += 1
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        if self.current_function and self.current_function in self.functions:
            self.functions[self.current_function]["loops"] += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While):
        self.loop_count += 1
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            # while True: without break check
            has_break = any(isinstance(child, ast.Break) for child in ast.walk(node))
            if not has_break:
                self.has_infinite_loop_risk = True
        if self.current_function and self.current_function in self.functions:
            self.functions[self.current_function]["loops"] += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_If(self, node: ast.If):
        self.branch_count += 1
        if self.current_function and self.current_function in self.functions:
            self.functions[self.current_function]["branches"] += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            callee = node.func.id
            self.call_targets.add(callee)
            if self.current_function:
                self.functions[self.current_function]["calls"].append(callee)
                if callee == self.current_function:
                    self.has_recursion = True
        self.generic_visit(node)


class CodeSmellVisitor(ast.NodeVisitor):
    """AST visitor detecting performance hazards, memory antipatterns, and code smells."""

    def __init__(self):
        self.smells: List[Dict[str, Any]] = []
        self.loop_depth = 0

    def visit_For(self, node: ast.For):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Check mutable default argument antipattern (e.g. def f(x=[]):)
        for default in node.args.defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.smells.append({
                    "type": "MutableDefaultArgument",
                    "severity": "Warning",
                    "line": node.lineno,
                    "message": f"Function '{node.name}' uses mutable default argument. Use None and instantiate inside function body.",
                })
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        # Check quadratic string/list accumulation inside loops
        if self.loop_depth > 0 and isinstance(node.op, ast.Add):
            self.smells.append({
                "type": "QuadraticAccumulationInLoop",
                "severity": "Performance Hazard",
                "line": node.lineno,
                "message": "Repeated accumulation (+ / +=) inside loop causes O(N^2) memory reallocation. Use list.append() or ''.join().",
            })
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        # Check linear 'in [list]' search inside loop
        if self.loop_depth > 0:
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.In) and isinstance(comparator, ast.List):
                    self.smells.append({
                        "type": "LinearSearchInLoop",
                        "severity": "Performance Hazard",
                        "line": node.lineno,
                        "message": "Membership test ('in') on literal list inside loop creates O(N*M) quadratic complexity. Convert to set for O(1) lookups.",
                    })
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # Check bare except: clause
        if node.type is None:
            self.smells.append({
                "type": "BareExceptClause",
                "severity": "Code Quality",
                "line": node.lineno,
                "message": "Bare 'except:' intercepts KeyboardInterrupt and SystemExit. Specify 'except Exception:' or concrete errors.",
            })
        self.generic_visit(node)


class CodeIntelligenceService:
    """Service providing deep coding analytics, AST complexity, and sandbox evaluation."""

    def analyze_python_code(self, code: str) -> Dict[str, Any]:
        """Parse code into AST, calculate complexity metrics, and assess algorithmic risks."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {
                "valid": False,
                "syntax_error": {
                    "message": exc.msg,
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "text": exc.text,
                },
                "summary": f"Syntax Error on line {exc.lineno}: {exc.msg}",
            }

        visitor = ComplexityVisitor()
        visitor.visit(tree)

        smell_visitor = CodeSmellVisitor()
        smell_visitor.visit(tree)

        pattern_info = self.classify_algorithmic_pattern(code)

        # Estimate Big-O time complexity
        if visitor.has_recursion:
            if visitor.max_loop_depth > 0:
                estimated_time_complexity = "O(N * log N) or O(N^2) [Recursive Loop]"
            else:
                estimated_time_complexity = "O(log N) to O(2^N) [Recursive Branching]"
        elif visitor.max_loop_depth == 0:
            estimated_time_complexity = "O(1) [Constant Time]"
        elif visitor.max_loop_depth == 1:
            estimated_time_complexity = "O(N) [Linear Time]"
        elif visitor.max_loop_depth == 2:
            estimated_time_complexity = "O(N^2) [Quadratic Time]"
        elif visitor.max_loop_depth == 3:
            estimated_time_complexity = "O(N^3) [Cubic Time]"
        else:
            estimated_time_complexity = f"O(N^{visitor.max_loop_depth}) [Polynomial Time]"

        # Space complexity estimation
        has_list_comp = any(isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp)) for n in ast.walk(tree))
        estimated_space = "O(N) [Auxiliary Memory / Stack]" if (has_list_comp or visitor.has_recursion) else "O(1) [Constant In-Place Memory]"

        # Cyclomatic complexity approximation: M = E - N + 2P ~= branches + loops + 1
        cyclomatic = visitor.branch_count + visitor.loop_count + 1

        functions_info = list(visitor.functions.values())

        summary_lines = [
            "### 💻 Algorithmic Code Analysis\n",
            f"- **Time Complexity Estimate:** `{estimated_time_complexity}`",
            f"- **Space Complexity Estimate:** `{estimated_space}`",
            f"- **Max Nesting Depth:** `{visitor.max_loop_depth}` loops deep",
            f"- **Cyclomatic Complexity:** `{cyclomatic}`",
            f"- **Recursion Detected:** `{'Yes' if visitor.has_recursion else 'No'}`",
            f"- **Infinite Loop Risk:** `{'⚠️ High (unbounded while)' if visitor.has_infinite_loop_risk else 'None'}`",
            f"- **Algorithmic Paradigm:** `{pattern_info['pattern']}` ({pattern_info['description']})"
        ]

        if smell_visitor.smells:
            summary_lines.append("\n**⚠️ Code Smells & Optimization Opportunities:**")
            for s in smell_visitor.smells:
                summary_lines.append(f"- `Line {s['line']}` **{s['type']}**: {s['message']}")

        return {
            "valid": True,
            "estimated_time_complexity": estimated_time_complexity,
            "estimated_space_complexity": estimated_space,
            "max_nesting_depth": visitor.max_loop_depth,
            "total_loops": visitor.loop_count,
            "total_branches": visitor.branch_count,
            "cyclomatic_complexity": cyclomatic,
            "is_recursive": visitor.has_recursion,
            "infinite_loop_risk": visitor.has_infinite_loop_risk,
            "functions": functions_info,
            "code_smells": smell_visitor.smells,
            "algorithmic_pattern": pattern_info,
            "summary": "\n".join(summary_lines),
        }

    def execute_safe_sandbox(self, code: str, timeout_seconds: float = 0.5) -> Dict[str, Any]:
        """Execute Python code in local development mode only if explicitly opted-in.
        
        Arbitrary Python execution is treated as UNSAFE for public production and is
        disabled by default. Containerized OS/Docker isolation is recorded as a future
        architectural roadmap item rather than claiming sandbox isolation.
        """
        env = os.environ.get("ENVIRONMENT", "development").strip().lower()
        allow_unsafe = os.environ.get("ENABLE_UNSAFE_LOCAL_CODE_EXECUTION", "false").strip().lower() in ("true", "1", "yes")

        if env == "production" or not allow_unsafe:
            return {
                "success": False,
                "status": "execution_unavailable",
                "error": "Arbitrary Python code execution is disabled in public production. Containerized isolation is required for safe execution.",
                "stdout": "",
                "execution_time_ms": 0.0,
            }

        # Local development fallback when explicitly configured:
        dangerous_tokens = [
            r"\bos\b", r"\bsys\b", r"\bsubprocess\b", r"\bshutil\b",
            r"\bopen\s*\(", r"\b__import__\b", r"\beval\s*\(", r"\bexec\s*\(",
            r"\bcompile\s*\(", r"\bsocket\b", r"\burllib\b", r"\brequests\b",
        ]
        for pattern in dangerous_tokens:
            if re.search(pattern, code):
                return {
                    "success": False,
                    "status": "security_violation",
                    "error": f"Security Violation: Restricted token matching '{pattern}' is not allowed in sandbox.",
                    "stdout": "",
                    "execution_time_ms": 0.0,
                }

        stdout_buffer = io.StringIO()
        exec_globals = {
            "__builtins__": SAFE_BUILTINS.copy(),
            "__name__": "__main__",
            "__doc__": None,
        }

        result_container: Dict[str, Any] = {
            "success": False,
            "error": None,
            "stdout": "",
            "execution_time_ms": 0.0,
        }

        def target():
            old_stdout = sys.stdout
            sys.stdout = stdout_buffer
            t0 = time.perf_counter()

            # Step and time guard to prevent infinite loops from locking the GIL or thread
            max_steps = 150000
            step_count = 0
            start_wall = time.perf_counter()

            def tracer(frame, event, arg):
                nonlocal step_count
                step_count += 1
                if step_count > max_steps or (time.perf_counter() - start_wall) > timeout_seconds:
                    raise TimeoutError(f"Execution exceeded maximum sandbox limit of {timeout_seconds}s.")
                return tracer

            sys.settrace(tracer)
            try:
                exec(code, exec_globals)
                elapsed = (time.perf_counter() - t0) * 1000.0
                result_container["success"] = True
                result_container["stdout"] = stdout_buffer.getvalue()
                result_container["execution_time_ms"] = round(elapsed, 2)
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                result_container["success"] = False
                result_container["error"] = f"{type(exc).__name__}: {str(exc)}"
                result_container["stdout"] = stdout_buffer.getvalue()
                result_container["execution_time_ms"] = round(elapsed, 2)
            finally:
                sys.settrace(None)
                sys.stdout = old_stdout

        worker = threading.Thread(target=target)
        worker.daemon = True
        worker.start()
        worker.join(timeout=timeout_seconds + 0.5)

        if worker.is_alive():
            result_container["success"] = False
            result_container["error"] = f"TimeoutError: Execution exceeded maximum sandbox limit of {timeout_seconds}s."

        return result_container

    def synthesize_boundary_test_cases(self, func_name: str, arg_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate canonical edge/boundary test cases for competitive programming and algorithms."""
        tests = [
            {"name": "Empty / Null Input", "input": "[] or ''", "purpose": "Checks for ZeroDivisionError or IndexError"},
            {"name": "Single Element", "input": "[42] or 'a'", "purpose": "Checks for loop initialization and bounds"},
            {"name": "All Negative Numbers", "input": "[-5, -12, -1, -99]", "purpose": "Ensures accumulator doesn't assume min = 0"},
            {"name": "Duplicate Elements", "input": "[3, 3, 3, 3]", "purpose": "Checks for set uniqueness and comparison bugs"},
            {"name": "Already Sorted", "input": "[1, 2, 3, 4, 5]", "purpose": "Verifies best-case early exit efficiency"},
            {"name": "Reverse Sorted", "input": "[5, 4, 3, 2, 1]", "purpose": "Verifies worst-case partition and pivot choices"},
            {"name": "Large Value Limits", "input": "[10**9, -(10**9)]", "purpose": "Checks 32/64-bit integer overflow edge cases"},
        ]
        return tests

    def classify_algorithmic_pattern(self, text: str) -> Dict[str, Any]:
        """Classify code or query into canonical algorithmic design patterns."""
        low = text.lower()
        if "sliding window" in low or ("window" in low and ("subarray" in low or "substring" in low)):
            return {
                "pattern": "Sliding Window",
                "description": "Maintains a dynamic window across contiguous elements to optimize subset bounds.",
                "canonical_time": "O(N)",
                "canonical_space": "O(K) or O(1)",
                "common_use_cases": ["Longest Substring Without Repeating Characters", "Minimum Size Subarray Sum", "Max Consecutive Ones"],
            }
        if "two pointer" in low or "two-pointer" in low:
            return {
                "pattern": "Two Pointers",
                "description": "Uses opposing or fast-slow indices to scan linear collections without nested loops.",
                "canonical_time": "O(N)",
                "canonical_space": "O(1)",
                "common_use_cases": ["Two Sum II (Sorted)", "Container With Most Water", "Trapping Rain Water", "3Sum"],
            }
        if "monotonic stack" in low or "next greater" in low or "daily temperature" in low:
            return {
                "pattern": "Monotonic Stack",
                "description": "Maintains monotonic invariant to find nearest larger/smaller elements in linear time.",
                "canonical_time": "O(N)",
                "canonical_space": "O(N)",
                "common_use_cases": ["Next Greater Element", "Largest Rectangle in Histogram", "Daily Temperatures"],
            }
        if "dijkstra" in low or "shortest path" in low:
            return {
                "pattern": "Dijkstra's Algorithm",
                "description": "Greedy priority-queue traversal computing single-source shortest paths on non-negative weighted graphs.",
                "canonical_time": "O((V + E) log V)",
                "canonical_space": "O(V)",
                "common_use_cases": ["Network Delay Time", "Cheapest Flights Within K Stops", "Path with Minimum Effort"],
            }
        if "topological" in low or "kahn" in low:
            return {
                "pattern": "Topological Sort",
                "description": "Linear ordering of vertices in a Directed Acyclic Graph (DAG) using indegree queues.",
                "canonical_time": "O(V + E)",
                "canonical_space": "O(V)",
                "common_use_cases": ["Course Schedule I & II", "Alien Dictionary", "Build Order Dependencies"],
            }
        if "union find" in low or "disjoint set" in low or "dsu" in low:
            return {
                "pattern": "Disjoint Set Union (DSU)",
                "description": "Near constant time connectivity tracking with path compression and union by rank.",
                "canonical_time": "O(α(N)) ≈ O(1)",
                "canonical_space": "O(N)",
                "common_use_cases": ["Number of Provinces", "Redundant Connection", "Kruskal's Minimum Spanning Tree"],
            }
        if "binary search" in low or "bisect" in low or "lower_bound" in low:
            return {
                "pattern": "Binary Search",
                "description": "Halves search space logarithmically over monotonic solution domains.",
                "canonical_time": "O(log N)",
                "canonical_space": "O(1)",
                "common_use_cases": ["Search in Rotated Sorted Array", "Koko Eating Bananas", "Find Peak Element"],
            }
        if "trie" in low or "prefix tree" in low:
            return {
                "pattern": "Trie (Prefix Tree)",
                "description": "Tree data structure for high-throughput string and prefix retrieval.",
                "canonical_time": "O(L) per operation",
                "canonical_space": "O(N * L)",
                "common_use_cases": ["Implement Trie", "Word Search II", "Design Add and Search Words"],
            }
        if "knapsack" in low or "dynamic programming" in low or "memoiz" in low or "tabulation" in low:
            return {
                "pattern": "Dynamic Programming",
                "description": "Decomposes complex problems into overlapping subproblems with optimal substructure.",
                "canonical_time": "O(N * W)",
                "canonical_space": "O(N) or O(N * W)",
                "common_use_cases": ["Coin Change", "Longest Increasing Subsequence", "Edit Distance", "0/1 Knapsack"],
            }

        return {
            "pattern": "General Algorithmic Pattern",
            "description": "Idiomatic structured algorithmic procedure.",
            "canonical_time": "O(N)",
            "canonical_space": "O(1)",
            "common_use_cases": ["Data manipulation", "Mathematical calculations", "State machines"],
        }

    def get_algorithmic_template(self, pattern_key: str, language: str = "python") -> Dict[str, Any]:
        """Return production-ready, type-annotated algorithmic template with Big-O comments."""
        templates: Dict[str, Dict[str, str]] = {
            "binary_search": {
                "python": (
                    "def binary_search(arr: list[int], target: int) -> int:\n"
                    "    \"\"\"Binary search on monotonic array. Time: O(log N), Space: O(1).\"\"\"\n"
                    "    left, right = 0, len(arr) - 1\n"
                    "    while left <= right:\n"
                    "        mid = left + (right - left) // 2\n"
                    "        if arr[mid] == target:\n"
                    "            return mid\n"
                    "        elif arr[mid] < target:\n"
                    "            left = mid + 1\n"
                    "        else:\n"
                    "            right = mid - 1\n"
                    "    return -1\n"
                ),
                "typescript": (
                    "function binarySearch(arr: number[], target: number): number {\n"
                    "  let left = 0, right = arr.length - 1;\n"
                    "  while (left <= right) {\n"
                    "    const mid = left + Math.floor((right - left) / 2);\n"
                    "    if (arr[mid] === target) return mid;\n"
                    "    if (arr[mid] < target) left = mid + 1;\n"
                    "    else right = mid - 1;\n"
                    "  }\n"
                    "  return -1;\n"
                    "}\n"
                ),
            },
            "sliding_window": {
                "python": (
                    "def max_subarray_sum(nums: list[int], k: int) -> int:\n"
                    "    \"\"\"Fixed-size sliding window. Time: O(N), Space: O(1).\"\"\"\n"
                    "    if len(nums) < k: return 0\n"
                    "    window_sum = sum(nums[:k])\n"
                    "    max_sum = window_sum\n"
                    "    for i in range(k, len(nums)):\n"
                    "        window_sum += nums[i] - nums[i - k]\n"
                    "        max_sum = max(max_sum, window_sum)\n"
                    "    return max_sum\n"
                ),
            },
            "dijkstra": {
                "python": (
                    "import heapq\n\n"
                    "def dijkstra(graph: dict[int, list[tuple[int, int]]], start: int) -> dict[int, int]:\n"
                    "    \"\"\"Dijkstra shortest path with min-heap. Time: O((V+E)log V), Space: O(V).\"\"\"\n"
                    "    distances = {node: float('inf') for node in graph}\n"
                    "    distances[start] = 0\n"
                    "    pq = [(0, start)]\n"
                    "    while pq:\n"
                    "        curr_dist, u = heapq.heappop(pq)\n"
                    "        if curr_dist > distances[u]: continue\n"
                    "        for v, weight in graph.get(u, []):\n"
                    "            d = curr_dist + weight\n"
                    "            if d < distances[v]:\n"
                    "                distances[v] = d\n"
                    "                heapq.heappush(pq, (d, v))\n"
                    "    return distances\n"
                ),
            },
            "union_find": {
                "python": (
                    "class UnionFind:\n"
                    "    \"\"\"DSU with path compression and union by rank. Time: O(alpha(N)), Space: O(N).\"\"\"\n"
                    "    def __init__(self, size: int):\n"
                    "        self.parent = list(range(size))\n"
                    "        self.rank = [0] * size\n\n"
                    "    def find(self, x: int) -> int:\n"
                    "        if self.parent[x] != x:\n"
                    "            self.parent[x] = self.find(self.parent[x])\n"
                    "        return self.parent[x]\n\n"
                    "    def union(self, x: int, y: int) -> bool:\n"
                    "        root_x, root_y = self.find(x), self.find(y)\n"
                    "        if root_x == root_y: return False\n"
                    "        if self.rank[root_x] < self.rank[root_y]:\n"
                    "            self.parent[root_x] = root_y\n"
                    "        elif self.rank[root_x] > self.rank[root_y]:\n"
                    "            self.parent[root_y] = root_x\n"
                    "        else:\n"
                    "            self.parent[root_y] = root_x\n"
                    "            self.rank[root_x] += 1\n"
                    "        return True\n"
                ),
            },
        }
        pattern = pattern_key.lower().replace("-", "_").replace(" ", "_")
        lang = language.lower()
        if pattern in templates and lang in templates[pattern]:
            code = templates[pattern][lang]
        elif pattern in templates and "python" in templates[pattern]:
            code = templates[pattern]["python"]
        else:
            code = templates["binary_search"]["python"]
            pattern = "binary_search"

        return {
            "pattern": pattern,
            "language": lang,
            "code": code,
        }

    def run_test_suite(self, code: str, tests: List[Dict[str, Any]], timeout_per_test: float = 0.5) -> Dict[str, Any]:
        """Execute Python code against test assertions and report structured results."""
        passed_count = 0

        # Construct sandbox harness running each test case
        harness = [
            code,
            "\n_test_report = []",
        ]

        for idx, t in enumerate(tests):
            test_name = t.get("name", f"Test_{idx + 1}")
            call_expr = t.get("call") or ""
            expected = t.get("expected")

            # Fallback if call is not provided: use func_name and input
            if not call_expr and "function" in t and "input" in t:
                call_expr = f"{t['function']}(*{t['input']})"

            harness.append(f"""
try:
    _actual = {call_expr}
    _expected = {repr(expected)}
    _passed = (_actual == _expected)
    _test_report.append({{"name": {repr(test_name)}, "passed": _passed, "actual": repr(_actual), "expected": repr(_expected), "error": None}})
except Exception as _e:
    _test_report.append({{"name": {repr(test_name)}, "passed": False, "actual": None, "expected": repr({repr(expected)}), "error": f"{{type(_e).__name__}}: {{str(_e)}}"}})
""")

        harness.append("print('###TEST_JSON###' + json.dumps(_test_report))")
        full_code = "\n".join(harness)

        sandbox_res = self.execute_safe_sandbox(full_code, timeout_seconds=timeout_per_test * max(len(tests), 1))
        if not sandbox_res["success"]:
            return {
                "success": False,
                "error": sandbox_res["error"],
                "total_tests": len(tests),
                "passed_tests": 0,
                "failed_tests": len(tests),
                "results": [{"name": t.get("name", "Test"), "passed": False, "error": sandbox_res["error"]} for t in tests],
                "execution_time_ms": sandbox_res["execution_time_ms"],
            }

        stdout = sandbox_res["stdout"]
        if "###TEST_JSON###" in stdout:
            try:
                json_part = stdout.split("###TEST_JSON###")[1].strip()
                parsed_results = json.loads(json_part)
                passed_count = sum(1 for r in parsed_results if r.get("passed"))
                return {
                    "success": True,
                    "total_tests": len(parsed_results),
                    "passed_tests": passed_count,
                    "failed_tests": len(parsed_results) - passed_count,
                    "results": parsed_results,
                    "execution_time_ms": sandbox_res["execution_time_ms"],
                }
            except Exception as exc:
                logger.warning(f"Failed to parse test report JSON: {exc}")

        return {
            "success": False,
            "error": "Failed to extract test results from sandbox stdout",
            "stdout": stdout,
            "total_tests": len(tests),
            "passed_tests": 0,
            "failed_tests": len(tests),
            "results": [],
            "execution_time_ms": sandbox_res["execution_time_ms"],
        }

    def solve_coding_query(self, query: str, code_snippet: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Automatically detect coding analysis / sandbox query and return structured report."""
        lower = query.lower().strip()

        # Extract code if present in markdown code block or parameter
        code = code_snippet
        if not code:
            code_block_match = re.search(r"```(?:python)?\s*\n?(?P<code>.+?)\n?```", query, flags=re.DOTALL)
            if code_block_match:
                code = code_block_match.group("code").strip()
            else:
                # Check for bare def/class/lambda or multi-line python code
                bare_match = re.search(r"(?:^|\n)(?P<code>(?:def |class |import |from |for |while |if ).+)", query, flags=re.DOTALL)
                if bare_match:
                    code = bare_match.group("code").strip()

        if not code:
            return None

        # Check for complexity analysis request
        is_complexity_query = any(kw in lower for kw in ["time complexity", "space complexity", "big o", "big-o", "complexity of", "analyze this code", "analyze code"])
        is_execute_query = any(kw in lower for kw in ["run this code", "execute this code", "test this code", "run python", "output of"])

        if is_complexity_query or is_execute_query:
            analysis = self.analyze_python_code(code)
            report_sections = [analysis.get("summary", "")]

            if is_execute_query and analysis.get("valid"):
                sandbox_res = self.execute_safe_sandbox(code)
                report_sections.append("### ⚡ Sandbox Execution Result\n")
                if sandbox_res["success"]:
                    report_sections.append(f"```text\n{sandbox_res['stdout'] or '[No stdout output (completed successfully)]'}\n```")
                    report_sections.append(f"*Execution Time:* `{sandbox_res['execution_time_ms']} ms`")
                else:
                    report_sections.append(f"**Execution Status:** ❌ Error\n`{sandbox_res['error']}`")

            # Add boundary test suggestions
            fn_names = [f["name"] for f in analysis.get("functions", [])]
            primary_fn = fn_names[0] if fn_names else "solution"
            test_cases = self.synthesize_boundary_test_cases(primary_fn)
            report_sections.append("### 🧪 Recommended Boundary Test Cases\n")
            for t in test_cases[:4]:
                report_sections.append(f"- **{t['name']}** (`{t['input']}`): {t['purpose']}")

            return {
                "success": True,
                "analysis": analysis,
                "markdown": "\n\n".join(report_sections),
            }

        return None


code_intelligence_service = CodeIntelligenceService()
