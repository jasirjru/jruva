"""Telemetry, Per-Request Distributed Tracing, and Cost Tracking Service.

Captures end-to-end request diagnostics across all LLM inference pathways:
- Per-request trace generation with unique trace ID
- TTFT (Time-To-First-Token) and end-to-end latency measurement
- Prompt, completion, and total token accounting
- Real-time USD cost estimation across multi-provider pricing models
- Sliding-window latency percentiles (p50, p90, p95, p99)
- Provider availability, failover frequencies, and circuit breaker metrics
"""
import datetime
import math
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# Official / standard API pricing per million tokens ($ USD / 1M tokens)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Google Gemini
    "gemini-2.0-flash": {"prompt_per_m": 0.10, "completion_per_m": 0.40},
    "gemini-3.7-flash": {"prompt_per_m": 0.10, "completion_per_m": 0.40},
    "gemini-1.5-flash": {"prompt_per_m": 0.075, "completion_per_m": 0.30},
    "gemini-1.5-pro": {"prompt_per_m": 1.25, "completion_per_m": 5.00},

    # Groq Hosted Open Weights
    "llama-3.3-70b-versatile": {"prompt_per_m": 0.59, "completion_per_m": 0.79},
    "openai/gpt-oss-120b": {"prompt_per_m": 0.60, "completion_per_m": 0.80},
    "openai/gpt-oss-20b": {"prompt_per_m": 0.10, "completion_per_m": 0.20},
    "groq/compound-mini": {"prompt_per_m": 0.10, "completion_per_m": 0.20},
    "llama-3.1-8b-instant": {"prompt_per_m": 0.05, "completion_per_m": 0.08},

    # Local Ollama & Open Source Free Tier (Zero cloud API cost)
    "ollama": {"prompt_per_m": 0.0, "completion_per_m": 0.0},
    "my-specialist": {"prompt_per_m": 0.0, "completion_per_m": 0.0},
    "liquid/lfm-2.5-2.6b:free": {"prompt_per_m": 0.0, "completion_per_m": 0.0},
    "google/gemma-4-31b-it:free": {"prompt_per_m": 0.0, "completion_per_m": 0.0},
    "inclusionai/ling-3.0-flash-vl:free": {"prompt_per_m": 0.0, "completion_per_m": 0.0},
    "nex-agi/nex-n2.5-pro:free": {"prompt_per_m": 0.0, "completion_per_m": 0.0},

    # Fallback conservative estimate
    "default": {"prompt_per_m": 0.20, "completion_per_m": 0.50},
}


def estimate_token_count(text: str) -> int:
    """Accurately estimate token count for text using standard ~4 chars/token heuristic."""
    if not text:
        return 0
    # Standard rule of thumb: max(words * 1.33, chars / 4)
    char_tokens = math.ceil(len(text) / 4.0)
    word_tokens = math.ceil(len(text.split()) * 1.33)
    return max(1, max(char_tokens, word_tokens))


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for prompt and completion token counts based on pricing registry."""
    pricing = MODEL_PRICING.get(model_name.lower())
    if not pricing:
        if "ollama" in model_name.lower():
            pricing = MODEL_PRICING["ollama"]
        elif "gemini" in model_name.lower():
            pricing = MODEL_PRICING["gemini-2.0-flash"]
        elif "llama" in model_name.lower() or "groq" in model_name.lower():
            pricing = MODEL_PRICING["llama-3.3-70b-versatile"]
        else:
            pricing = MODEL_PRICING["default"]

    prompt_cost = (prompt_tokens / 1_000_000.0) * pricing["prompt_per_m"]
    completion_cost = (completion_tokens / 1_000_000.0) * pricing["completion_per_m"]
    return round(prompt_cost + completion_cost, 6)


def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate p50, p90, p95, p99 percentiles from a list of latencies."""
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}

    s = sorted(latencies)
    n = len(s)

    def p(pct: float) -> float:
        if n == 1:
            return round(s[0], 2)
        k = (n - 1) * pct
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return round(s[f], 2)
        return round(s[f] * (c - k) + s[c] * (k - f), 2)

    return {
        "min": round(s[0], 2),
        "p50": p(0.50),
        "p90": p(0.90),
        "p95": p(0.95),
        "p99": p(0.99),
        "max": round(s[-1], 2),
    }


@dataclass
class RequestTrace:
    """Granular trace metadata for an individual inference request."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    endpoint: str = "/api/v1/chat"
    streaming: bool = False
    requested_model: str = "auto"
    routed_provider: str = "unknown"
    effective_model: str = "unknown"
    tier: str = "B"
    cached: bool = False
    was_failover: bool = False
    failover_reason: Optional[str] = None
    ttft_ms: Optional[float] = None
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    status: str = "success"  # "success", "failover_recovered", "error"
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    task_type: Optional[str] = None
    rag_used: bool = False
    retrieved_chunk_count: int = 0
    rag_reranker_used: bool = False
    rag_degraded: bool = False
    rag_degraded_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryService:
    """Thread-safe in-memory metrics and distributed request tracing engine."""

    def __init__(self, max_traces: int = 500) -> None:
        self._lock = threading.Lock()
        self._max_traces = max_traces
        self._traces: deque[RequestTrace] = deque(maxlen=max_traces)

        # Aggregate counters
        self._total_requests: int = 0
        self._successful_requests: int = 0
        self._failover_requests: int = 0
        self._dropped_requests: int = 0
        self._cache_hits: int = 0

        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0

        # Provider breakdown
        self._provider_request_counts: Dict[str, int] = {"gemini": 0, "groq": 0, "ollama": 0}
        self._provider_token_counts: Dict[str, int] = {"gemini": 0, "groq": 0, "ollama": 0}
        self._provider_cost_usd: Dict[str, float] = {"gemini": 0.0, "groq": 0.0, "ollama": 0.0}

        # Tier breakdown
        self._tier_request_counts: Dict[str, int] = {"B": 0, "A": 0, "S": 0}
        self._tier_cost_usd: Dict[str, float] = {"B": 0.0, "A": 0.0, "S": 0.0}

        # Latency samples for percentile calculations
        self._all_latencies: List[float] = []
        self._ttft_latencies: List[float] = []
        self._failover_latencies: List[float] = []

        # Alertable failure counters by component (Charter Rule 1 compliance)
        self._silent_failure_counts: Dict[str, int] = {
            "rag_retrieval": 0,
            "document_rag": 0,
            "memory_extraction": 0,
            "math_sandbox": 0,
            "code_sandbox": 0,
            "visual_analysis": 0,
            "web_search": 0,
            "router_shadow": 0,
            "session_store": 0,
            "workspace_store": 0,
            "auth_service": 0,
            "image_service": 0,
            "llm_provider": 0,
        }

    def record_silent_failure(self, component: str, reason: str = "") -> None:
        """Record an alertable swallowed/handled exception event into telemetry metrics."""
        with self._lock:
            self._silent_failure_counts[component] = self._silent_failure_counts.get(component, 0) + 1

    def record_trace(self, trace: RequestTrace) -> None:
        """Record a completed request trace and update rolling aggregates."""
        with self._lock:
            self._traces.append(trace)
            self._total_requests += 1

            if trace.status != "error":
                self._successful_requests += 1
            else:
                self._dropped_requests += 1

            if trace.cached:
                self._cache_hits += 1

            # Tier metrics accounting
            t_key = (trace.tier or "B").upper()
            if t_key in self._tier_request_counts:
                self._tier_request_counts[t_key] += 1
                self._tier_cost_usd[t_key] = round(self._tier_cost_usd[t_key] + trace.estimated_cost_usd, 6)
            else:
                self._tier_request_counts[t_key] = 1
                self._tier_cost_usd[t_key] = round(trace.estimated_cost_usd, 6)

            if trace.was_failover:
                self._failover_requests += 1
                if trace.total_latency_ms > 0:
                    self._failover_latencies.append(trace.total_latency_ms)

            # Latency accounting
            if trace.total_latency_ms > 0:
                self._all_latencies.append(trace.total_latency_ms)
                if len(self._all_latencies) > 2000:
                    self._all_latencies = self._all_latencies[-1000:]

            if trace.ttft_ms is not None and trace.ttft_ms > 0:
                self._ttft_latencies.append(trace.ttft_ms)
                if len(self._ttft_latencies) > 2000:
                    self._ttft_latencies = self._ttft_latencies[-1000:]

            # Token and cost accounting
            self._total_prompt_tokens += trace.prompt_tokens
            self._total_completion_tokens += trace.completion_tokens
            self._total_cost_usd = round(self._total_cost_usd + trace.estimated_cost_usd, 6)

            provider = (trace.routed_provider or "unknown").lower()
            if provider in self._provider_request_counts:
                self._provider_request_counts[provider] += 1
                self._provider_token_counts[provider] += trace.total_tokens
                self._provider_cost_usd[provider] = round(
                    self._provider_cost_usd[provider] + trace.estimated_cost_usd, 6
                )
            else:
                self._provider_request_counts[provider] = 1
                self._provider_token_counts[provider] = trace.total_tokens
                self._provider_cost_usd[provider] = round(trace.estimated_cost_usd, 6)

    def get_traces(self, limit: int = 50, provider: Optional[str] = None, failed_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieve recent request traces in reverse chronological order."""
        with self._lock:
            trace_list = list(self._traces)

        if provider:
            trace_list = [t for t in trace_list if t.routed_provider.lower() == provider.lower()]
        if failed_only:
            trace_list = [t for t in trace_list if t.status == "error"]

        return [t.to_dict() for t in reversed(trace_list[-limit:])]

    def get_metrics_summary(self, circuit_breakers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive telemetry metrics dashboard payload."""
        with self._lock:
            total = self._total_requests
            success = self._successful_requests
            failovers = self._failover_requests
            dropped = self._dropped_requests

            availability = round((success / total * 100), 2) if total > 0 else 100.0
            failover_rate = round((failovers / total * 100), 2) if total > 0 else 0.0

            overall_latency_pct = calculate_percentiles(self._all_latencies)
            ttft_latency_pct = calculate_percentiles(self._ttft_latencies)
            failover_latency_pct = calculate_percentiles(self._failover_latencies)

            return {
                "system": {
                    "total_requests": total,
                    "successful_requests": success,
                    "dropped_requests": dropped,
                    "failover_requests": failovers,
                    "availability_pct": availability,
                    "failover_rate_pct": failover_rate,
                },
                "latency_ms": {
                    "total": overall_latency_pct,
                    "ttft": ttft_latency_pct,
                    "failover": failover_latency_pct,
                },
                "tokens_and_cost": {
                    "total_prompt_tokens": self._total_prompt_tokens,
                    "total_completion_tokens": self._total_completion_tokens,
                    "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
                    "accumulated_cost_usd": self._total_cost_usd,
                    "by_provider": {
                        p: {
                            "requests": self._provider_request_counts.get(p, 0),
                            "tokens": self._provider_token_counts.get(p, 0),
                            "cost_usd": self._provider_cost_usd.get(p, 0.0),
                        }
                        for p in set(self._provider_request_counts.keys()).union({"gemini", "groq", "ollama"})
                    },
                },
                "tiers": {
                    "requests_by_tier": dict(self._tier_request_counts),
                    "cost_by_tier": dict(self._tier_cost_usd),
                    "cache_hits": self._cache_hits,
                },
                "silent_failures": {
                    "total": sum(self._silent_failure_counts.values()),
                    "by_component": dict(self._silent_failure_counts),
                },
                "circuits": circuit_breakers or {},
            }

    def reset_metrics(self) -> None:
        """Reset all in-memory telemetry counters and trace history."""
        with self._lock:
            self._traces.clear()
            self._total_requests = 0
            self._successful_requests = 0
            for k in self._silent_failure_counts:
                self._silent_failure_counts[k] = 0
            self._failover_requests = 0
            self._dropped_requests = 0
            self._cache_hits = 0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_cost_usd = 0.0
            self._provider_request_counts = {"gemini": 0, "groq": 0, "ollama": 0}
            self._provider_token_counts = {"gemini": 0, "groq": 0, "ollama": 0}
            self._provider_cost_usd = {"gemini": 0.0, "groq": 0.0, "ollama": 0.0}
            self._tier_request_counts = {"B": 0, "A": 0, "S": 0}
            self._tier_cost_usd = {"B": 0.0, "A": 0.0, "S": 0.0}
            self._all_latencies.clear()
            self._ttft_latencies.clear()
            self._failover_latencies.clear()


# Global singleton instance
telemetry_service = TelemetryService()
