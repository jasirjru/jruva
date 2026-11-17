"""Multi-Provider LLM Router with automatic circuit breaking and zero-downtime failover.

Cloud-only production routing architecture:
- Gemini API (Google Cloud AI)
- Groq LPU (GPT-OSS 120B Flagship & Qwen 3.8 27B Deep Coding)
- OpenRouter Free Tier (Gemma 4 31B Fallback)
Eliminates local laptop, Ollama daemon (127.0.0.1:11434), and personal GPU dependencies.
Gracefully migrates legacy saved model settings to verified cloud equivalents.
"""
import asyncio
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from app.core.attachments import is_visual_attachment
from app.core.config import settings
from app.core.exceptions import AllProvidersUnavailableException, LLMProviderException, QuotaExhaustedException
from app.core.model_registry import is_local_model
from app.services.llm.base import BaseLLMProvider
from app.services.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.openai_compat_provider import build_openai_compatible_providers
from app.services.llm.openrouter_provider import OpenRouterProvider
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)


def detect_task_type(prompt: Optional[str]) -> str:
    """Classify user prompt into dedicated cloud task tiers:
    - 'coding': writing, debugging, or reviewing code -> Groq Qwen 3.8 27B
    - 'reasoning': math, complex analysis, multi-step logic, or planning -> Groq GPT-OSS 120B
    - 'documentation': summarizing, drafting, OCR, syllabi, marks cards, documents -> Gemini 3.6 Flash / Groq
    - 'daily': general chat, quick Q&A, greetings, everyday tasks -> Gemini 3.6 Flash
    """
    if not prompt or not prompt.strip():
        return "daily"

    # Isolate user message from appended RAG context or citations if present
    raw = prompt.split("\n\n[")[0].split("\n[OFFICIAL")[0].split("\n[DOCUMENT")[0].strip().lower()
    text = raw or prompt.strip().lower()

    # 0. Image generation intent: Never classify as daily or text LLM
    try:
        from app.services.image.image_service import image_service
        is_img, _ = image_service.detect_intent(text)
        if is_img:
            return "image"
    except Exception:
        pass

    # 1. Coding intent: code blocks, syntax, languages, functions, debugging
    coding_patterns = [
        r"```",
        r"\bdef\s+\w+",
        r"\bclass\s+\w+",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import",
        r"\bfunction\b",
        r"\bconst\b",
        r"\blet\b",
        r"\bvar\b",
        r"\breturn\b",
        r"\bpython\b",
        r"\bjavascript\b",
        r"\btypescript\b",
        r"\breact\b",
        r"\bfastapi\b",
        r"\bsql\b",
        r"\bhtml\b",
        r"\bcss\b",
        r"\bcode\b",
        r"\bcoding\b",
        r"\bdebug\b",
        r"\brefactor\b",
        r"\balgorithm\b",
        r"\bdata structure\b",
        r"\bapi\b",
        r"\bendpoint\b",
        r"\bdatabase\b",
        r"\bquery\b",
        r"\berror\b",
        r"\bexception\b",
        r"\btraceback\b",
        r"\bbug\b",
        r"\bunit test\b",
        r"\bpytest\b",
    ]
    if any(re.search(p, text) for p in coding_patterns):
        return "coding"

    # 2. Documentation & Academic record intent: summaries, drafting, OCR, ordinances, syllabi
    doc_patterns = [
        r"\b(summarize|summary|summaries|tldr|abstract|draft|drafting|minutes of|executive summary|notes)\b",
        r"\b(document|documentation|readme|handbook|ordinance|ordinances|syllabus|syllabi|cbcs|curriculum|regulation|regulations|policy document)\b",
        r"\b(ocr|transcribe|transcription|extract text|extract.*table|tabular|scanned|marks card|grade card|marksheet|marksheets|transcript|transcripts|fee receipt|hall ticket|admit card)\b",
        r"\b(longer-context|tabular reasoning|cgpa|sgpa)\b",
    ]
    if any(re.search(p, text) for p in doc_patterns):
        return "documentation"

    # 3. Reasoning intent: math, proofs, logic puzzles, step-by-step problem solving
    reasoning_patterns = [
        r"\b(math|mathematics|equation|equations|calculate|calculus|derivative|derivatives|integral|integrals|algebra|matrix|matrices|probability|statistics)\b",
        r"\b(theorem|theorems|proof|proofs|prove|proves|proving|solve for x|logic puzzle|multi-step|step-by-step|deep reasoning|formal logic)\b",
        r"\b(in \d+ days|what day will it be|day of the week|modular arithmetic|discount|original price|km/h|speed in km|train travels)\b",
        r"\b(complex analysis|optimization|game theory|combinatorics|discrete math|linear equations?)\b",
        r"\b(visualization planning|architectural trade-off|system design trade-offs|formal verification|uniqueness)\b",
        r"\b(physics|quantum|mechanics|schrodinger|lagrangian|hamiltonian|maxwell|thermodynamics|entropy|carnot|electrodynamics|lorentz|relativity)\b",
        r"\b(chemistry|reaction\s+mechanism|sn1|sn2|electrophilic|nucleophile|gibbs|nernst|stereochemistry|crystal\s+field|chemical\s+kinetics)\b",
        r"\b(biology|dna\s+replication|transcription|translation|okazaki|krebs\s+cycle|cellular\s+respiration|atp\s+synthase|hardy[- ]weinberg|enzyme\s+kinetics|michaelis[- ]menten)\b",
        r"\bprove\b",
        r"\bproof\b",
        r"\bderive\b",
        r"\bderivation\b",
        r"\bevaluate the limit\b",
        r"\bcalculate the probability\b",
        r"\bcalculate\b",
        r"\bcalculation\b",
        r"\blimit\b",
        r"\bformula\b",
        r"\bstep by step\b",
        r"\bthink step by step\b",
        r"\bchain of thought\b",
        r"\bdeep reasoning\b",
        r"\blogical reasoning\b",
        r"\bpuzzle\b",
        r"\briddle\b",
        r"\bgeometry\b",
        r"\bphysics problem\b",
        r"\bwhy does\b",
        r"\bcompare and contrast\b",
        r"\banalyze deeply\b",
    ]
    if any(re.search(p, text) for p in reasoning_patterns):
        return "reasoning"

    # 4. Default: daily general chat
    return "daily"


# Canonical Public Model Catalog for JRUVA AI Platform (Cloud-Only)
AVAILABLE_MODELS: List[Dict[str, Any]] = [
    {
        "id": "auto",
        "name": "Auto",
        "provider": "auto",
        "icon": "shield",
        "badge": "Auto",
        "speed": "Smart Router",
        "description": "Intelligent auto-routing based on prompt complexity across verified cloud tiers.",
        "recommended": True,
        "hidden": False,
        "runtime": "cloud",
        "fallback_capable": True,
    },
    {
        "id": "gemini-3.6-flash",
        "name": "Gemini Flash",
        "provider": "gemini",
        "icon": "sparkles",
        "badge": "Fast",
        "speed": "Ultra-Fast",
        "description": "General chat, quick Q&A, and everyday tasks on Google Cloud.",
        "recommended": True,
        "hidden": False,
        "runtime": "cloud",
        "fallback_capable": True,
    },
    {
        "id": "qwen/qwen3.8-27b",
        "name": "Qwen 3.8",
        "provider": "groq",
        "icon": "code",
        "badge": "Coder",
        "speed": "~300 tok/s",
        "description": "High-throughput coding, refactoring, and algorithmic reasoning on Groq LPU.",
        "recommended": False,
        "hidden": False,
        "runtime": "cloud",
        "fallback_capable": True,
    },
    {
        "id": "groq/deepseek-r1-distill-llama-70b",
        "name": "DeepSeek R1",
        "provider": "groq",
        "icon": "brain",
        "badge": "Unavailable",
        "speed": "Decommissioned",
        "description": "Upstream DeepSeek endpoint decommissioned on Groq.",
        "recommended": False,
        "hidden": True,
        "runtime": "cloud",
        "fallback_capable": False,
        "available": False,
    },
    # Internal models preserved for backwards compatibility and fallback
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS 120B (Groq LPU)",
        "provider": "groq",
        "icon": "zap",
        "badge": "Flagship",
        "speed": "~300 tok/s",
        "description": "Ultra-fast flagship LPU inference on Groq Cloud with frontier reasoning.",
        "recommended": False,
        "hidden": True,
        "runtime": "cloud",
        "fallback_capable": True,
    },
    {
        "id": "google/gemma-4-31b-it:free",
        "name": "Gemma 4 31B",
        "provider": "openrouter",
        "icon": "server",
        "badge": "Free Tier",
        "speed": "Fallback",
        "description": "Free-tier OpenRouter fallback model (Gemma 4 31B).",
        "recommended": False,
        "hidden": True,
        "runtime": "cloud",
        "fallback_capable": False,
    },
]

# Multi-Tier Compute Configuration (All Cloud-Backed)
TIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "S+++": {
        "name": "Tier S+++ (Supreme Frontier)",
        "default_primary": "groq",
        "default_backup": "gemini",
        "primary_model": "openai/gpt-oss-120b",
        "backup_model": "gemini-3.5-flash-lite",
        "max_tokens": 4096,
        "rag_top_k": 12,
        "rate_limit_rpm": 300,
        "target_ttft_ms": 100,
        "thinking_budget": 0,
    },
    "S++": {
        "name": "Tier S++ (Supreme Zero-Lag)",
        "default_primary": "groq",
        "default_backup": "gemini",
        "primary_model": "qwen/qwen3.8-27b",
        "backup_model": "gemini-3.5-flash-lite",
        "max_tokens": 4096,
        "rag_top_k": 8,
        "rate_limit_rpm": 240,
        "target_ttft_ms": 120,
        "thinking_budget": 0,
    },
    "S": {
        "name": "Tier S (Advanced Reasoning & Coding)",
        "default_primary": "groq",
        "default_backup": "gemini",
        "primary_model": "qwen/qwen3.8-27b",
        "backup_model": "gemini-3.5-flash-lite",
        "max_tokens": 2048,
        "rag_top_k": 4,
        "rate_limit_rpm": 120,
        "target_ttft_ms": 250,
    },
    "A": {
        "name": "Tier A (Professional / Dev)",
        "default_primary": "gemini",
        "default_backup": "groq",
        "primary_model": "gemini-3.6-flash",
        "backup_model": "openai/gpt-oss-120b",
        "max_tokens": 4096,
        "rag_top_k": 6,
        "rate_limit_rpm": 60,
        "target_ttft_ms": 500,
    },
    "B": {
        "name": "Tier B (Standard / Free)",
        "default_primary": "groq",
        "default_backup": "gemini",
        "primary_model": "qwen/qwen3.8-27b",
        "backup_model": "gemini-3.5-flash-lite",
        "max_tokens": 2048,
        "rag_top_k": 3,
        "rate_limit_rpm": 15,
        "target_ttft_ms": 300,
    },
    "FT": {
        "name": "Tier FT (Hosted Specialist)",
        "default_primary": "groq",
        "default_backup": "gemini",
        "primary_model": getattr(settings, "FINETUNED_MODEL_ID", "openai/gpt-oss-120b"),
        "backup_model": "gemini-3.5-flash-lite",
        "max_tokens": 4096,
        "rag_top_k": 8,
        "rate_limit_rpm": 120,
        "target_ttft_ms": 200,
        "version": getattr(settings, "ACTIVE_FINETUNED_VERSION", "cloud-v1"),
        "enabled": getattr(settings, "ENABLE_FINETUNED_TIER", False),
    },
}


class GenerationResult(tuple):
    """Backwards-compatible 4-tuple (text, provider, model, was_failover) with rich metadata."""
    def __new__(
        cls,
        text: str,
        provider: str,
        model: str,
        was_failover: bool,
        fallback_reason: Optional[str] = None,
        requested_model: Optional[str] = None,
        requested_model_label: Optional[str] = None,
        actual_model: Optional[str] = None,
        actual_provider: Optional[str] = None,
        fallback_used: Optional[bool] = None,
    ):
        instance = super().__new__(cls, (text, provider, model, was_failover))
        instance.text = text
        instance.provider = provider
        instance.model = model
        instance.was_failover = was_failover
        instance.is_failover = was_failover
        instance.fallback_used = fallback_used if fallback_used is not None else was_failover
        instance.fallback_reason = fallback_reason
        instance.requested_model = requested_model or model
        instance.requested_model_label = requested_model_label or model
        instance.actual_model = actual_model or model
        instance.actual_provider = actual_provider or provider
        return instance


class StreamToken(tuple):
    """Backwards-compatible 4-tuple (token, provider, model, is_failover) with rich metadata."""
    def __new__(
        cls,
        token: str,
        provider: str,
        model: str,
        is_failover: bool,
        fallback_reason: Optional[str] = None,
        requested_model: Optional[str] = None,
        requested_model_label: Optional[str] = None,
        actual_model: Optional[str] = None,
        actual_provider: Optional[str] = None,
        fallback_used: Optional[bool] = None,
    ):
        instance = super().__new__(cls, (token, provider, model, is_failover))
        instance.token = token
        instance.provider = provider
        instance.model = model
        instance.is_failover = is_failover
        instance.was_failover = is_failover
        instance.fallback_used = fallback_used if fallback_used is not None else is_failover
        instance.fallback_reason = fallback_reason
        instance.requested_model = requested_model or model
        instance.requested_model_label = requested_model_label or model
        instance.actual_model = actual_model or model
        instance.actual_provider = actual_provider or provider
        return instance


def get_model_label(model_id: Optional[str]) -> str:
    """Return user-friendly display label for a model identifier."""
    if not model_id or model_id == "auto":
        return "Auto"
    for m in AVAILABLE_MODELS:
        if m.get("id") == model_id:
            return m.get("name", model_id)
    if "gemini" in model_id.lower():
        return "Gemini Flash"
    if "qwen" in model_id.lower():
        return "Qwen 3.8"
    if "gpt-oss" in model_id.lower():
        return "GPT-OSS 120B"
    if is_local_model(model_id):
        return f"{model_id} (Legacy Migrated)"
    if "liquid" in model_id.lower():
        return "Liquid LFM"
    if "gemma-4" in model_id.lower():
        return "Gemma 4 31B"
    return model_id


class RequestBudget:
    """Explicit monotonic time budget enforcing end-to-end request deadlines."""

    def __init__(
        self,
        total_seconds: float = 35.0,
        first_token_target_seconds: float = 8.0,
    ) -> None:
        self.total_seconds = total_seconds
        self.first_token_target_seconds = first_token_target_seconds
        self.start_time = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed_seconds)

    def is_expired(self) -> bool:
        return self.remaining_seconds <= 0.0

    def can_attempt_candidate(self, min_seconds: float = 3.0) -> bool:
        return self.remaining_seconds >= min_seconds

    def timeout_for_candidate(self, default_timeout: float = 8.0) -> float:
        return max(0.0, min(default_timeout, self.remaining_seconds))

    def ensure_window(self, min_seconds: float) -> None:
        """Extend the deadline so at least min_seconds remain (slow setup must not starve the model)."""
        if self.remaining_seconds < min_seconds:
            self.total_seconds = self.elapsed_seconds + min_seconds


class LLMRouter:
    """Intelligent multi-provider routing engine with stateful circuit breaking and failover."""

    def __init__(
        self,
        gemini_provider: Optional[GeminiProvider] = None,
        groq_provider: Optional[GroqProvider] = None,
        ollama_provider: Optional[Any] = None,
        openrouter_provider: Optional[OpenRouterProvider] = None,
        local_qwen_provider: Optional[Any] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        extra_providers: list[BaseLLMProvider] | None = None,
    ) -> None:
        self.gemini_provider = gemini_provider or GeminiProvider()
        self.groq_provider = groq_provider or GroqProvider()
        self.openrouter_provider = openrouter_provider or OpenRouterProvider()
        # Extra free tiers (Mistral, Cloudflare, Cerebras, NVIDIA, SambaNova), each enabled only by its API key env var.
        self.extra_providers: list[BaseLLMProvider] = (
            list(extra_providers) if extra_providers is not None else build_openai_compatible_providers()
        )
        # Local providers are retained strictly as optional mock handles in test suites
        self.ollama_provider = ollama_provider
        self.local_qwen_provider = local_qwen_provider
        self.circuit_config = circuit_config or CircuitBreakerConfig(
            failure_threshold=3,
            cooldown_seconds=15.0,
            canary_success_threshold=2,
        )
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "gemini": CircuitBreaker("gemini", self.circuit_config),
            "groq": CircuitBreaker("groq", self.circuit_config),
            "openrouter": CircuitBreaker("openrouter", self.circuit_config),
        }

    def get_breaker(self, provider_name: str) -> CircuitBreaker:
        """Retrieve or lazily create a CircuitBreaker for a given provider."""
        if provider_name not in self.circuit_breakers:
            self.circuit_breakers[provider_name] = CircuitBreaker(provider_name, self.circuit_config)
        return self.circuit_breakers[provider_name]

    def get_circuit_metrics(self) -> Dict[str, Any]:
        """Return diagnostic metrics and state summaries across all provider circuits."""
        return {name: cb.get_summary() for name, cb in self.circuit_breakers.items()}

    def get_catalog(self, include_hidden: bool = False) -> List[Dict]:
        """Return catalog of verified cloud models with current availability flags and health states.

        Zero laptop connections: Never probes localhost:11434 or local GPU endpoints.
        """
        now = time.time()
        cached = getattr(self, "_full_catalog_cache", None)
        cache_time = getattr(self, "_catalog_cache_time", 0.0)
        if cached is not None and (now - cache_time) < 3.0:
            if not include_hidden:
                return [dict(m) for m in cached if not m.get("hidden", False)]
            return [dict(m) for m in cached]

        full_catalog = []
        gemini_ready = self.gemini_provider.is_available and self.get_breaker("gemini").allow_request()
        groq_ready = self.groq_provider.is_available and self.get_breaker("groq").allow_request()
        openrouter_ready = self.openrouter_provider.is_available and self.get_breaker("openrouter").allow_request()

        from app.services.llm.circuit_breaker import model_health_tracker, HealthState

        for model in AVAILABLE_MODELS:
            entry = dict(model)
            prov = entry["provider"]
            model_id = entry["id"]

            if prov == "gemini":
                h = model_health_tracker.get_health("gemini", model_id)
                entry["available"] = gemini_ready and (h in (HealthState.HEALTHY, HealthState.DEGRADED))
                entry["health"] = h.value
                entry["actual_backend_model"] = model_id
            elif prov == "groq":
                if "deepseek" in model_id.lower() or "llama-3.3" in model_id.lower():
                    entry["available"] = False
                    entry["health"] = HealthState.MISCONFIGURED.value
                    entry["actual_backend_model"] = "openai/gpt-oss-120b"
                else:
                    h = model_health_tracker.get_health("groq", model_id)
                    entry["available"] = groq_ready and (h in (HealthState.HEALTHY, HealthState.DEGRADED))
                    entry["health"] = h.value
                    entry["actual_backend_model"] = model_id
            elif prov == "openrouter":
                h = model_health_tracker.get_health("openrouter", model_id)
                entry["available"] = openrouter_ready and (h in (HealthState.HEALTHY, HealthState.DEGRADED))
                entry["health"] = h.value
                entry["actual_backend_model"] = model_id
            elif prov == "ollama":
                if getattr(self, "ollama_provider", None) and self.ollama_provider.is_available:
                    entry["available"] = self.ollama_provider.is_model_available(model_id)
                else:
                    entry["available"] = False
                entry["health"] = HealthState.HEALTHY.value if entry["available"] else HealthState.DOWN.value
                entry["actual_backend_model"] = model_id
            else:  # "auto"
                any_healthy = gemini_ready or groq_ready or openrouter_ready or any(
                    p.is_available and self.get_breaker(p.provider_name).allow_request()
                    for p in getattr(self, "extra_providers", [])
                )
                entry["available"] = any_healthy
                entry["health"] = HealthState.HEALTHY.value if any_healthy else HealthState.DOWN.value
                entry["actual_backend_model"] = "auto-router"

            full_catalog.append(entry)

        self._full_catalog_cache = full_catalog
        self._catalog_cache_time = now

        if not include_hidden:
            return [dict(m) for m in full_catalog if not m.get("hidden", False)]
        return [dict(m) for m in full_catalog]

    def _resolve_providers(
        self,
        requested_model: Optional[str] = None,
        attachments: Optional[List[Any]] = None,
        tier: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Tuple[BaseLLMProvider, Optional[BaseLLMProvider], str]:
        """Resolve primary and backup cloud providers and targeted model ID.

        Gracefully maps legacy local selections (Ollama, local_qwen) to verified cloud providers.
        """
        req = (requested_model or settings.DEFAULT_PROVIDER or "auto").lower().strip()
        tier_key = tier.upper() if tier and tier.upper() in TIER_CONFIGS else "B"

        gemini_ready = self.gemini_provider.is_available and self.get_breaker("gemini").allow_request()
        groq_ready = self.groq_provider.is_available and self.get_breaker("groq").allow_request()
        openrouter_ready = self.openrouter_provider.is_available and self.get_breaker("openrouter").allow_request()

        # Check for visual / document attachments
        has_visuals = any(is_visual_attachment(att) for att in (attachments or []))

        if has_visuals:
            if groq_ready and self._can_see(self.groq_provider, attachments):
                # Groq's vision model answers images in well under a second with 1,000 free requests/day.
                vision_model = getattr(self.groq_provider, "VISION_MODELS", ["qwen/qwen3.8-27b"])[0]
                return self.groq_provider, (self.gemini_provider if gemini_ready else None), vision_model
            if gemini_ready:
                # Visual attachments require a multimodal provider. Do not silently fallback to text-only models.
                return self.gemini_provider, None, "gemini-3.6-flash"
            else:
                raise LLMProviderException(
                    "Multimodal visual and document analysis requires a vision-capable AI model (Google Gemini). "
                    "The multimodal provider is currently unavailable, and no visual fallback model is active. "
                    "Please retry in a few moments."
                )

        # 1. DIRECT CLOUD MODEL REQUESTS
        # Daily / Gemini family
        if req in ("gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.0-flash", "gemini", "gemini-flash") or req.startswith("gemini"):
            primary = self.gemini_provider
            backup = self.groq_provider if (settings.ENABLE_AUTO_FAILOVER and groq_ready) else self.openrouter_provider
            target_gemini = "gemini-3.5-flash-lite" if req == "gemini-3.5-flash-lite" else "gemini-3.6-flash"
            return primary, backup, target_gemini

        # Legacy local model requests (Ollama, local GPU) -> Smoothly migrate to verified cloud provider
        if is_local_model(req):
            if (
                self.ollama_provider
                and getattr(self.ollama_provider, "is_available", False)
                and (
                    getattr(settings, "ALLOW_LOCAL_DEV_OLLAMA", False)
                    or getattr(self.ollama_provider, "_allow_direct_inference", False)
                    or "mock" in type(self.ollama_provider).__name__.lower()
                )
            ):
                return self.ollama_provider, (self.groq_provider if groq_ready else self.gemini_provider), req

            logger.info(f"[LLMRouter] Obsolete local model '{req}' requested. Gracefully migrating to cloud tier.")
            if any(tok in req for tok in ("code", "coder", "coding", "qwen")):
                target = "qwen/qwen3.8-27b"
                primary = self.groq_provider if groq_ready else (self.openrouter_provider if openrouter_ready else self.gemini_provider)
                backup = self.gemini_provider if (gemini_ready and primary != self.gemini_provider) else None
                return primary, backup, target
            else:
                target = "openai/gpt-oss-120b"
                primary = self.groq_provider if groq_ready else self.gemini_provider
                backup = self.gemini_provider if (gemini_ready and primary != self.gemini_provider) else self.openrouter_provider
                return primary, backup, target

        # Coding Engine (Qwen 3.8 / Groq LPU)
        if req in ("qwen/qwen3.8-27b", "qwen 3.8", "qwen", "qwen-coder", "openrouter/qwen-2.5-coder-32b", "qwen-2.5-coder-32b", "coding") or ("qwen" in req and "coder" in req):
            if groq_ready:
                backup = self.openrouter_provider if openrouter_ready else self.gemini_provider
                return self.groq_provider, backup, "qwen/qwen3.8-27b"
            elif openrouter_ready:
                return self.openrouter_provider, self.gemini_provider, "google/gemma-4-31b-it:free"
            else:
                return self.gemini_provider, None, "gemini-3.6-flash"

        # DeepSeek R1 requests -> Decommissioned upstream! Redirect to Groq flagship reasoning
        if "deepseek" in req or req == "groq/deepseek-r1-distill-llama-70b" or req in ("deep-reasoning", "reasoning"):
            primary = self.groq_provider
            backup = self.gemini_provider if (settings.ENABLE_AUTO_FAILOVER and gemini_ready) else self.openrouter_provider
            return primary, backup, "openai/gpt-oss-120b"

        if req.startswith("openrouter/") or req in ("openrouter", "liquid", "liquid/lfm-2.5-2.6b:free", "google/gemma-4-31b-it:free"):
            primary = self.openrouter_provider
            backup = self.groq_provider if (settings.ENABLE_AUTO_FAILOVER and groq_ready) else self.gemini_provider
            return primary, backup, "google/gemma-4-31b-it:free"

        if (
            req in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound-mini", "groq/compound", "llama-3.3-70b-versatile"]
            or req.startswith("openai/")
            or req.startswith("groq/")
            or "llama" in req
        ):
            primary = self.groq_provider
            backup = self.gemini_provider if (settings.ENABLE_AUTO_FAILOVER and gemini_ready) else None
            return primary, backup, "openai/gpt-oss-120b"

        if req == "groq":
            primary = self.groq_provider
            backup = self.gemini_provider if (settings.ENABLE_AUTO_FAILOVER and gemini_ready) else None
            return primary, backup, getattr(self.groq_provider, "model", "openai/gpt-oss-120b")

        # 2. AUTO MODE: TASK-BASED ROUTING AMONG VERIFIED CLOUD TIERS
        if not prompt:
            if tier_key == "B":
                if groq_ready:
                    return self.groq_provider, (self.gemini_provider if gemini_ready else None), "qwen/qwen3.8-27b"
                elif gemini_ready:
                    return self.gemini_provider, None, "gemini-3.5-flash-lite"
            elif tier_key == "A":
                if gemini_ready:
                    return self.gemini_provider, (self.groq_provider if groq_ready else None), "gemini-3.6-flash"
                elif groq_ready:
                    return self.groq_provider, None, "openai/gpt-oss-120b"
            elif tier_key == "S":
                if groq_ready:
                    return self.groq_provider, (self.gemini_provider if gemini_ready else None), "qwen/qwen3.8-27b"
                elif gemini_ready:
                    return self.gemini_provider, None, "gemini-3.6-flash"
            elif tier_key == "S++":
                if groq_ready:
                    return self.groq_provider, (self.gemini_provider if gemini_ready else None), "qwen/qwen3.8-27b"
                elif gemini_ready:
                    return self.gemini_provider, None, "gemini-3.5-flash-lite"
            elif tier_key == "S+++":
                if groq_ready:
                    return self.groq_provider, (self.gemini_provider if gemini_ready else None), "openai/gpt-oss-120b"
                elif gemini_ready:
                    return self.gemini_provider, None, "gemini-3.5-flash-lite"

        # Detect task type from prompt
        task_type = detect_task_type(prompt)
        logger.debug(f"[LLMRouter] Auto-routing among verified tiers. Prompt task classified as: {task_type}")

        # IMAGE GENERATION -> Route strictly to vision-capable models
        if task_type == "image":
            if gemini_ready:
                return self.gemini_provider, (self.openrouter_provider if openrouter_ready else None), "gemini-3.6-flash"
            elif openrouter_ready:
                return self.openrouter_provider, None, "google/gemma-4-31b-it:free"
            else:
                return self.gemini_provider, None, "gemini-3.6-flash"

        # CODING -> Groq Qwen 3.8 27B
        if task_type == "coding":
            if groq_ready:
                backup = self.openrouter_provider if openrouter_ready else self.gemini_provider
                return self.groq_provider, backup, "qwen/qwen3.8-27b"
            elif openrouter_ready:
                return self.openrouter_provider, self.gemini_provider, "google/gemma-4-31b-it:free"
            else:
                return self.gemini_provider, None, "gemini-3.6-flash"

        # REASONING -> Groq GPT-OSS 120B Flagship
        if task_type == "reasoning":
            if groq_ready:
                backup = self.gemini_provider if gemini_ready else (self.openrouter_provider if openrouter_ready else None)
                return self.groq_provider, backup, "openai/gpt-oss-120b"
            elif gemini_ready:
                return self.gemini_provider, (self.openrouter_provider if openrouter_ready else None), "gemini-3.6-flash"
            else:
                return self.openrouter_provider, None, "google/gemma-4-31b-it:free"

        # DOCUMENTATION -> Gemini 3.6 Flash / Groq
        if task_type == "documentation":
            if gemini_ready:
                backup = self.groq_provider if groq_ready else (self.openrouter_provider if openrouter_ready else None)
                return self.gemini_provider, backup, "gemini-3.6-flash"
            elif groq_ready:
                return self.groq_provider, (self.gemini_provider if gemini_ready else None), "openai/gpt-oss-120b"
            else:
                return self.openrouter_provider, None, "google/gemma-4-31b-it:free"

        # DAILY / GENERAL -> Groq Qwen 3.8 27B first (fastest first token, measured), Gemini 3.6 Flash backup.
        # See PROJECT_CONTEXT.md "Free-Tier Upgrade — Phase 1" and tests/test_routing_speed_first.py.
        if groq_ready:
            backup = self.gemini_provider if gemini_ready else (self.openrouter_provider if openrouter_ready else None)
            return self.groq_provider, backup, "qwen/qwen3.8-27b"
        elif gemini_ready:
            return self.gemini_provider, (self.openrouter_provider if openrouter_ready else None), "gemini-3.6-flash"
        elif openrouter_ready:
            return self.openrouter_provider, (self.gemini_provider if gemini_ready else None), "google/gemma-4-31b-it:free"
        else:
            return self.gemini_provider, None, "gemini-3.6-flash"

    @staticmethod
    def _can_see(provider: BaseLLMProvider, attachments: list[Any] | None) -> bool:
        """Gemini takes every visual attachment; other providers only when they verify these exact attachments."""
        if provider.provider_name == "gemini":
            return True
        supports = getattr(provider, "supports_vision_attachments", None)
        return callable(supports) and supports(attachments) is True

    def _collect_fallback_candidates(
        self,
        primary_name: str,
        backup: Optional[BaseLLMProvider],
    ) -> List[BaseLLMProvider]:
        """Assemble an ordered list of verified cloud fallback providers."""
        candidates: List[BaseLLMProvider] = []
        seen = {primary_name}
        if backup and backup.is_available and backup.provider_name not in seen:
            candidates.append(backup)
            seen.add(backup.provider_name)

        # Order: Groq and Gemini (largest free buckets), then env-enabled OpenAI-compatible tiers,
        # then OpenRouter (50 free requests/day shared across all :free models).
        all_provs = [self.groq_provider, self.gemini_provider, *getattr(self, "extra_providers", []), self.openrouter_provider]
        # In tests with mock providers, cascade to any mock providers configured
        if self.ollama_provider and "mock" in type(self.ollama_provider).__name__.lower():
            all_provs.append(self.ollama_provider)

        for prov in all_provs:
            if prov and prov.is_available and prov.provider_name not in seen:
                candidates.append(prov)
                seen.add(prov.provider_name)

        return candidates

    async def _execute_shadow_call(
        self,
        conversation_id: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        shadow_model_id: Optional[str] = None,
    ) -> None:
        """Execute shadow call asynchronously in the background across cloud models."""
        try:
            from app.services.conversation_log_service import conversation_log_service
            target_model = shadow_model_id or getattr(settings, "SHADOW_MODEL_ID", "openai/gpt-oss-120b")
            shadow_text = ""

            if self.groq_provider and self.groq_provider.is_available:
                try:
                    shadow_text = await self.groq_provider.generate_response(
                        prompt=prompt,
                        system_instruction=system_instruction,
                        history=history,
                    )
                except Exception as groq_err:
                    logger.warning(
                        f"[SilentFailureGuard] [ShadowMode] Groq fallback: {groq_err}",
                        exc_info=True,
                    )
                    telemetry_service.record_silent_failure("router_shadow", str(groq_err))

            if not shadow_text and self.gemini_provider and self.gemini_provider.is_available:
                try:
                    shadow_text = await self.gemini_provider.generate_response(
                        prompt=prompt,
                        system_instruction=system_instruction,
                        history=history,
                    )
                except Exception as gem_err:
                    logger.warning(
                        f"[SilentFailureGuard] [ShadowMode] Gemini fallback: {gem_err}",
                        exc_info=True,
                    )
                    telemetry_service.record_silent_failure("router_shadow", str(gem_err))

            if shadow_text:
                conversation_log_service.record_shadow_evaluation(
                    conversation_id=conversation_id,
                    shadow_model=target_model,
                    shadow_response=shadow_text,
                )
        except Exception as exc:
            logger.warning(f"[SilentFailureGuard] Shadow mode execution failed: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("router_shadow", str(exc))

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        requested_model: Optional[str] = None,
        attachments: Optional[List[Any]] = None,
        tier: Optional[str] = None,
        budget: RequestBudget | None = None,
    ) -> GenerationResult:
        """Generate response synchronously via cloud cascade with zero laptop dependencies."""
        if budget is None:
            task_type = detect_task_type(prompt)
            is_deep = task_type in ("reasoning", "coding", "documentation") or bool(attachments)
            total_sec = 60.0 if is_deep else 25.0
            first_tok_target = 8.0 if is_deep else 4.0
            budget = RequestBudget(total_seconds=total_sec, first_token_target_seconds=first_tok_target)

        primary, backup, model_id = self._resolve_providers(requested_model, attachments, tier=tier, prompt=prompt)
        primary_breaker = self.get_breaker(primary.provider_name)
        req_id = requested_model or "auto"
        req_label = get_model_label(req_id)

        # Detect if requested model was an obsolete local model migrated to cloud
        initial_failover = False
        initial_reason: Optional[str] = None
        if is_local_model(req_id) and primary != getattr(self, "ollama_provider", None):
            initial_failover = True
            initial_reason = "local_model_deprecated"

        # Assemble complete candidate cascade: primary first, followed by fallbacks
        candidates: List[BaseLLMProvider] = []
        if primary_breaker.allow_request():
            candidates.append(primary)
        else:
            logger.warning(f"[LLMRouter] Circuit for {primary.provider_name} is OPEN. Skipping primary.")

        for fb in self._collect_fallback_candidates(primary.provider_name, backup):
            if fb not in candidates:
                candidates.append(fb)

        # For visual attachments, ensure only vision-capable providers can be attempted
        has_visuals = any(is_visual_attachment(att) for att in (attachments or []))
        if has_visuals:
            candidates = [c for c in candidates if self._can_see(c, attachments)]

        last_error: Optional[Exception] = None
        for cand in candidates:
            cand_name = cand.provider_name
            cand_breaker = self.get_breaker(cand_name)
            if not cand_breaker.allow_request():
                logger.warning(f"[LLMRouter] Circuit for {cand_name} is OPEN; skipping.")
                continue

            cand_model_override = model_id if cand == primary else None
            cand_expected_model = cand_model_override or getattr(cand, "model", model_id)

            if not budget.can_attempt_candidate(min_seconds=3.0):
                logger.warning(
                    f"[LLMRouter] Request budget exhausted ({budget.remaining_seconds:.1f}s remaining). "
                    f"Aborting cascade before {cand_name}."
                )
                break

            is_fallback_cand = (cand != primary)
            start_cand = time.perf_counter()
            cand_timeout = budget.timeout_for_candidate(default_timeout=15.0)
            if cand_name in ("ollama", "local_qwen"):
                cand_timeout = (
                    getattr(settings, "LOCAL_LLM_DIRECT_TIMEOUT_SECONDS", 120.0)
                    if requested_model and requested_model.lower().strip() not in ("auto", "") and cand == primary
                    else 15.0
                )

            # Reserve a probe slot only now, right before dispatch, so every
            # reservation is released in the finally below.
            if not cand_breaker.try_acquire():
                logger.warning(f"[LLMRouter] No canary slot free for {cand_name}; skipping.")
                continue

            try:
                try:
                    text = await asyncio.wait_for(
                        cand.generate_response(
                            prompt=prompt,
                            system_instruction=system_instruction,
                            history=history,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            attachments=attachments,
                            model_override=cand_model_override,
                        ),
                        timeout=cand_timeout,
                    )
                except TypeError as te:
                    if "model_override" in str(te):
                        text = await asyncio.wait_for(
                            cand.generate_response(
                                prompt=prompt,
                                system_instruction=system_instruction,
                                history=history,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                attachments=attachments,
                            ),
                            timeout=cand_timeout,
                        )
                    else:
                        raise

                latency = (time.perf_counter() - start_cand) * 1000
                cand_breaker.record_success()

                if is_fallback_cand:
                    primary_breaker.record_failover(latency)

                # Phase 4: Shadow-Mode Background Evaluation (Zero user impact)
                if getattr(settings, "ENABLE_SHADOW_MODE", False):
                    import random
                    if random.random() * 100 < getattr(settings, "SHADOW_TRAFFIC_PCT", 0.0):
                        asyncio.create_task(
                            self._execute_shadow_call(
                                conversation_id="",
                                prompt=prompt,
                                system_instruction=system_instruction,
                                history=history,
                            )
                        )

                actual_model = getattr(cand, "last_active_model", None) or cand_expected_model
                was_failover = initial_failover or is_fallback_cand or (actual_model != model_id)
                failover_reason = initial_reason
                if not failover_reason and was_failover:
                    if is_fallback_cand:
                        failover_reason = (
                            "rate_limited"
                            if isinstance(last_error, QuotaExhaustedException)
                            or (last_error and "429" in str(last_error))
                            else "provider_failover"
                        )
                    elif actual_model != model_id:
                        from app.services.llm.circuit_breaker import model_health_tracker, HealthState
                        health = model_health_tracker.get_health(cand_name, model_id)
                        if health == HealthState.RATE_LIMITED:
                            failover_reason = "rate_limited"
                        else:
                            failover_reason = "model_fallback"

                logger.info(
                    f"[LLMRouter] Success via {cand_name} (model={actual_model}, requested={req_id}, "
                    f"latency={latency:.1f}ms, failover={was_failover})"
                )
                return GenerationResult(
                    text,
                    cand_name,
                    actual_model,
                    was_failover,
                    actual_model=actual_model,
                    actual_provider=cand_name,
                    requested_model=req_id,
                    requested_model_label=req_label,
                    fallback_used=was_failover,
                    fallback_reason=failover_reason,
                )

            except Exception as exc:
                cand_breaker.record_failure(str(exc))
                last_error = exc
                logger.warning(
                    f"[LLMRouter] Candidate {cand_name} failed ({exc}). Cascading to next candidate..."
                )
                continue

            finally:
                cand_breaker.release()

        primary_breaker.record_drop()
        fail_summary = f": {last_error}" if last_error else ""
        raise AllProvidersUnavailableException(
            detail=f"Cascading failure: all available cloud providers failed{fail_summary}. AI capacity is temporarily unavailable. Please try again shortly.",
            circuits=self.get_circuit_metrics(),
            retry_after_seconds=int(self.circuit_config.cooldown_seconds),
        )

    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        requested_model: Optional[str] = None,
        attachments: Optional[List[Any]] = None,
        tier: Optional[str] = None,
        budget: RequestBudget | None = None,
    ) -> AsyncGenerator[Tuple[str, str, str, bool], None]:
        """Stream response chunks via cloud cascade with zero laptop dependencies."""
        if budget is None:
            task_type = detect_task_type(prompt)
            is_deep = task_type in ("reasoning", "coding", "documentation") or bool(attachments)
            total_sec = 60.0 if is_deep else 25.0
            first_tok_target = 8.0 if is_deep else 4.0
            budget = RequestBudget(total_seconds=total_sec, first_token_target_seconds=first_tok_target)

        primary, backup, model_id = self._resolve_providers(requested_model, attachments, tier=tier, prompt=prompt)
        primary_breaker = self.get_breaker(primary.provider_name)
        req_id = requested_model or "auto"
        req_label = get_model_label(req_id)

        initial_failover = False
        initial_reason: Optional[str] = None
        if is_local_model(req_id) and primary != getattr(self, "ollama_provider", None):
            initial_failover = True
            initial_reason = "local_model_deprecated"

        candidates: List[BaseLLMProvider] = []
        if primary_breaker.allow_request():
            candidates.append(primary)
        else:
            logger.warning(f"[LLMRouter] Stream circuit for {primary.provider_name} is OPEN. Skipping primary.")

        for fb in self._collect_fallback_candidates(primary.provider_name, backup):
            if fb not in candidates:
                candidates.append(fb)

        # For visual attachments, ensure only vision-capable providers can be attempted
        has_visuals = any(is_visual_attachment(att) for att in (attachments or []))
        if has_visuals:
            candidates = [c for c in candidates if self._can_see(c, attachments)]

        last_error: Optional[Exception] = None
        for cand in candidates:
            cand_name = cand.provider_name
            cand_breaker = self.get_breaker(cand_name)
            if not cand_breaker.allow_request():
                logger.warning(f"[LLMRouter] Stream circuit for {cand_name} is OPEN; skipping.")
                continue

            cand_model_override = model_id if cand == primary else None
            cand_expected_model = cand_model_override or getattr(cand, "model", model_id)

            if not budget.can_attempt_candidate(min_seconds=3.0):
                logger.warning(
                    f"[LLMRouter] Stream budget exhausted ({budget.remaining_seconds:.1f}s remaining). "
                    f"Aborting failover before {cand_name}."
                )
                break

            is_fallback_cand = (cand != primary)
            start_cand = time.perf_counter()
            first_token_seen = False
            has_yielded = False
            valid_tokens_count = 0
            stream_iter = None  # bound before try so the finally never sees a stale/unbound iterator

            if not cand_breaker.try_acquire():
                logger.warning(f"[LLMRouter] No canary slot free for {cand_name}; skipping stream.")
                continue

            try:
                try:
                    stream_iter = cand.stream_response(
                        prompt=prompt,
                        system_instruction=system_instruction,
                        history=history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        attachments=attachments,
                        model_override=cand_model_override,
                    ).__aiter__()
                except TypeError as te:
                    if "model_override" in str(te):
                        stream_iter = cand.stream_response(
                            prompt=prompt,
                            system_instruction=system_instruction,
                            history=history,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            attachments=attachments,
                        ).__aiter__()
                    else:
                        raise

                first_token_timeout = budget.timeout_for_candidate(default_timeout=budget.first_token_target_seconds)
                try:
                    first_raw_token = await asyncio.wait_for(stream_iter.__anext__(), timeout=first_token_timeout)
                except StopAsyncIteration:
                    raise LLMProviderException(f"Provider {cand_name} returned empty stream.")
                except TimeoutError as to_err:
                    raise LLMProviderException(f"Provider {cand_name} timed out waiting for first token (> {first_token_timeout:.1f}s).") from to_err

                actual_model = getattr(cand, "last_active_model", None) or cand_expected_model
                was_failover = initial_failover or is_fallback_cand or (actual_model != model_id)
                failover_reason = initial_reason
                if not failover_reason and was_failover:
                    if is_fallback_cand:
                        failover_reason = (
                            "rate_limited"
                            if isinstance(last_error, QuotaExhaustedException)
                            or (last_error and "429" in str(last_error))
                            else "provider_failover"
                        )
                    elif actual_model != model_id:
                        from app.services.llm.circuit_breaker import model_health_tracker, HealthState
                        health = model_health_tracker.get_health(cand_name, model_id)
                        if health == HealthState.RATE_LIMITED:
                            failover_reason = "rate_limited"
                        else:
                            failover_reason = "model_fallback"

                # Yield first token only when it has visible text. Leading whitespace-only chunks are dropped so
                # a provider that sends nothing but whitespace still counts as empty and the cascade continues
                # (otherwise the UI showed a blank answer with an error).
                sanitized_first = re.sub(r"</?(?:unstructured|structured)>", "", first_raw_token)
                if sanitized_first.strip():
                    first_token_seen = True
                    if is_fallback_cand:
                        cand_latency = (time.perf_counter() - start_cand) * 1000
                        primary_breaker.record_failover(cand_latency)
                    has_yielded = True
                    valid_tokens_count += len(sanitized_first.strip())
                    current_model = getattr(cand, "last_active_model", None) or actual_model
                    yield StreamToken(
                        sanitized_first,
                        cand_name,
                        current_model,
                        was_failover,
                        fallback_reason=failover_reason,
                        requested_model=req_id,
                        requested_model_label=req_label,
                        actual_model=current_model,
                        actual_provider=cand_name,
                        fallback_used=was_failover,
                    )

                # Fetch and yield subsequent tokens with bounded streaming wait
                while True:
                    if budget.is_expired():
                        logger.warning(
                            f"[LLMRouter] Stream exceeded total request budget ({budget.total_seconds:.0f}s). Raising timeout."
                        )
                        raise LLMProviderException(
                            f"Stream on '{cand_name}' exceeded request deadline ({budget.total_seconds:.0f}s)."
                        )

                    chunk_timeout = budget.timeout_for_candidate(default_timeout=15.0)
                    try:
                        raw_token = await asyncio.wait_for(stream_iter.__anext__(), timeout=chunk_timeout)
                    except StopAsyncIteration:
                        break
                    except TimeoutError as to_err:
                        raise LLMProviderException(
                            f"Provider {cand_name} timed out waiting for stream chunk (> {chunk_timeout:.1f}s)."
                        ) from to_err

                    sanitized_token = re.sub(r"</?(?:unstructured|structured)>", "", raw_token)
                    if not sanitized_token or (not has_yielded and not sanitized_token.strip()):
                        continue

                    if not first_token_seen:
                        first_token_seen = True
                        if is_fallback_cand:
                            cand_latency = (time.perf_counter() - start_cand) * 1000
                            primary_breaker.record_failover(cand_latency)

                    has_yielded = True
                    valid_tokens_count += len(sanitized_token.strip())
                    current_model = getattr(cand, "last_active_model", None) or actual_model

                    yield StreamToken(
                        sanitized_token,
                        cand_name,
                        current_model,
                        was_failover,
                        fallback_reason=failover_reason,
                        requested_model=req_id,
                        requested_model_label=req_label,
                        actual_model=current_model,
                        actual_provider=cand_name,
                        fallback_used=was_failover,
                    )

                if not has_yielded or valid_tokens_count == 0:
                    raise LLMProviderException(f"Provider {cand_name} yielded only empty tags or whitespace.")

                cand_breaker.record_success()
                return

            except Exception as exc:
                cand_breaker.record_failure(str(exc))
                last_error = exc

                # If tokens were already emitted to user, NEVER concatenate a different model (mid-stream corruption guard)
                if has_yielded:
                    primary_breaker.record_drop()
                    logger.error(f"[LLMRouter] Stream failure mid-emission on {cand_name}: {exc}")
                    raise AllProvidersUnavailableException(
                        detail=f"Stream error on '{cand_name}': {exc}",
                        circuits=self.get_circuit_metrics(),
                        retry_after_seconds=int(self.circuit_config.cooldown_seconds),
                    )

                logger.warning(
                    f"[LLMRouter] Stream provider {cand_name} failed before emission ({exc}). Cascading to next candidate..."
                )
                continue

            finally:
                # Runs on success, failure and cancellation (GeneratorExit), so the
                # HALF_OPEN probe slot is never leaked.
                cand_breaker.release()
                if stream_iter is not None and hasattr(stream_iter, "aclose"):
                    try:
                        await stream_iter.aclose()
                    except Exception as close_err:
                        logger.debug(f"[LLMRouter] Provider stream cleanup error on {cand_name}: {close_err}")

        primary_breaker.record_drop()
        fail_summary = f": {last_error}" if last_error else ""
        detail_msg = (
            f"Request timed out after {budget.total_seconds:.0f}s. Please retry with a simpler query."
            if budget.is_expired()
            else f"Stream cascade failure: all available providers failed{fail_summary}. AI capacity is temporarily unavailable. Please try again shortly."
        )
        raise AllProvidersUnavailableException(
            detail=detail_msg,
            circuits=self.get_circuit_metrics(),
            retry_after_seconds=int(self.circuit_config.cooldown_seconds),
        )
