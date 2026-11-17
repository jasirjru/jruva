"""Centralized Canonical Model Registry for Jruva AI Platform.

Consolidates all cloud LLM model definitions, provider affiliations, fallback paths,
and runtime capabilities to eliminate duplicated strings and stale 404 model IDs.
Local-only dependencies (Ollama, local GPU daemons) are permanently retired in favor
of truthful, hosted cloud AI routing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ProviderName(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    FASTPATH = "fastpath"
    # Legacy markers retained strictly for historical log deserialization
    OLLAMA = "ollama"
    LOCAL_QWEN = "local_qwen"


class ModelPurpose(str, Enum):
    GENERAL_CHAT = "general_chat"
    DEEP_CODING = "deep_coding"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    DOCUMENT_VISION = "document_vision"
    ACADEMIC_RAG = "academic_rag"


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    provider: ProviderName
    purpose: ModelPurpose
    availability: str  # "production", "fallback", "decommissioned"
    fallback_model_id: Optional[str]
    capabilities: List[str] = field(default_factory=list)
    display_name: str = ""
    description: str = ""


# Canonical Registry of Active Supported Cloud Models
MODEL_REGISTRY: Dict[str, ModelDefinition] = {
    # Gemini Tier A Primary
    "gemini-3.6-flash": ModelDefinition(
        model_id="gemini-3.6-flash",
        provider=ProviderName.GEMINI,
        purpose=ModelPurpose.GENERAL_CHAT,
        availability="production",
        fallback_model_id="gemini-3.5-flash-lite",
        capabilities=["chat", "multimodal", "streaming", "long_context"],
        display_name="Gemini 3.6 Flash",
        description="Google AI multimodal model with sub-second TTFT and native 1M context.",
    ),
    "gemini-3.5-flash-lite": ModelDefinition(
        model_id="gemini-3.5-flash-lite",
        provider=ProviderName.GEMINI,
        purpose=ModelPurpose.GENERAL_CHAT,
        availability="production",
        fallback_model_id="openai/gpt-oss-120b",
        capabilities=["chat", "multimodal", "streaming", "high_rate_limit"],
        display_name="Gemini 3.5 Flash Lite",
        description="Google AI lightweight model with high rate limits and fast TTFT.",
    ),
    # Groq Tier A Primary Flagship
    "openai/gpt-oss-120b": ModelDefinition(
        model_id="openai/gpt-oss-120b",
        provider=ProviderName.GROQ,
        purpose=ModelPurpose.GENERAL_CHAT,
        availability="production",
        fallback_model_id="qwen/qwen3.8-27b",
        capabilities=["chat", "reasoning", "streaming", "fast_inference"],
        display_name="GPT-OSS 120B (Groq LPU)",
        description="Ultra-fast flagship LPU inference on Groq Cloud with frontier reasoning.",
    ),
    # Groq Deep Coding & Reasoning Specialist
    "qwen/qwen3.8-27b": ModelDefinition(
        model_id="qwen/qwen3.8-27b",
        provider=ProviderName.GROQ,
        purpose=ModelPurpose.DEEP_CODING,
        availability="production",
        fallback_model_id="openai/gpt-oss-120b",
        capabilities=["coding", "reasoning", "chat", "streaming", "fast_inference"],
        display_name="Qwen 3.8 27B (Groq LPU)",
        description="High-speed coding, refactoring, and algorithmic reasoning on Groq LPU.",
    ),
    # OpenRouter Tier B Fallback
    "google/gemma-4-31b-it:free": ModelDefinition(
        model_id="google/gemma-4-31b-it:free",
        provider=ProviderName.OPENROUTER,
        purpose=ModelPurpose.DEEP_CODING,
        availability="fallback",
        fallback_model_id="openai/gpt-oss-120b",
        capabilities=["coding", "chat", "streaming"],
        display_name="Gemma 4 31B",
        description="Free-tier OpenRouter fallback model (Gemma 4 31B).",
    ),
}


def get_canonical_model(model_request: Optional[str]) -> ModelDefinition:
    """Resolve user or system request string to canonical cloud ModelDefinition.

    Obsolete local-only models (Ollama, local GPU) are smoothly migrated to appropriate
    verified cloud equivalents to maintain full backward compatibility for existing user
    sessions and projects without hanging on local daemons.
    """
    if not model_request:
        return MODEL_REGISTRY["openai/gpt-oss-120b"]

    req = model_request.strip().lower()

    # Exact match in canonical registry
    if req in MODEL_REGISTRY:
        return MODEL_REGISTRY[req]

    # Stale/Legacy Gemini aliases normalized forward to active verified gemini-3.6-flash
    if req in (
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-flash",
        "gemini",
        "gemini-pro",
        "gemini-flash-latest",
        "gemini-3.5-flash",  # retired by Google; requests are served by gemini-3.6-flash
    ):
        return MODEL_REGISTRY["gemini-3.6-flash"]

    # Groq flagship aliases (including decommissioned llama-3.3-70b-versatile)
    if req in (
        "groq",
        "llama",
        "llama3",
        "llama-3.3-70b",
        "llama3.3",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "gpt-oss-120b",
        "gpt-120b",
    ):
        return MODEL_REGISTRY["openai/gpt-oss-120b"]

    # Coding engine aliases (maps truthfully to active Groq Qwen 3.8 27B)
    if req in (
        "qwen",
        "qwen3.8",
        "qwen/qwen3.8-27b",
        "qwen-coder",
        "qwen 2.5 coder",
        "qwen-2.5-coder-32b",
        "openrouter/qwen-2.5-coder-32b",
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "coding",
    ):
        return MODEL_REGISTRY["qwen/qwen3.8-27b"]

    # Decommissioned DeepSeek R1 aliases -> maps safely to Groq flagship reasoning
    if req in (
        "deepseek",
        "deepseek-r1",
        "deepseek r1",
        "groq/deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-llama-70b",
        "reasoning",
    ):
        return MODEL_REGISTRY["openai/gpt-oss-120b"]

    # OpenRouter fallback aliases
    if req in ("liquid", "liquid/lfm-2.5-2.6b:free", "google/gemma-4-31b-it:free", "openrouter"):
        return MODEL_REGISTRY["google/gemma-4-31b-it:free"]

    # Legacy local Ollama & GPU specialist aliases -> gracefully migrate forward to verified cloud models
    if (
        req.startswith("ollama")
        or "specialist" in req
        or req in ("qwen3-8b-local", "local_qwen", "qwen3-local", "qwen3-8b", "local", "my-specialist")
    ):
        if any(tok in req for tok in ("code", "coder", "coding")):
            return MODEL_REGISTRY["qwen/qwen3.8-27b"]
        return MODEL_REGISTRY["openai/gpt-oss-120b"]

    # Safe default: Groq GPT-OSS 120B
    return MODEL_REGISTRY["openai/gpt-oss-120b"]


DECOMMISSIONED_MODELS = {
    "gemini-2.0-flash",
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "deepseek/deepseek-r1:free",
    # Local laptop-only models retired for cloud deployment
    "ollama/jru-specialist:v4",
    "ollama/jru-specialist:v6",
    "ollama/jru-specialist:v5",
    "ollama/jru-specialist:v3",
    "ollama/jru-specialist:v2",
    "jru-specialist:v4",
    "jru-specialist:v6",
    "jru-specialist:v5",
    "jru-specialist:v3",
    "jru-specialist:v2",
    "qwen3-8b-local",
    "my-specialist:latest",
}


def is_decommissioned(model_id: str) -> bool:
    """Return True if model_id is permanently retired/decommissioned and must never be requested upstream."""
    if not model_id:
        return False
    norm = model_id.strip().lower()
    return norm in DECOMMISSIONED_MODELS or any(d in norm for d in ["deepseek-r1-distill", "llama-3.3-70b-versatile", "gemini-2.0"])


def is_local_model(model_id: str) -> bool:
    """Return True if model_id is a deprecated local Ollama or laptop-only model."""
    if not model_id:
        return False
    norm = model_id.strip().lower()
    return (
        norm.startswith("ollama")
        or "specialist" in norm
        or norm in ("qwen3-8b-local", "local_qwen", "qwen3-local", "my-specialist", "local")
    )
