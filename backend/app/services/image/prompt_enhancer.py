"""LLM-Driven Dynamic Prompt Expansion Engine for FLUX.1 & SOTA Vision Synthesis.
Transforms brief user concepts into master-level photographic blueprints with explicit
anatomical layout, camera optics, atmospheric lighting, and material textures.
"""
import re
import time
import logging
import asyncio
from typing import Dict, Optional
from collections import OrderedDict
import threading

from app.core.config import settings
from app.services.llm.groq_provider import GroqProvider

logger = logging.getLogger(__name__)

STYLE_DIRECTIVES: Dict[str, str] = {
    "photorealistic": (
        "Style: Ultra-photorealistic documentary photography. 85mm f/1.4 prime lens, shallow depth of field, "
        "subsurface skin scattering, natural pores, soft ambient daylight, Hasselblad 100MP clarity, no airbrushing."
    ),
    "cinematic": (
        "Style: High-budget cinematic film still. Shot on Arri Alexa 65mm anamorphic lens, moody chiaroscuro lighting, "
        "volumetric haze, subtle film grain, dramatic atmospheric color grading, 8k resolution."
    ),
    "architecture": (
        "Style: Architectural Digest luxury exterior/interior photography. Wide 24mm tilt-shift lens, crisp geometric lines, "
        "natural golden hour sunlight, polished concrete, sustainable glass reflections, landscaped modernism."
    ),
    "cyberpunk": (
        "Style: High-tech futuristic cyberpunk realism. Wet reflective asphalt, neon volumetric lighting in cyan and magenta, "
        "subtle holographic overlays, micro-electronic details, cinematic dystopian atmosphere."
    ),
    "anime": (
        "Style: High-end Japanese theatrical animation still. Makoto Shinkai aesthetic, luminous painted clouds, "
        "soft cel-shading, vibrant chromatic illumination, intricate hand-drawn linework."
    ),
}


class PromptExpansionCache:
    """Thread-safe LRU cache for instant sub-millisecond prompt expansions."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 7200):
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        with self._lock:
            if key not in self._cache:
                return None
            item = self._cache[key]
            if now > item["expires_at"]:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return item["expanded"]

    def set(self, key: str, expanded: str) -> None:
        now = time.time()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = {
                "expanded": expanded,
                "expires_at": now + self._ttl,
            }


class LLMPromptEnhancer:
    """AI-powered prompt engineer that decomposes human prompts into FLUX.1 master blueprints."""

    def __init__(self):
        self._cache = PromptExpansionCache()
        self._groq_provider: Optional[GroqProvider] = None

    def _get_provider(self) -> GroqProvider:
        if self._groq_provider is None:
            self._groq_provider = GroqProvider()
        return self._groq_provider

    async def expand(
        self,
        raw_prompt: str,
        style: str = "photorealistic",
        timeout_sec: float = 2.2,
    ) -> str:
        """Expand user concept into a cinema-grade diffusion prompt within strict timeout bounds."""
        clean = re.sub(r"[\r\n\t]+", " ", raw_prompt).strip()
        clean = re.sub(r'["\']', "", clean)

        # If prompt is already an elaborate master prompt (>30 words), keep as is
        if len(clean.split()) > 30:
            return clean

        normalized_style = style.lower().strip() if style else "photorealistic"
        cache_key = f"{clean.lower()}:{normalized_style}"

        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"[PromptEnhancer] Cache HIT for '{clean[:30]}'")
            return cached

        style_instruction = STYLE_DIRECTIVES.get(
            normalized_style, STYLE_DIRECTIVES["photorealistic"]
        )

        system_instruction = (
            "You are an expert generative AI prompt engineer for FLUX.1 and Midjourney v6 photorealistic synthesis. "
            "Your task: Translate the user's concept into a cinema-grade, highly detailed visual prompt. "
            "CRITICAL RULES:\n"
            "1. If the user specifies an unusual anatomical feature or count (e.g., '4 eyes', '3 arms', 'alien face'), "
            "explicitly describe the exact physical layout, alignment, and spatial position on the body/face.\n"
            "2. Integrate camera optics (focal length, depth of field), atmospheric lighting (god rays, chiaroscuro), "
            "and material textures (skin pores, reflections, fabric).\n"
            f"3. Strictly adhere to this visual treatment: {style_instruction}\n"
            "4. Keep the output under 75 words. Be dense with visual information.\n"
            "5. Output ONLY the raw prompt. Never include introductory text, explanations, or quotes."
        )

        try:
            provider = self._get_provider()
            expanded = await asyncio.wait_for(
                provider.generate_response(
                    prompt=f"Subject: {clean}",
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_tokens=140,
                ),
                timeout=timeout_sec,
            )

            # Sanitize LLM output
            cleaned_result = re.sub(r'^(?:Prompt|Expanded Prompt|Here is the prompt):\s*', '', expanded.strip(), flags=re.IGNORECASE)
            cleaned_result = re.sub(r'^["\']+|["\']+$', '', cleaned_result).strip()
            # Normalize typography to standard ASCII
            cleaned_result = (
                cleaned_result
                .replace('\u2011', '-')
                .replace('\u2012', '-')
                .replace('\u2013', '-')
                .replace('\u2014', '-')
                .replace('\u2018', "'")
                .replace('\u2019', "'")
                .replace('\u201c', '"')
                .replace('\u201d', '"')
            )

            if len(cleaned_result) > 15:
                self._cache.set(cache_key, cleaned_result)
                logger.info(f"[PromptEnhancer] Successfully expanded '{clean[:25]}' -> '{cleaned_result[:50]}...'")
                return cleaned_result

        except asyncio.TimeoutError:
            logger.warning(f"[PromptEnhancer] Expansion timed out after {timeout_sec}s for '{clean[:30]}'. Using rule-based fallback.")
        except Exception as exc:
            logger.warning(f"[PromptEnhancer] Expansion error: {exc}. Using rule-based fallback.")

        # High-fidelity rule-based fallback if LLM is unreachable
        return self._rule_based_fallback(clean, normalized_style)

    def _rule_based_fallback(self, clean_subject: str, style: str) -> str:
        """Fast, robust deterministic fallback when LLM is offline or times out."""
        lower = clean_subject.lower()

        # Handle explicit counting or anatomical variations
        if "4 eyes" in lower or "four eyes" in lower:
            return (
                f"Surreal cinematic portrait of a humanoid with four distinct eyes, two pairs of luminous eyes "
                f"vertically aligned across cheekbones and brow, hyper-detailed skin pores, chiaroscuro lighting, "
                f"85mm portrait lens, 8k resolution, crisp photorealism"
            )

        if style == "cinematic":
            return f"{clean_subject}, cinematic film still, shot on 65mm Arri Alexa, anamorphic lens, volumetric lighting, 8k"
        elif style == "architecture":
            return f"{clean_subject}, architectural digest photography, clean structural lines, modern lighting, 8k resolution"
        elif style == "cyberpunk":
            return f"{clean_subject}, futuristic cyberpunk street scene, neon reflections, volumetric haze, sharp 8k detail"
        elif style == "anime":
            return f"{clean_subject}, Makoto Shinkai anime aesthetic, luminous lighting, vibrant clouds, detailed art"
        else:
            return (
                f"{clean_subject}, natural studio lighting, 85mm portrait lens, shallow depth of field, "
                f"hyper-realistic skin texture, 8k resolution, photorealistic"
            )


prompt_enhancer = LLMPromptEnhancer()
