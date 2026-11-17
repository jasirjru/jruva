"""Image Generation Service using FLUX.1 & SDXL.
Includes multi-attempt retry logic, prompt sanitization, and backend response validation.
"""
import os
import json
import re
import urllib.parse
import random
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
import httpx

from app.services.image.prompt_enhancer import prompt_enhancer
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "static"
)

# Strict NSFW / Adult / Inappropriate Imagery Deny-List Patterns
INAPPROPRIATE_IMAGE_PATTERNS = [
    r"(?:\b18\s*\+|\b(?:nsfw|nude|nudity|naked|strip|stripper|erotic|porn|porno|pornography|xxx|hentai|sex|sexy|sexual|boobs?|breasts?|tits?|dick|penis|vagina|pussy|ass|butt|lingerie|bikini|underwear|thong|topless|bottomless|fetish|seductive|lewd)\b)",
    r"\b(?:blood|gore|mutilat\w+|decapitat\w+|kill\w+|murder\w+|suicide|corpse|slaughter)\b",
]
COMPILED_SAFETY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INAPPROPRIATE_IMAGE_PATTERNS]


def check_image_safety(prompt: str) -> Tuple[bool, Optional[str]]:
    """Strictly evaluate prompt for 18+, adult, suggestive, or inappropriate content.
    Returns (is_safe, error_reason).
    """
    clean_prompt = prompt.strip().lower()
    for pattern in COMPILED_SAFETY_PATTERNS:
        if pattern.search(clean_prompt):
            return False, "⚠️ Content Policy Violation: Adult, 18+, suggestive, or inappropriate imagery is strictly prohibited."
    return True, None


SHOWCASE_PRESETS = [
    {
        "keywords": ("library", "central library", "research library", "study", "studying", "reading", "textbook", "books", "bookshelf", "study hall"),
        "filename": "library.jpg",
        "title": "University Central Research Library",
        "model": "Verified Official Asset",
        "description": "University Central Research Library with academic reference collections.",
    },
    {
        "keywords": ("robot", "robotics", "humanoid", "ai lab", "engineering lab", "tech lab", "automation"),
        "filename": "robotics.jpg",
        "title": "Autonomous Robotics & AI Engineering Lab",
        "model": "Verified Official Asset",
        "description": "Autonomous Robotics & AI Engineering Laboratory.",
    },
    {
        "keywords": ("moot", "moot court", "court", "courtroom", "law", "legal", "judge", "justice", "bench", "chamber"),
        "filename": "moot_court.jpg",
        "title": "Smart Law Moot Court & Legal Innovation Center",
        "model": "Verified Official Asset",
        "description": "Smart Law Moot Court for legal proceedings and judicial training.",
    },
    {
        "keywords": ("pharmacy", "pharmaceutical", "chemistry", "biotech", "biotechnology", "medicine", "formulation", "drug lab"),
        "filename": "pharmacy.jpg",
        "title": "Pharmaceutical Formulation & Biotech Lab",
        "model": "Verified Official Asset",
        "description": "Advanced Pharmaceutical Formulation and Biotechnology Research Laboratory.",
    },
    {
        "keywords": ("auditorium", "convocation", "stage", "theater", "theatre", "hall", "conference", "ceremony"),
        "filename": "auditorium.jpg",
        "title": "Grand University Convocation & Tech Auditorium",
        "model": "Verified Official Asset",
        "description": "University Convocation Auditorium with high-capacity seating and staging.",
    },
    {
        "keywords": ("campus", "courtyard", "grounds", "main building", "academic building", "architecture", "institute", "jru"),
        "filename": "campus.jpg",
        "title": "University Campus Grounds & Main Building",
        "model": "Verified Official Asset",
        "description": "University campus grounds, architecture, and academic infrastructure.",
    },
]

IMAGE_PATTERNS = [
    # Explicit commands: /imagine, /image, /draw, /paint, /render
    r"^\s*/(?:imagine|image|draw|paint|render)\s+(?P<subject>.+)",
    # "image: ...", "draw: ...", "photo: ...", "render: ..."
    r"^\s*(?:image|draw|photo|render|paint)\s*:\s*(?P<subject>.+)",
    # Direct drawing/painting verbs: "draw me a ...", "sketch a ...", "paint ...", "render ..."
    r"^\s*(?:(?:please|kindly|can\s+you|could\s+you)\s+)?(?:draw|paint|sketch|render|illustrate)(?:\s+me)?\s+(?:an?\s+)?(?P<subject>.+)",
    # Generate/create/make/show/give + image/picture/photo... + of/for/about + subject (with generous typo tolerance on generate: gen[a-z]{3,7}e?)
    r"^\s*(?:(?:please|kindly|can\s+you|could\s+you|would\s+you|i\s+want|i\s+need|give(?:\s+me)?|show(?:\s+me)?)\s+)?(?:gen[a-z]{3,7}e?|create|make|render|produce|show(?:\s+me)?|give(?:\s+me)?|display)\s+(?:an?\s+)?(?:(?P<style>photorealistic|photo-realistic|realistic|hyperrealistic|cinematic|high-quality|high-res|hd|4k|8k|3d|isometric|scenic|beautiful|stunning)\s+)?(?:image|imaeg|picture|pic|photo\b|photograph\b|portrait|illustration|drawing|artwork|render(?:ing)?|painting|wallpaper|avatar|logo)\s+(?:of|for|about)?\s*(?P<subject>.+)",
    # Generate/create/make/render + visual style descriptor (WITHOUT mandatory image noun): e.g. "Generate a photorealistic red Ferrari parked beside a beach at sunset"
    r"^\s*(?:(?:please|kindly|can\s+you|could\s+you|would\s+you|i\s+want|i\s+need|give(?:\s+me)?|show(?:\s+me)?)\s+)?(?:gen[a-z]{3,7}e?|create|make|render|produce|display)\s+(?:an?\s+)?(?P<style>photorealistic|photo-realistic|realistic|hyperrealistic|cinematic|high-quality|high-res|hd|4k|8k|3d|isometric|scenic|oil\s+painting|watercolor|anime|manga|digital\s+art|concept\s+art)\s+(?:(?:render(?:ing)?|image|picture|photo|illustration|drawing|painting)\s+(?:of|for|about)?\s*)?(?P<subject>.+)",
    # Standalone visual noun: '(an) image/picture/photo... of <anything>'
    r"^\s*(?:an?\s+)?(?:(?P<style>photorealistic|photo-realistic|realistic|hyperrealistic|cinematic|high-quality|high-res|hd|4k|8k|3d|isometric)\s+)?(?:image|imaeg|picture|pic|photo\b|photograph\b|portrait|illustration|drawing|painting|wallpaper|avatar)\s+(?:of|for|about)\s+(?P<subject>.+)",
    # "generate a [anything] image/picture/render/photo"
    r"^\s*(?:(?:please|kindly|can\s+you|could\s+you)\s+)?(?:gen[a-z]{3,7}e?|create|make|render)\s+(?P<subject>.+?)\s+(?:image|imaeg|picture|pic|photo|render|wallpaper|drawing|painting|artwork)\s*$",
    # Direct generation of visual objects/scenes with style/quality adjectives
    r"^\s*(?:(?:please|kindly|can\s+you|could\s+you)\s+)?(?:gen[a-z]{3,7}e?|create|make|render|paint|draw)\s+(?:an?\s+)?(?P<subject>(?:(?:photorealistic|photo-realistic|realistic|hyperrealistic|cinematic|good|beautiful|stunning|scenic)\s+)?(?:ferrari|porsche|lamborghini|bmw|mercedes|bugatti|mclaren|car|supercar|vehicle|automobile|bike|motorcycle|bicycle|spaceship|castle|dragon|galaxy|landscape|room|building|robot|creature|monster|superhero|warrior|sunset|sunrise|beach|mountain|ocean|forest|scenery|character|avatar|planet|nebula|cyberpunk|isometric|earphone|headphones|headphone|earbuds|cat|dog|tiger|lion|animal|portrait).+)",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in IMAGE_PATTERNS]

NON_IMAGE_TEXT_KEYWORDS = {
    "code", "python", "javascript", "typescript", "html", "css", "script",
    "essay", "story", "poem", "summary", "text", "response", "spreadsheet", "csv",
    "email", "letter", "function", "class", "query", "sql", "regex",
    "database", "api", "schema", "json", "yaml", "test", "tests",
    "unit test", "component", "algorithm", "solution", "explanation", "analysis"
}

_LEADING_GEN_VERB = re.compile(
    r"^\s*(?:(?:please|kindly|can\s+you|could\s+you|would\s+you)\s+)?"
    r"(?:generate|draw|create|make|geneate|genreate|genrate|render|paint)\b"
)


def _has_word(text: str, words) -> bool:
    """True when any of `words` (single words or phrases) appears as whole words in `text`."""
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


# ─── Multi-Tier Image Generation Reliability System ──────────────────────────

import time
import hashlib
from collections import OrderedDict
import threading


class ImagePromptCache:
    """Thread-safe in-memory LRU prompt cache for sub-15ms identical prompt resolution."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _hash_key(self, prompt: str, width: int, height: int) -> str:
        raw = f"{prompt.lower().strip()}:{width}x{height}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, prompt: str, width: int, height: int) -> Optional[Dict[str, Any]]:
        key = self._hash_key(prompt, width, height)
        now = time.time()
        with self._lock:
            if key not in self._cache:
                return None
            item = self._cache[key]
            if now > item["expires_at"]:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return item["data"]

    def set(self, prompt: str, width: int, height: int, data: Dict[str, Any]) -> None:
        key = self._hash_key(prompt, width, height)
        now = time.time()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = {
                "data": data,
                "expires_at": now + self._ttl,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class ProviderHealthMonitor:
    """Tracks latency, consecutive failures, and degradation across image generation endpoints."""

    def __init__(self):
        self._lock = threading.Lock()
        self._providers = {
            "flux": {"healthy": True, "consecutive_failures": 0, "last_checked": 0.0, "latency_ms": 0},
            "turbo": {"healthy": True, "consecutive_failures": 0, "last_checked": 0.0, "latency_ms": 0},
            "sdxl": {"healthy": True, "consecutive_failures": 0, "last_checked": 0.0, "latency_ms": 0},
        }

    def is_healthy(self, provider: str) -> bool:
        with self._lock:
            p = self._providers.get(provider)
            return p["healthy"] if p else True

    def record_success(self, provider: str, latency_ms: int = 0):
        with self._lock:
            if provider in self._providers:
                self._providers[provider]["healthy"] = True
                self._providers[provider]["consecutive_failures"] = 0
                self._providers[provider]["last_checked"] = time.time()
                self._providers[provider]["latency_ms"] = latency_ms

    def record_failure(self, provider: str, status_code: int = 0, reason: str = ""):
        with self._lock:
            if provider in self._providers:
                self._providers[provider]["consecutive_failures"] += 1
                self._providers[provider]["last_checked"] = time.time()
                # If 3 consecutive failures, mark degraded
                if self._providers[provider]["consecutive_failures"] >= 3:
                    self._providers[provider]["healthy"] = False
                    logger.warning(
                        f"[ProviderHealth] Provider '{provider}' marked DEGRADED after "
                        f"{self._providers[provider]['consecutive_failures']} consecutive errors (status={status_code}, reason={reason})"
                    )


# Global singletons
_prompt_cache = ImagePromptCache()
_health_monitor = ProviderHealthMonitor()
_concurrency_semaphore = asyncio.Semaphore(8)  # S-tier burst concurrency guard


class ImageGenerationService:
    """Service to detect, enhance, validate, and synthesize FLUX / SDXL images across reliability tiers."""

    def __init__(self):
        self.cache = _prompt_cache
        self.health = _health_monitor
        self.semaphore = _concurrency_semaphore

    def _check_showcase_presets(
        self, prompt: str, enhanced_prompt: str, seed: int, tier: str
    ) -> Optional[Dict[str, Any]]:
        """Check if query matches pre-rendered ultra-high-fidelity showcase assets for instant zero-latency demo."""
        lower = prompt.lower()
        best_preset = None
        best_score = 0

        for preset in SHOWCASE_PRESETS:
            score = sum(1 for kw in preset["keywords"] if kw in lower)
            if score > best_score:
                file_path = os.path.join(STATIC_DIR, "demo", preset["filename"])
                if os.path.exists(file_path):
                    best_score = score
                    best_preset = preset

        if best_preset and best_score > 0:
            logger.info(f"[ImageService] Showcase MATCH -> {best_preset['filename']} (score={best_score})")
            return {
                "success": True,
                "image_url": f"/static/demo/{best_preset['filename']}",
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "model": best_preset["model"],
                "title": best_preset.get("title", "Verified University Asset"),
                "description": best_preset.get("description", ""),
                "seed": seed,
                "width": 1024,
                "height": 1024,
                "tier": tier,
                "cached": True,
                "failover": False,
            }
        return None

    async def _generate_via_huggingface(self, *args, **kwargs):
        """Disabled: account free credits and model deployment are unverified."""
        return None

    def detect_intent(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if user prompt is requesting image synthesis and extract the core subject."""
        stripped = text.strip()
        for pattern in COMPILED_PATTERNS:
            match = pattern.match(stripped)
            if match:
                subject = match.group("subject").strip()
                style = match.groupdict().get("style")
                if style and style.lower() not in subject.lower():
                    subject = f"{style} {subject}".strip()
                subject = re.sub(r"[.!?]+$", "", subject)
                lower_sub = subject.lower()
                # Ignore if text generation keywords are requested (e.g. "generate python code")
                if any(w in lower_sub.split() for w in NON_IMAGE_TEXT_KEYWORDS):
                    continue
                if len(subject) >= 2:
                    return True, subject

        lower = stripped.lower()
        # Fallback: check if words clearly imply image generation
        has_gen_verb = bool(re.search(r"\b(?:gen[a-z]{3,7}e?|draw|create|make|render|paint|sketch|illustrate)\b", lower))
        has_img_noun = bool(re.search(r"\b(?:image|imaeg|picture|pic|photo|wallpaper|portrait|illustration|drawing)\b", lower))
        has_text_noun = any(w in lower.split() for w in NON_IMAGE_TEXT_KEYWORDS)
        if has_gen_verb and has_img_noun and not has_text_noun:
            cleaned = re.sub(
                r"^\s*(?:(?:please|kindly|can\s+you|could\s+you)\s+)?"
                r"(?:gen[a-z]{3,7}e?|draw|create|make|render|paint|sketch|illustrate)\s+"
                r"(?:an?\s+)?"
                r"(?:(?:photorealistic|photo-realistic|realistic|cinematic|high-quality|high-res|hd|4k|8k)\s+)?"
                r"(?:image|imaeg|picture|pic|photo\b|photograph\b|wallpaper|portrait|illustration|drawing)\s*(?:of|for|about)?\s*",
                "",
                stripped,
                flags=re.IGNORECASE
            ).strip()
            if len(cleaned) >= 2:
                return True, cleaned

        # Catch common conversational visual requests with visual subjects or styles.
        # Whole words only, and the verb must open the request: "Render's docs ... per workspace"
        # is a question about a company, not "render ... space".
        if _LEADING_GEN_VERB.match(lower) and \
           not any(w in lower.split() for w in NON_IMAGE_TEXT_KEYWORDS) and \
           (
               _has_word(lower, [
                   "car", "ferrari", "porsche", "lamborghini", "vehicle", "bike", "bicycle", "motorcycle",
                   "portrait", "boy", "girl", "cat", "dog", "tiger", "lion", "animal", "sunset", "sunrise",
                   "beach", "mountain", "ocean", "landscape", "scenery", "spaceship", "city", "dragon",
                   "castle", "space", "robot", "cyberpunk", "wallpaper", "avatar", "logo", "earphone", "headphones",
                   "headphone", "earbuds"
               ]) or
               any(s in lower for s in ("photorealistic", "photo-realistic", "realistic", "cinematic", "hyperrealistic", "3d render", "oil painting", "watercolor"))
           ):
            style_match = re.search(r"\b(photorealistic|photo-realistic|realistic|cinematic|hyperrealistic|3d render|oil painting|watercolor)\b", stripped, re.IGNORECASE)
            style_prefix = style_match.group(1) if style_match else ""

            sub = re.sub(
                r"^\s*(?:please\s+)?(?:generate|create|draw|make|geneate|genreate|genrate|render|paint)\s+(?:an?\s+)?(?:image|picture|photo\s+of\s+)?",
                "",
                stripped,
                flags=re.IGNORECASE
            ).strip()
            sub = re.sub(r"^(?:an?\s+)?(?:image|picture|photo|illustration|drawing)\s+(?:of\s+)?", "", sub, flags=re.IGNORECASE).strip()

            if style_prefix and style_prefix.lower() not in sub.lower():
                sub = f"{style_prefix} {sub}".strip()

            if len(sub) >= 2:
                return True, sub

        # Catch requests to see or view verified campus facilities/assets
        if _has_word(lower, ("show", "view", "display", "see", "photo", "pic", "picture", "image")) and \
           not _has_word(lower, NON_IMAGE_TEXT_KEYWORDS | {"timing", "timings", "fee", "fees", "rule", "rules", "syllabus", "admission", "contact", "phone", "email"}):
            for preset in SHOWCASE_PRESETS:
                if _has_word(lower, preset["keywords"]):
                    return True, preset["keywords"][0]

        return False, None

    async def enhance_prompt_async(self, subject: str, style: str = "photorealistic") -> str:
        """Asynchronously expand user concept into cinema-grade blueprint using LLMPromptEnhancer with rule-based fallback."""
        try:
            return await prompt_enhancer.expand(subject, style=style)
        except Exception as e:
            logger.warning(f"[ImageService] Async prompt enhancement failed: {e}. Using sync fallback.")
            return self.enhance_prompt(subject)

    def enhance_prompt(self, subject: str) -> str:
        """Enrich user prompt for aesthetic quality while strictly respecting user-defined style and medium."""
        # Sanitize whitespace and quotes that break URL encoding
        clean_subject = re.sub(r'[\r\n\t]+', ' ', subject).strip()
        clean_subject = re.sub(r'["\']', '', clean_subject)
        lower = clean_subject.lower()

        # If subject is already very detailed (>25 words), leave exactly as-is
        if len(clean_subject.split()) > 25:
            return clean_subject

        # Explicit artistic medium, visual style, or photographic treatment keywords
        style_keywords = (
            # Traditional mediums
            "oil painting", "watercolor", "watercolour", "acrylic", "gouache", "pastel",
            "charcoal", "sketch", "drawing", "line art", "ink", "pencil", "canvas",
            # Digital & Illustrative styles
            "pixel art", "8-bit", "16-bit", "anime", "manga", "ghibli", "comic", "cartoon",
            "vector", "flat art", "minimalist", "logo", "icon", "pop art", "concept art",
            "cyberpunk", "steampunk", "vaporwave", "synthwave", "surrealism", "impressionism",
            # 3D & Digital engines
            "3d render", "isometric", "octane", "blender", "unreal engine", "cgi", "claymation",
            # Photography & Lighting specifics
            "photo", "photograph", "photography", "polaroid", "macro", "drone", "35mm",
            "black and white", "monochrome", "sepia", "film grain", "cinematic lighting"
        )
        user_specified_style = any(kw in lower for kw in style_keywords)

        # Case 1: User explicitly specified their style/medium -> Preserve exact intent, append only neutral fidelity
        if user_specified_style:
            if any(qualifier in lower for qualifier in ("detailed", "high quality", "8k", "masterpiece", "award winning")):
                return clean_subject
            return f"{clean_subject}, high quality, detailed composition"

        # Case 2: Bare subject without specified style -> Apply subject-appropriate default visual treatment
        if any(w in lower for w in ("car", "vehicle", "coupe", "automobile", "sedan", "supercar", "suv", "truck")):
            return (
                f"{clean_subject}, sleek modern automotive photography, studio lighting, "
                f"clean reflections, shallow depth of field, 8k resolution, crisp focus"
            )
        elif any(w in lower for w in ("bike", "motorcycle", "bicycle", "superbike")):
            return (
                f"{clean_subject}, high-performance motorcycle photography, dramatic lighting, "
                f"clean reflections, 8k resolution, photorealistic, crisp focus"
            )
        elif any(w in lower for w in ("earphone", "earphones", "headphone", "headphones", "earbuds", "gadget", "watch", "phone", "laptop")):
            return (
                f"{clean_subject}, sleek industrial product photography, clean reflective background, "
                f"soft studio lighting, macro lens detail, 8k UHD, crisp focus"
            )
        elif any(w in lower for w in ("boy", "girl", "man", "woman", "person", "portrait", "face", "warrior", "astronaut")):
            return (
                f"{clean_subject}, natural portrait photography, soft rim lighting, "
                f"85mm portrait lens, shallow depth of field, highly detailed, 8k resolution"
            )
        elif any(w in lower for w in ("landscape", "nature", "mountain", "forest", "sunset", "ocean", "beach", "river", "desert")):
            return (
                f"{clean_subject}, scenic landscape photography, golden hour natural light, "
                f"high dynamic range, 8k resolution, crisp atmospheric depth"
            )
        elif any(w in lower for w in ("cat", "dog", "tiger", "lion", "animal", "bird", "wildlife")):
            return (
                f"{clean_subject}, wildlife photography, natural lighting, fine fur and texture detail, "
                f"shallow depth of field, 8k resolution, sharp focus"
            )
        else:
            return (
                f"{clean_subject}, clean studio lighting, highly detailed, photorealistic, 8k resolution, sharp focus"
            )




    def _categorize_error(self, status: int, detail: str) -> Tuple[str, str]:
        """Categorize error status and detail into strongly-typed error code and user-facing explanation."""
        if status == 429:
            return "rate_limited", "Image provider rate limit reached (429). Please wait a moment before retrying."
        elif status in (408, 504):
            return "timeout", "The image generation service timed out waiting for worker."
        elif status in (400, 422):
            return "content_filter", "Prompt could not be processed due to content moderation or invalid parameters."
        elif status in (500, 502, 503):
            return "server_error", f"Upstream image generation server returned error ({status})."
        elif "timeout" in detail.lower():
            return "timeout", "The image request timed out while generating."
        elif "connect" in detail.lower() or "network" in detail.lower():
            return "network_error", "Network connection to image generation service failed."
        else:
            return "server_error", f"Image generation delayed or failed: {detail or 'Provider unavailable'}"

    async def generate_and_validate(self, prompt, model="flux", width=1024, height=1024,
                                    seed=None, enhance=True, tier="B", style="photorealistic", owner_id=None,
                                    idempotency_key=None, generation_allowed=True):
        """Generate genuine provider bytes through the shared private image path."""
        if not generation_allowed:
            return {"success": False, "model": "none", "image_url": "",
                    "error_type": "image_intent_required", "status_code": 409,
                    "error_message": "This saved image request has no recoverable intent identity. Start a new request only if you intend another image."}
        from app.services.image.genuine_service import generate
        return await generate(prompt, owner=owner_id, model=model, width=width, height=height,
                              seed=seed, enhance=enhance, tier=tier, style=style,
                              idempotency_key=idempotency_key)

    def generate_image_url(self, prompt, **kwargs):
        """Synchronous callers cannot synthesize or substitute catalog photos."""
        safe, reason = check_image_safety(prompt)
        return {"success": False, "image_url": "error:async_required:Use%20authenticated%20generation",
                "original_prompt": prompt, "enhanced_prompt": prompt, "model": "none", "seed": 0,
                "width": 0, "height": 0, "error_type": "async_required" if safe else "content_filter",
                "error_message": "Use authenticated image generation." if safe else reason}


image_service = ImageGenerationService()

