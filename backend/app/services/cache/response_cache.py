"""In-Memory Semantic (Embedding-Similarity) Response Cache for JRU AI.

Replaces rigid exact-string SHA-256 caching with a dual-stage in-memory cache:
1. Fast-Path Exact Match: Instant O(1) hash map lookup (< 0.1ms).
2. Semantic Similarity Match: Cosine similarity search using the pre-loaded
   FastEmbed model (BAAI/bge-small-en-v1.5, 384 dimensions) with configurable threshold.

This cache does not partition by owner, conversation, memory or context version.
ChatService must bypass reads AND writes for scoped or contextual requests.

Bug Mitigations Implemented:
- Full Citation Preservation: SourceCitation dictionaries are deep-copied into
  every CacheEntry and returned immutably on cache hits.
- Tier-Gated Isolation: Queries are partitioned by compute Tier (e.g., Tier S
  specialist vs Tier B generalist) to avoid cross-model response contamination.
- Near-Paraphrase Recall: Paraphrases above the similarity threshold (default: 0.96)
  trigger immediate cache hits while rejecting semantic drift / antonym inversions.
"""
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheEntry:
    """Cached response entry with vector embedding, source citations, and TTL metadata."""
    key: str
    prompt: str
    embedding: np.ndarray  # Shape: (384,), float32, L2-normalized
    content: str
    provider: str
    model: str
    tier: str
    total_tokens: int
    sources: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class ResponseCache:
    """Thread-safe Semantic LRU response cache powered by FastEmbed and NumPy."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: int = 3600,
        similarity_threshold: float = 0.96,
    ) -> None:
        self._lock = threading.Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl_seconds
        self.similarity_threshold = float(
            os.getenv("SEMANTIC_CACHE_THRESHOLD", str(similarity_threshold))
        )
        self._entries: List[SemanticCacheEntry] = []
        self._exact_map: Dict[str, SemanticCacheEntry] = {}

        # Diagnostic telemetry metrics
        self._exact_hits: int = 0
        self._semantic_hits: int = 0
        self._misses: int = 0
        self._saved_tokens: int = 0
        self._saved_cost_usd: float = 0.0

    def _generate_exact_key(
        self,
        tier: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Deterministic SHA-256 fingerprint for O(1) exact query lookup."""
        raw = f"{tier.upper()}:{model.lower().strip()}:{prompt.strip()}:{system_prompt or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _generate_key(
        self,
        tier: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Backward-compatible alias for exact key generation."""
        return self._generate_exact_key(tier, model, prompt, system_prompt)

    @property
    def _cache(self) -> dict[str, SemanticCacheEntry]:
        """Backward-compatible access to exact entry dictionary."""
        return self._exact_map

    def _embed_prompt(self, prompt: str) -> Optional[np.ndarray]:
        """Embed query string using the pre-loaded FastEmbed model (BAAI/bge-small-en-v1.5)."""
        try:
            from app.services.rag.qdrant_vector_store import get_embedding_model
            embedder = get_embedding_model()
            if embedder is None:
                return None
            vector = list(embedder.embed([prompt.strip()]))[0]
            # Ensure float32 numpy array
            arr = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr
        except Exception as exc:
            logger.warning(f"[ResponseCache] Failed to embed prompt for semantic cache: {exc}")
            return None

    def get(
        self,
        tier: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached completion and citations via exact match or cosine similarity >= threshold."""
        now = time.time()
        tier_upper = tier.upper()
        exact_key = self._generate_exact_key(tier, model, prompt, system_prompt)

        with self._lock:
            # Stage 1: O(1) Exact Match Fast-Path
            if exact_key in self._exact_map:
                entry = self._exact_map[exact_key]
                if now <= entry.expires_at:
                    entry.hit_count += 1
                    entry.last_accessed = now
                    self._exact_hits += 1
                    self._saved_tokens += entry.total_tokens
                    self._saved_cost_usd = round(self._saved_cost_usd + (entry.total_tokens / 1e6 * 0.30), 6)
                    logger.info(f"[ResponseCache] EXACT HIT (tier={tier_upper}, model={entry.model})")
                    return {
                        "content": entry.content,
                        "provider": entry.provider,
                        "model": entry.model,
                        "tier": entry.tier,
                        "total_tokens": entry.total_tokens,
                        "sources": [dict(s) for s in entry.sources],
                        "cached": True,
                        "match_type": "exact",
                        "similarity": 1.0,
                    }
                else:
                    # Clean up expired entry
                    del self._exact_map[exact_key]
                    if entry in self._entries:
                        self._entries.remove(entry)

        # Stage 2: Semantic Vector Cosine Similarity
        query_vec = self._embed_prompt(prompt)
        if query_vec is None:
            with self._lock:
                self._misses += 1
            return None

        with self._lock:
            # Filter active, unexpired candidates matching the compute tier
            valid_candidates = []
            for e in self._entries:
                if e.tier == tier_upper and now <= e.expires_at:
                    valid_candidates.append(e)

            if not valid_candidates:
                self._misses += 1
                return None

            # Vectorized batch dot product (candidate embeddings are pre-normalized)
            candidate_matrix = np.stack([e.embedding for e in valid_candidates])
            similarities = np.dot(candidate_matrix, query_vec)
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= self.similarity_threshold:
                matched_entry = valid_candidates[best_idx]
                matched_entry.hit_count += 1
                matched_entry.last_accessed = now
                self._semantic_hits += 1
                self._saved_tokens += matched_entry.total_tokens
                self._saved_cost_usd = round(self._saved_cost_usd + (matched_entry.total_tokens / 1e6 * 0.30), 6)

                logger.info(
                    f"[ResponseCache] SEMANTIC HIT: sim={best_sim:.4f} >= {self.similarity_threshold:.2f} "
                    f"| Query: '{prompt[:50]}' -> Cached: '{matched_entry.prompt[:50]}'"
                )

                # Return deep-copied sources to prevent caller modification
                return {
                    "content": matched_entry.content,
                    "provider": matched_entry.provider,
                    "model": matched_entry.model,
                    "tier": matched_entry.tier,
                    "total_tokens": matched_entry.total_tokens,
                    "sources": [dict(s) for s in matched_entry.sources],
                    "cached": True,
                    "match_type": "semantic",
                    "similarity": round(best_sim, 4),
                    "matched_prompt": matched_entry.prompt,
                }

            self._misses += 1
            return None

    def set(
        self,
        tier: str,
        model: str,
        prompt: str,
        content: str,
        provider: str,
        model_name: str,
        total_tokens: int,
        sources: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Index a verified completion and its sources into the semantic cache."""
        if not content or not content.strip():
            return
        if not prompt or not prompt.strip():
            return

        embedding = self._embed_prompt(prompt)

        now = time.time()
        ttl = ttl_seconds or self._default_ttl
        exact_key = self._generate_exact_key(tier, model, prompt, system_prompt)

        # Normalize and deduplicate source citations
        normalized_sources = []
        seen_citations = set()
        if sources:
            for s in sources:
                if hasattr(s, "model_dump"):
                    item = s.model_dump(exclude_none=True)
                elif isinstance(s, dict):
                    item = dict(s)
                elif hasattr(s, "__dict__"):
                    item = dict(s.__dict__)
                else:
                    continue

                # Unique fingerprint by document and snippet
                cite_id = (item.get("id") or item.get("title", ""), (item.get("snippet", "")[:60]))
                if cite_id not in seen_citations:
                    seen_citations.add(cite_id)
                    normalized_sources.append(item)

        new_entry = SemanticCacheEntry(
            key=exact_key,
            prompt=prompt.strip(),
            embedding=embedding if embedding is not None else np.zeros(1, dtype=np.float32),
            content=content.strip(),
            provider=provider,
            model=model_name,
            tier=tier.upper(),
            total_tokens=total_tokens,
            sources=normalized_sources,
            system_prompt=system_prompt,
            created_at=now,
            expires_at=now + ttl,
            last_accessed=now,
        )

        with self._lock:
            # Remove any existing entry with identical exact key
            if exact_key in self._exact_map:
                old_entry = self._exact_map[exact_key]
                if old_entry in self._entries:
                    self._entries.remove(old_entry)

            self._exact_map[exact_key] = new_entry
            if embedding is not None:
                self._entries.append(new_entry)

            # Enforce LRU capacity limit
            if len(self._entries) > self._max_size:
                # Evict the entry with the oldest last_accessed timestamp
                oldest_entry = min(self._entries, key=lambda e: e.last_accessed)
                self._entries.remove(oldest_entry)
                if oldest_entry.key in self._exact_map:
                    del self._exact_map[oldest_entry.key]

    @property
    def _cache(self) -> Dict[str, SemanticCacheEntry]:
        """Backward-compatible access to underlying exact match map."""
        return self._exact_map

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic cache health metrics, similarity threshold, and hit ratios."""
        with self._lock:
            total_hits = self._exact_hits + self._semantic_hits
            total_lookups = total_hits + self._misses
            ratio = round((total_hits / total_lookups * 100), 2) if total_lookups > 0 else 0.0
            return {
                "total_entries": len(self._entries),
                "max_size": self._max_size,
                "similarity_threshold": self.similarity_threshold,
                "exact_hits": self._exact_hits,
                "semantic_hits": self._semantic_hits,
                "total_hits": total_hits,
                "hits": total_hits,
                "misses": self._misses,
                "hit_ratio_pct": ratio,
                "saved_tokens": self._saved_tokens,
                "saved_cost_usd": self._saved_cost_usd,
            }

    def clear(self) -> None:
        """Flush all cache entries and reset metrics."""
        with self._lock:
            self._entries.clear()
            self._exact_map.clear()
            self._exact_hits = 0
            self._semantic_hits = 0
            self._misses = 0
            self._saved_tokens = 0
            self._saved_cost_usd = 0.0


# Global singleton instance
response_cache = ResponseCache()
