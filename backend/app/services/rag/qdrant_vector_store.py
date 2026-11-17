"""Qdrant Embedded Vector Store & Neural Cross-Encoder Reranker Service.

Replaces in-memory substring Jaccard matching with:
1. Embedded Qdrant Vector Store (file-backed or in-memory) with cosine distance metric
2. FastEmbed ONNX Dense Vector Embeddings (BAAI/bge-small-en-v1.5, 384 dimensions)
3. FlashRank Neural Cross-Encoder Reranker (ms-marco-TinyBERT-L-2-v2) for deep cross-attention
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Lazy global instances for model caching
_EMBED_MODEL = None
_RANKER_MODEL = None
_QDRANT_CLIENT = None

_EMBED_INIT_LOCK = threading.Lock()
_RANKER_INIT_LOCK = threading.Lock()
_QDRANT_INIT_LOCK = threading.Lock()


def get_process_rss_mb() -> float:
    """Return the current resident set size (RSS) memory in MiB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def is_embedding_enabled() -> tuple:
    """Check if dense vector embedding is enabled and safe given memory constraints."""
    try:
        from app.core.config import settings
        is_prod = (getattr(settings, "ENVIRONMENT", "") or "").lower().strip() == "production"
        # 1. In production (512MB Render free tier), local dense embeddings are disabled by default
        if is_prod and not getattr(settings, "ENABLE_LOCAL_EMBEDDINGS", False):
            return False, "local_embeddings_disabled_in_production"
        # 2. General config toggle
        if not getattr(settings, "ENABLE_LOCAL_EMBEDDINGS", True):
            return False, "local_embeddings_disabled_by_config"
        # 3. Dynamic RSS memory safety headroom (512MB Render limit safeguard in production)
        if is_prod:
            rss = get_process_rss_mb()
            max_rss = getattr(settings, "MAX_PROCESS_RSS_MB", 380.0)
            if rss > max_rss:
                logger.warning(
                    f"[EmbeddingModel] Current RSS ({rss:.1f} MiB) exceeds safety ceiling ({max_rss:.1f} MiB). "
                    "Disabling local dense embeddings to prevent OOM."
                )
                return False, f"rss_exceeds_budget_{rss:.1f}mb"
        return True, "enabled"
    except Exception as exc:
        return False, f"check_failed: {exc}"


def is_flashrank_enabled() -> tuple:
    """Check if FlashRank cross-encoder reranking is enabled and safe given memory constraints."""
    try:
        from app.core.config import settings
        # 1. In low-memory mode, FlashRank is disabled unless explicitly enabled
        if getattr(settings, "LOW_MEMORY_RAG_MODE", True):
            if not getattr(settings, "ENABLE_FLASHRANK", False):
                return False, "low_memory_mode_active"
        # 2. General config toggle
        if not getattr(settings, "ENABLE_FLASHRANK", False):
            return False, "flashrank_disabled_by_config"
        # 3. Dynamic RSS memory safety headroom (512MB Render limit safeguard)
        rss = get_process_rss_mb()
        max_rss = getattr(settings, "MAX_PROCESS_RSS_MB", 380.0)
        if rss > max_rss:
            logger.warning(
                f"[NeuralReranker] Current RSS ({rss:.1f} MiB) exceeds safety ceiling ({max_rss:.1f} MiB). "
                "Disabling FlashRank to prevent OOM."
            )
            return False, f"rss_exceeds_budget_{rss:.1f}mb"
        return True, "enabled"
    except Exception as exc:
        return False, f"check_failed: {exc}"


class DomainBGEEmbedding:
    """High-speed ONNX runtime runner for fine-tuned JRU domain BGE embeddings."""

    def __init__(self, model_path: str, tokenizer_path: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=512)

    def embed(self, texts: Any):
        import numpy as np
        text_list = list(texts)
        for text in text_list:
            enc = self.tokenizer.encode(str(text))
            inputs = {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            }
            outputs = self.session.run(None, inputs)
            yield outputs[0][0]


def get_embedding_model():
    """Retrieve or lazily initialize domain-tuned or FastEmbed text embedding engine."""
    global _EMBED_MODEL
    enabled, reason = is_embedding_enabled()
    if not enabled:
        logger.debug(f"[EmbeddingModel] Dense embedding bypassed ({reason}); returning None.")
        return None

    if _EMBED_MODEL is None:
        with _EMBED_INIT_LOCK:
            if _EMBED_MODEL is None:
                onnx_model_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../../../data/models/jru-bge-domain/model.onnx")
                )
                tokenizer_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../../../data/models/jru-bge-domain/tokenizer.json")
                )
                if os.path.exists(onnx_model_path) and os.path.exists(tokenizer_path):
                    try:
                        logger.info(f"Loading Fine-Tuned JRU Domain BGE ONNX Model from {onnx_model_path}...")
                        _EMBED_MODEL = DomainBGEEmbedding(onnx_model_path, tokenizer_path)
                        return _EMBED_MODEL
                    except Exception as exc:
                        logger.warning(f"Failed to load fine-tuned ONNX domain model: {exc}. Falling back to FastEmbed.")

                from fastembed import TextEmbedding
                logger.info("Initializing FastEmbed embedding model (BAAI/bge-small-en-v1.5)...")
                _EMBED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", **_model_cache_kwargs())
    return _EMBED_MODEL


def _model_cache_kwargs() -> dict:
    """Use MODEL_CACHE_DIR when set, so models fetched at build time (scripts/prefetch_models.py)
    are reused instead of downloaded on the first request after every cold start."""
    cache_dir = os.getenv("MODEL_CACHE_DIR", "").strip()
    return {"cache_dir": cache_dir} if cache_dir else {}


def get_reranker_model():
    """Retrieve or lazily initialize FlashRank neural cross-encoder ranker."""
    global _RANKER_MODEL
    if _RANKER_MODEL is None:
        with _RANKER_INIT_LOCK:
            if _RANKER_MODEL is None:
                from flashrank import Ranker
                logger.info("Initializing FlashRank neural cross-encoder (ms-marco-TinyBERT-L-2-v2)...")
                _RANKER_MODEL = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", **_model_cache_kwargs())
    return _RANKER_MODEL


from app.services.rag.document_rag import BM25Index, tokenize


class BoundedBM25Cache:
    """Thread-safe bounded LRU cache for collection BM25 indexes."""

    def __init__(self, maxsize: int = 10):
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, val: Dict[str, Any]) -> None:
        with self._lock:
            # Always store the new value: incremental BM25 rebuilds replace the entry.
            self._cache[key] = val
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Drop one collection's index without touching other collections."""
        with self._lock:
            self._cache.pop(key, None)

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def size(self) -> int:
        return len(self)


# Global bounded in-memory cache for BM25 collection indices (max 10 collections)
_BM25_COLLECTION_CACHE = BoundedBM25Cache(maxsize=10)

# Page size used when scrolling a whole collection to rebuild its BM25 corpus
BM25_SCROLL_PAGE_SIZE = 1000


class EmbeddingUnavailableError(RuntimeError):
    """Raised when a persistent collection cannot be embedded with real vectors."""


class QueryEmbeddingCache:
    """Thread-safe LRU cache for high-dimensional dense query embeddings."""

    def __init__(self, maxsize: int = 2048) -> None:
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def get(self, query: str) -> Optional[List[float]]:
        key = self._normalize(query)
        if not key:
            return None
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def set(self, query: str, vector: List[float]) -> None:
        key = self._normalize(query)
        if not key or not vector:
            return
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                self._cache[key] = vector
                if len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


query_embedding_cache = QueryEmbeddingCache()


def get_or_compute_query_embedding(query: str) -> list[float] | None:
    """Retrieve pre-computed query embedding from LRU cache or compute and cache it."""
    cached = query_embedding_cache.get(query)
    if cached is not None:
        return cached
    embedder = get_embedding_model()
    if embedder is None:
        return None
    vector = list(embedder.embed([query]))[0].tolist()
    query_embedding_cache.set(query, vector)
    return vector


_KNOWN_COLLECTIONS: Set[str] = set()
_COLLECTIONS_LOCK = threading.Lock()



def get_qdrant_client(storage_path: Optional[str] = None):
    """Retrieve or lazily initialize Qdrant client (standalone container URL or local embedded)."""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        with _QDRANT_INIT_LOCK:
            if _QDRANT_CLIENT is None:
                from qdrant_client import QdrantClient
                try:
                    from app.core.config import settings
                    qdrant_url = os.getenv("QDRANT_URL") or getattr(settings, "QDRANT_URL", None)
                    qdrant_api_key = os.getenv("QDRANT_API_KEY") or getattr(settings, "QDRANT_API_KEY", None)
                except Exception:
                    qdrant_url = os.getenv("QDRANT_URL")
                    qdrant_api_key = os.getenv("QDRANT_API_KEY")
                if qdrant_url:
                    qdrant_url = qdrant_url.strip()
                    if qdrant_api_key:
                        qdrant_api_key = qdrant_api_key.strip()
                    logger.info(f"Connecting to standalone Qdrant server at {qdrant_url}...")
                    _QDRANT_CLIENT = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                else:
                    path = storage_path or os.getenv("QDRANT_STORAGE_PATH")
                    if not path:
                        # Canonical absolute path: <repo>/backend/data/qdrant_db
                        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/qdrant_db"))
                    if path == ":memory:":
                        logger.info("Initializing in-memory Qdrant client...")
                        _QDRANT_CLIENT = QdrantClient(":memory:")
                    else:
                        os.makedirs(path, exist_ok=True)
                        logger.info(f"Initializing file-backed Qdrant client at {path}...")
                        _QDRANT_CLIENT = QdrantClient(path=path)
    return _QDRANT_CLIENT


class QdrantVectorStore:
    """Embedded Vector Store powered by Qdrant and FastEmbed with Hybrid Search (BM25 + Dense + RRF)."""

    VECTOR_DIM = 384

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self.client = get_qdrant_client(storage_path)

    def _is_known_collection(self, collection_name: str) -> bool:
        """Check if collection exists using memory cache to eliminate remote network roundtrips."""
        with _COLLECTIONS_LOCK:
            if collection_name in _KNOWN_COLLECTIONS:
                return True
            try:
                collections = [c.name for c in self.client.get_collections().collections]
                _KNOWN_COLLECTIONS.update(collections)
                return collection_name in _KNOWN_COLLECTIONS
            except Exception as exc:
                logger.warning(f"Failed to check collections in Qdrant: {exc}")
                return False

    def ensure_collection(self, collection_name: str) -> None:
        """Create Qdrant collection with 384-dim cosine distance and owner_id payload index."""
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
        if not self._is_known_collection(collection_name):
            logger.info(f"Creating Qdrant collection: {collection_name} (dim={self.VECTOR_DIM}, Cosine)")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE),
            )
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="owner_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.debug(f"Payload index creation on {collection_name}: {e}")
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="file_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.debug(f"Payload index file_id on {collection_name}: {e}")
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="chunk_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.debug(f"Payload index chunk_id on {collection_name}: {e}")
            with _COLLECTIONS_LOCK:
                _KNOWN_COLLECTIONS.add(collection_name)

    def delete_document_chunks(
        self,
        collection_name: str,
        file_id: str,
        owner_id: Optional[str] = None,
    ) -> int:
        """Delete all chunks belonging to a document from Qdrant and invalidate BM25 cache."""
        if not self.client.collection_exists(collection_name):
            return 0
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            must_conditions = [
                FieldCondition(key="file_id", match=MatchValue(value=file_id))
            ]
            if owner_id:
                must_conditions.append(
                    FieldCondition(key="owner_id", match=MatchValue(value=owner_id))
                )
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(must=must_conditions),
                wait=True,
            )
            if self.client.count(collection_name, count_filter=Filter(must=must_conditions), exact=True).count:
                raise RuntimeError("Document vectors remain after deletion")
            # Invalidate only this collection's BM25 index; next use re-scrolls current points
            _BM25_COLLECTION_CACHE.invalidate(collection_name)
            logger.info(f"Deleted chunks for file_id {file_id} from {collection_name}.")
            return 1
        except Exception as exc:
            logger.warning(f"Failed to delete chunks for file {file_id} from {collection_name}: {exc}")
            raise RuntimeError("Document vector deletion failed") from exc

    def index_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        owner_id: Optional[str] = None,
    ) -> int:
        """Embed document text chunks, upsert into Qdrant with tenant ownership, and update BM25 index.
        
        Returns:
            Number of indexed vectors.
        """
        if not chunks:
            return 0

        self.ensure_collection(collection_name)
        from qdrant_client.models import PointStruct

        embedder = get_embedding_model()
        if embedder is None:
            embeddings = self._placeholder_vectors_or_raise(collection_name, len(chunks))
        else:
            texts = [
                f"File: {c.get('doc_name', 'document')}\n{c['text']}" if c.get("doc_name") else c["text"]
                for c in chunks
            ]
            embeddings = list(embedder.embed(texts))

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_owner = str(chunk.get("owner_id") or owner_id or "public")
            chunk["owner_id"] = chunk_owner
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection_name}_{chunk.get('id', idx)}"))
            payload = {
                "chunk_id": chunk.get("id", f"doc-{idx}"),
                "file_id": chunk.get("file_id", ""),
                "doc_name": chunk.get("doc_name", "document"),
                "owner_id": chunk_owner,
                "section": chunk.get("section", ""),
                "subsection": chunk.get("subsection", ""),
                "breadcrumb": chunk.get("breadcrumb", ""),
                "page": chunk.get("page", 1),
                "text": chunk["text"],
            }
            vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            points.append(PointStruct(id=point_id, vector=vec_list, payload=payload))

        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Successfully indexed {len(points)} vectors into {collection_name} (owner={owner_id or 'public'}).")

        # Build in-memory BM25 index for sparse keyword matching
        self._build_bm25_index(collection_name, chunks)
        return len(points)

    def delete_owner_chunks(self, collection_name: str, owner_id: str) -> None:
        """Erase owner-tagged orphan vectors without touching other owners."""
        if not owner_id or not self.client.collection_exists(collection_name):
            return
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        selector = Filter(must=[FieldCondition(key="owner_id", match=MatchValue(value=owner_id))])
        self.client.delete(collection_name, points_selector=selector, wait=True)
        if self.client.count(collection_name, count_filter=selector, exact=True).count:
            raise RuntimeError("Owned vectors remain after erasure")
        _BM25_COLLECTION_CACHE.invalidate(collection_name)

    def _placeholder_vectors_or_raise(self, collection_name: str, count: int) -> list[list[float]]:
        """Decide what to store when no embedder is available.

        Zero vectors in a collection that is dense-searched poison cosine scoring,
        so a persistent collection fails the write instead. Two cases keep the old
        placeholder: in-memory transient attachment collections (dense search there
        needs a query embedding, which is also unavailable, so it returns nothing),
        and deployments that disable local embeddings by configuration, where dense
        search never runs and the collection is BM25-only by design.
        """
        _, reason = is_embedding_enabled()
        bm25_only_by_config = str(reason).startswith("local_embeddings_disabled")
        if collection_name.startswith("transient_") or bm25_only_by_config:
            logger.info(
                f"Embedding model unavailable ({reason}); storing {count} BM25-only chunks in {collection_name}."
            )
            return [[0.0] * self.VECTOR_DIM for _ in range(count)]
        raise EmbeddingUnavailableError(
            f"Embedding model unavailable ({reason}); refusing to index {count} chunks into "
            f"'{collection_name}' with zero vectors. Retry when embeddings are available."
        )

    def _scroll_collection_chunks(self, collection_name: str) -> list[dict[str, Any]]:
        """Read every point payload in a collection, following scroll pagination."""
        chunks: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                limit=BM25_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                chunks.append({
                    "id": p.payload.get("chunk_id", str(p.id)),
                    "file_id": p.payload.get("file_id", ""),
                    "doc_name": p.payload.get("doc_name", "document"),
                    "owner_id": p.payload.get("owner_id", "public"),
                    "section": p.payload.get("section", ""),
                    "subsection": p.payload.get("subsection", ""),
                    "breadcrumb": p.payload.get("breadcrumb", ""),
                    "page": p.payload.get("page", 1),
                    "text": p.payload.get("text", ""),
                })
            if offset is None or not points:
                return chunks

    def _build_bm25_index(self, collection_name: str, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to the collection's BM25 index, seeding it from Qdrant on a cache miss."""
        existing = _BM25_COLLECTION_CACHE.get(collection_name)
        if existing is not None:
            base_chunks = list(existing["chunks"])
        elif self._is_known_collection(collection_name):
            base_chunks = self._scroll_collection_chunks(collection_name)
        else:
            base_chunks = []
        self._fit_bm25(collection_name, base_chunks, chunks)

    def _fit_bm25(
        self,
        collection_name: str,
        base_chunks: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> None:
        """Fit and cache a BM25 index over base_chunks plus unseen chunks."""
        all_chunks = list(base_chunks)
        chunk_ids = {c.get("id") for c in all_chunks}
        for c in chunks:
            if c.get("id") not in chunk_ids:
                all_chunks.append(c)
                chunk_ids.add(c.get("id"))
        tokenized_docs = [
            tokenize(f"{c.get('doc_name', '')} {c.get('breadcrumb', '')} {c.get('section', '')} {c['text']}")
            for c in all_chunks
        ]
        bm25 = BM25Index()
        bm25.fit(tokenized_docs)
        _BM25_COLLECTION_CACHE.set(collection_name, {
            "bm25": bm25,
            "chunks": all_chunks,
        })
        logger.info(f"Built BM25 keyword index for '{collection_name}' ({len(all_chunks)} passages).")

    def _ensure_bm25_index(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Ensure BM25 index exists for collection; lazily populate via Qdrant scroll if needed."""
        cached = _BM25_COLLECTION_CACHE.get(collection_name)
        if cached is not None:
            return cached

        if not self._is_known_collection(collection_name):
            return None

        # Scroll every point out of Qdrant (paginated; no truncation)
        chunks = self._scroll_collection_chunks(collection_name)
        if not chunks:
            return None

        self._fit_bm25(collection_name, chunks, [])
        return _BM25_COLLECTION_CACHE.get(collection_name)

    def search_dense(
        self,
        collection_name: str,
        query: str,
        limit: int = 15,
        query_vector: Optional[List[float]] = None,
        owner_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense semantic search in Qdrant with tenant isolation filtering."""
        if not self._is_known_collection(collection_name):
            return []

        if query_vector is None:
            query_vector = get_or_compute_query_embedding(query)
        if not query_vector:
            return []

        query_filter = None
        if owner_id:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                query_filter = Filter(
                    should=[
                        FieldCondition(key="owner_id", match=MatchValue(value=owner_id)),
                        FieldCondition(key="owner_id", match=MatchValue(value="public")),
                    ]
                )
            except Exception:
                query_filter = None
        elif collection_name.startswith("user_"):
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                query_filter = Filter(
                    must=[
                        FieldCondition(key="owner_id", match=MatchValue(value="public")),
                    ]
                )
            except Exception:
                query_filter = None

        try:
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit * 2 if owner_id else limit,
                query_filter=query_filter,
            )
        except Exception as filter_err:
            logger.debug(f"[QdrantFilterFallback] Filter query failed, falling back to unindexed query + post-filter: {filter_err}")
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit * 4 if owner_id else limit,
            )

        results = []
        for point in search_result.points:
            pt_owner = point.payload.get("owner_id", "public")
            if collection_name.startswith("user_"):
                from app.services.tenant_access import private_chunk_visible
                if not private_chunk_visible(owner_id, point.payload):
                    continue
            if owner_id:
                if pt_owner not in (owner_id, "public"):
                    continue
            elif collection_name.startswith("user_"):
                if pt_owner != "public":
                    continue
            results.append({
                "id": point.payload.get("chunk_id"),
                "file_id": point.payload.get("file_id", ""),
                "doc_name": point.payload.get("doc_name"),
                "owner_id": pt_owner,
                "section": point.payload.get("section", ""),
                "breadcrumb": point.payload.get("breadcrumb", ""),
                "page": point.payload.get("page"),
                "text": point.payload.get("text"),
                "dense_score": float(point.score),
            })
            if len(results) >= limit:
                break
        return results

    def search_sparse_bm25(
        self,
        collection_name: str,
        query: str,
        limit: int = 15,
        owner_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform lexical keyword search using Okapi BM25 with tenant isolation filtering."""
        bm25_data = self._ensure_bm25_index(collection_name)
        if not bm25_data:
            return []

        bm25 = bm25_data["bm25"]
        chunks = bm25_data["chunks"]
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = bm25.score(query_tokens)
        scored_indices = sorted(
            [(idx, s) for idx, s in enumerate(scores) if s > 0.0],
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for idx, score in scored_indices:
            c = chunks[idx]
            if collection_name.startswith("user_"):
                from app.services.tenant_access import private_chunk_visible
                if not private_chunk_visible(owner_id, c):
                    continue
            c_owner = c.get("owner_id", "public")
            if owner_id:
                if c_owner not in (owner_id, "public"):
                    continue
            elif collection_name.startswith("user_"):
                if c_owner != "public":
                    continue
            results.append({
                "id": c.get("id"),
                "file_id": c.get("file_id", ""),
                "doc_name": c.get("doc_name"),
                "owner_id": c_owner,
                "section": c.get("section"),
                "breadcrumb": c.get("breadcrumb"),
                "page": c.get("page", 1),
                "text": c["text"],
                "sparse_score": float(score),
            })
            if len(results) >= limit:
                break
        return results

    def search_hybrid(
        self,
        collection_name: str,
        query: str,
        limit: int = 15,
        rrf_k: int = 20,
        query_vector: Optional[List[float]] = None,
        owner_id: Optional[str] = None,
        w_dense: float = 1.0,
        w_sparse: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search with tenant isolation across both retrieval streams."""
        # Sequential execution eliminates ThreadPoolExecutor thread-stack memory bloat
        dense_results = self.search_dense(collection_name, query, limit, query_vector, owner_id)
        sparse_results = self.search_sparse_bm25(collection_name, query, limit, owner_id)

        if not sparse_results:
            return dense_results
        if not dense_results:
            return sparse_results

        # Reciprocal Rank Fusion (RRF) with evidence-backed dense and sparse weighting
        dense_ranks = {item["id"]: rank for rank, item in enumerate(dense_results)}
        sparse_ranks = {item["id"]: rank for rank, item in enumerate(sparse_results)}

        all_items: Dict[str, Dict[str, Any]] = {}
        for item in dense_results + sparse_results:
            cid = item["id"]
            if cid not in all_items:
                all_items[cid] = dict(item)

        for cid, item in all_items.items():
            rrf = 0.0
            if cid in dense_ranks:
                rrf += w_dense / (rrf_k + dense_ranks[cid] + 1)
            if cid in sparse_ranks:
                rrf += w_sparse / (rrf_k + sparse_ranks[cid] + 1)
            item["rrf_score"] = float(rrf)

        fused = sorted(all_items.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return fused[:limit]

    def get_passage_by_id(
        self,
        collection_name: str,
        passage_id: str,
        file_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific passage payload from Qdrant with tenant boundary validation."""
        if not self._is_known_collection(collection_name):
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchValue
        must_conditions = [
            FieldCondition(key="chunk_id", match=MatchValue(value=passage_id))
        ]
        if file_id:
            must_conditions.append(
                FieldCondition(key="file_id", match=MatchValue(value=file_id))
            )
        if owner_id:
            must_conditions.append(
                FieldCondition(key="owner_id", match=MatchValue(value=owner_id))
            )

        try:
            points, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=must_conditions),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                # If not found via exact filter, fallback to cached BM25 index chunks
                bm25_data = self._ensure_bm25_index(collection_name)
                if bm25_data and "chunks" in bm25_data:
                    for c in bm25_data["chunks"]:
                        if str(c.get("id")) == str(passage_id):
                            if file_id and str(c.get("file_id")) != str(file_id):
                                continue
                            if collection_name.startswith("user_"):
                                from app.services.tenant_access import private_chunk_visible
                                if not private_chunk_visible(owner_id, c):
                                    return None
                            return {
                                "passage_id": c.get("id"),
                                "file_id": c.get("file_id", ""),
                                "doc_name": c.get("doc_name", "document"),
                                "owner_id": c.get("owner_id", "public"),
                                "section": c.get("section", ""),
                                "subsection": c.get("subsection", ""),
                                "breadcrumb": c.get("breadcrumb", ""),
                                "page": c.get("page", 1),
                                "text": c.get("text", ""),
                            }
                return None

            p = points[0]
            pt_owner = p.payload.get("owner_id", "public")
            if collection_name.startswith("user_"):
                from app.services.tenant_access import private_chunk_visible
                if not private_chunk_visible(owner_id, p.payload):
                    return None

            return {
                "passage_id": p.payload.get("chunk_id", str(p.id)),
                "file_id": p.payload.get("file_id", ""),
                "doc_name": p.payload.get("doc_name", "document"),
                "owner_id": pt_owner,
                "section": p.payload.get("section", ""),
                "subsection": p.payload.get("subsection", ""),
                "breadcrumb": p.payload.get("breadcrumb", ""),
                "page": p.payload.get("page", 1),
                "text": p.payload.get("text", ""),
            }
        except Exception as exc:
            logger.warning(f"Error fetching passage '{passage_id}' from {collection_name}: {exc}")
            return None


class NeuralReranker:
    """Two-stage cross-encoder reranker powered by FlashRank with graceful degradation."""

    def __init__(self) -> None:
        self.ranker = None
        self.degraded_reason: str | None = None
        enabled, reason = is_flashrank_enabled()
        if not enabled:
            self.is_available = False
            self.degraded = True
            self.degraded_reason = reason
        else:
            try:
                self.ranker = get_reranker_model()
                self.is_available = True
                self.degraded = False
                self.degraded_reason = None
            except Exception as exc:
                self.is_available = False
                self.degraded = True
                self.degraded_reason = f"init_failed: {type(exc).__name__}"
                logger.warning("[NeuralReranker] FlashRank init failed; using RRF order.", exc_info=True)

    @property
    def rag_metadata(self) -> Dict[str, Any]:
        return {
            "rag_reranker_used": self.is_available,
            "rag_degraded": self.degraded,
            "rag_degraded_reason": self.degraded_reason,
        }

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate passages using neural cross-attention scores.
        
        Args:
            query: User prompt or question
            candidates: List of chunk dictionaries containing 'text' and metadata
            top_n: Number of top reranked chunks to return (or top_k alias)
            
        Returns:
            Reranked list of chunk dictionaries sorted by cross-encoder score,
            or RRF candidates directly if reranker is unavailable or degraded.
        """
        limit = top_n if top_n is not None else (top_k if top_k is not None else 4)
        if not candidates:
            return []
        if len(candidates) <= 1:
            return candidates[:limit]

        if not self.is_available or self.ranker is None:
            # Degraded / Low-memory mode: Return top candidates sorted by RRF directly
            return candidates[:limit]

        try:
            from flashrank import RerankRequest

            passages = [
                {
                    "id": idx,
                    "text": f"File: {c.get('doc_name', 'document')}\n{c['text']}" if c.get("doc_name") else c["text"],
                    "meta": c,
                }
                for idx, c in enumerate(candidates)
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            reranked_results = self.ranker.rerank(rerank_request)

            output = []
            for r in reranked_results[:limit]:
                item = dict(r["meta"])
                item["rerank_score"] = float(r["score"])
                output.append(item)

            return output
        except Exception as exc:
            logger.warning(f"[NeuralReranker] Runtime reranking error: {exc}. Falling back to RRF candidates.")
            self.degraded = True
            self.degraded_reason = f"runtime_exception: {exc}"
            return candidates[:limit]
