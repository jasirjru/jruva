"""30-Question Ground-Truth Evaluation Benchmark for RAG Retrieval & Reranking.

Spans 3 realistic document modalities:
1. Technical PDF Whitepapers (10 questions on distributed systems, consensus, kernel memory)
2. Production Code Repositories (10 questions on Python/TypeScript AST, concurrency, sockets)
3. Structured Markdown Specifications (10 questions on tables, RFC standards, SLAs, schemas)
"""
from typing import Any, Dict, List

# 30 Document Passages
EVAL_DOCUMENTS: List[Dict[str, Any]] = [
    # --- 10 PDF TECHNICAL PASSAGES ---
    {
        "id": "pdf-01",
        "doc_name": "distributed_consensus_raft.pdf",
        "page": 4,
        "doc_type": "pdf",
        "text": (
            "In Raft consensus, leader election is governed by randomized election timers. "
            "Each follower maintains a randomized countdown timer between 150ms and 300ms. "
            "If a follower receives no heartbeat AppendEntries RPC before the timer elapses, "
            "it transitions to Candidate state, increments its term counter, votes for itself, "
            "and broadcasts RequestVote RPCs to all cluster peers. To avoid split-brain split votes, "
            "randomized timers ensure that in majority of cases only a single follower times out first."
        ),
    },
    {
        "id": "pdf-02",
        "doc_name": "vector_database_indexing.pdf",
        "page": 12,
        "doc_type": "pdf",
        "text": (
            "Hierarchical Navigable Small World (HNSW) graphs achieve sub-linear nearest neighbor "
            "search by building a multi-layer geometric graph. The top layers feature long-range sparse "
            "edges for rapid skip-list navigation across clusters, while the bottom ground layer (layer 0) "
            "contains dense nearest-neighbor links. The parameter M controls the maximum number of outgoing "
            "edges per node, while efConstruction dictates the size of the dynamic candidate list during graph indexing."
        ),
    },
    {
        "id": "pdf-03",
        "doc_name": "database_internals_lsm.pdf",
        "page": 7,
        "doc_type": "pdf",
        "text": (
            "Log-Structured Merge-trees (LSM trees) optimize write throughput by converting random disk I/O "
            "into sequential append operations. Writes first append to an in-memory MemTable backed by a Write-Ahead "
            "Log (WAL). When the MemTable exceeds its capacity threshold (typically 64MB), it is frozen into an immutable "
            "table and flushed to disk as a Level-0 Sorted String Table (SSTable). Background compaction merges overlapping "
            "key ranges into deeper levels."
        ),
    },
    {
        "id": "pdf-04",
        "doc_name": "kernel_virtual_memory.pdf",
        "page": 19,
        "doc_type": "pdf",
        "text": (
            "The Linux virtual memory manager utilizes multi-level page tables with Translation Lookaside Buffers (TLB). "
            "On x86-64 hardware with 4-level paging, virtual addresses are decomposed into PGD, P4D, PUD, PMD, and PTE indices. "
            "When a process dereferences an unmapped address, the CPU raises a page fault interrupt (#PF), prompting the "
            "kernel handle_mm_fault routine to allocate a physical frame, update PTE permissions, and flush stale TLB entries."
        ),
    },
    {
        "id": "pdf-05",
        "doc_name": "tls_cryptographic_handshake.pdf",
        "page": 8,
        "doc_type": "pdf",
        "text": (
            "TLS 1.3 reduces handshake latency to a single round-trip (1-RTT) by obsoleting static RSA key exchange in "
            "favor of Ephemeral Diffie-Hellman (ECDHE). In the ClientHello message, the client preemptively sends its "
            "key share along with supported cipher suites. The server responds with ServerHello and its own key share, "
            "allowing both endpoints to derive application traffic keys before certificates and finished verification records are exchanged."
        ),
    },
    {
        "id": "pdf-06",
        "doc_name": "microservices_circuit_breaker.pdf",
        "page": 3,
        "doc_type": "pdf",
        "text": (
            "A circuit breaker protects downstream services from cascading collapse during localized outages. "
            "In the CLOSED state, failures increment a sliding window error counter. When the failure rate exceeds "
            "the trip threshold (e.g. 50% over 20 requests), the circuit trips to OPEN, rejecting incoming calls "
            "immediately with fast-failure exceptions. After a cooldown delay, the circuit enters HALF_OPEN, allowing a "
            "canary probe fraction of traffic to verify if the dependency has recovered."
        ),
    },
    {
        "id": "pdf-07",
        "doc_name": "database_isolation_mvcc.pdf",
        "page": 15,
        "doc_type": "pdf",
        "text": (
            "Multi-Version Concurrency Control (MVCC) eliminates read-write locking conflicts by storing multiple versions "
            "of each row identified by creation and deletion transaction IDs (xmin and xmax). Under Snapshot Isolation, "
            "a transaction reads the latest committed tuple version that was committed prior to the start of its snapshot. "
            "Dead rows superseded by newer committed versions must be reclaimed by background VACUUM workers."
        ),
    },
    {
        "id": "pdf-08",
        "doc_name": "cache_coherence_protocols.pdf",
        "page": 11,
        "doc_type": "pdf",
        "text": (
            "The MESI cache coherence protocol assigns four states to cache lines: Modified, Exclusive, Shared, and Invalid. "
            "When Core A wants to write to a cache line in Shared state, it must broadcast an Invalidating bus transaction. "
            "All peer cores transition their matching cache line to Invalid. Once acknowledgments return, Core A upgrades its line "
            "to Modified and commits the store locally without main memory stalling."
        ),
    },
    {
        "id": "pdf-09",
        "doc_name": "distributed_tracing_opentelemetry.pdf",
        "page": 6,
        "doc_type": "pdf",
        "text": (
            "W3C TraceContext defines distributed context propagation across microservice RPC boundaries. "
            "The traceparent HTTP header comprises four hyphen-separated fields: version (00), 16-byte hex trace-id, "
            "8-byte hex parent-span-id, and 2-byte trace-flags (e.g., 01 for sampled). This ensures that spans generated across "
            "heterogeneous asynchronous queues and REST boundaries correlate into a unified trace graph."
        ),
    },
    {
        "id": "pdf-10",
        "doc_name": "vector_product_quantization.pdf",
        "page": 22,
        "doc_type": "pdf",
        "text": (
            "Product Quantization (PQ) compresses high-dimensional dense vector embeddings by dividing the D-dimensional space "
            "into M orthogonal sub-vectors of dimension D/M. For each sub-space, k-means clustering identifies K centroids (typically K=256). "
            "Each continuous floating-point sub-vector is then quantized into a single 1-byte centroid code, reducing a 1024-dimensional "
            "float32 embedding (4096 bytes) into an ultra-compact 64-byte code while supporting asymmetric distance computation."
        ),
    },

    # --- 10 CODE PASSAGES ---
    {
        "id": "code-01",
        "doc_name": "circuit_breaker.py",
        "page": 1,
        "doc_type": "code",
        "text": (
            "class CircuitBreaker:\n"
            "    def allow_request(self) -> bool:\n"
            "        if self.metrics.state == CircuitState.OPEN:\n"
            "            elapsed = time.time() - self.metrics.last_state_change\n"
            "            if elapsed >= self.config.cooldown_seconds:\n"
            "                self._transition_to(CircuitState.HALF_OPEN, 'Cooldown elapsed')\n"
            "                return True\n"
            "            return False\n"
            "        return True\n"
            "    def record_success(self) -> None:\n"
            "        if self.metrics.state == CircuitState.HALF_OPEN:\n"
            "            self.metrics.consecutive_successes += 1\n"
            "            if self.metrics.consecutive_successes >= self.config.canary_success_threshold:\n"
            "                self._transition_to(CircuitState.CLOSED, 'Canary threshold reached')\n"
        ),
    },
    {
        "id": "code-02",
        "doc_name": "session_store.py",
        "page": 1,
        "doc_type": "code",
        "text": (
            "class SQLiteSessionStore:\n"
            "    def _init_db(self):\n"
            "        with sqlite3.connect(self.db_path) as conn:\n"
            "            conn.execute('PRAGMA journal_mode=WAL;')\n"
            "            conn.execute('PRAGMA synchronous=NORMAL;')\n"
            "            conn.execute('PRAGMA busy_timeout=5000;')\n"
            "            conn.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (\n"
            "                session_id TEXT PRIMARY KEY,\n"
            "                title TEXT NOT NULL,\n"
            "                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
            "            );''')\n"
        ),
    },
    {
        "id": "code-03",
        "doc_name": "router.py",
        "page": 1,
        "doc_type": "code",
        "text": (
            "async def stream_response(self, prompt, requested_model, attachments=None):\n"
            "    primary, backup, model_id = self._resolve_providers(requested_model, attachments)\n"
            "    primary_breaker = self.get_breaker(primary.provider_name)\n"
            "    if not primary_breaker.allow_request():\n"
            "        if backup and backup.is_available:\n"
            "            async for token in backup.stream_response(prompt):\n"
            "                yield token, backup.provider_name, getattr(backup, 'model', 'fallback'), True\n"
            "            return\n"
            "        raise QuotaExhaustedException('Primary open and no backup available')\n"
        ),
    },
    {
        "id": "code-04",
        "doc_name": "ArtifactCanvas.tsx",
        "page": 1,
        "doc_type": "code",
        "text": (
            "export function computeMyersDiff(oldStr: string, newStr: string): DiffLine[] {\n"
            "  const oldLines = oldStr.split('\\n');\n"
            "  const newLines = newStr.split('\\n');\n"
            "  // LCS dynamic programming table for line-by-line diff\n"
            "  const dp: number[][] = Array(oldLines.length + 1).fill(0)\n"
            "    .map(() => Array(newLines.length + 1).fill(0));\n"
            "  for (let i = 0; i < oldLines.length; i++) {\n"
            "    for (let j = 0; j < newLines.length; j++) {\n"
            "      if (oldLines[i] === newLines[j]) dp[i+1][j+1] = dp[i][j] + 1;\n"
            "      else dp[i+1][j+1] = Math.max(dp[i+1][j], dp[i][j+1]);\n"
            "    }\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "id": "code-05",
        "doc_name": "rate_limiter.py",
        "page": 1,
        "doc_type": "code",
        "text": (
            "class TokenBucketLimiter:\n"
            "    def __init__(self, rate_per_minute: int = 60, burst_capacity: int = 10):\n"
            "        self.rate = rate_per_minute / 60.0\n"
            "        self.capacity = burst_capacity\n"
            "        self.tokens = float(burst_capacity)\n"
            "        self.last_check = time.monotonic()\n"
            "    def acquire(self, cost: float = 1.0) -> bool:\n"
            "        now = time.monotonic()\n"
            "        elapsed = now - self.last_check\n"
            "        self.last_check = now\n"
            "        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)\n"
            "        if self.tokens >= cost:\n"
            "            self.tokens -= cost\n"
            "            return True\n"
            "        return False\n"
        ),
    },
    {
        "id": "code-06",
        "doc_name": "web_search_service.py",
        "page": 1,
        "doc_type": "code",
        "text": (
            "def extract_ddg_results(html_content: str, max_results: int = 5) -> List[SearchResult]:\n"
            "    soup = BeautifulSoup(html_content, 'html.parser')\n"
            "    results = []\n"
            "    for result in soup.find_all('div', class_='result__body')[:max_results]:\n"
            "        title_tag = result.find('a', class_='result__a')\n"
            "        snippet_tag = result.find('a', class_='result__snippet')\n"
            "        if title_tag and title_tag.get('href'):\n"
            "            raw_url = title_tag['href']\n"
            "            actual_url = unquote(raw_url.split('uddg=')[1].split('&')[0]) if 'uddg=' in raw_url else raw_url\n"
            "            results.append(SearchResult(title=title_tag.get_text(), url=actual_url, snippet=snippet_tag.get_text()))\n"
            "    return results\n"
        ),
    },
    {
        "id": "code-07",
        "doc_name": "vector_similarity.py",
        "page": 1,
        "doc_type": "code",
        "text": (
            "def batch_cosine_similarity(query_vector: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:\n"
            "    # Vectorized cosine similarity computation with L2 normalization\n"
            "    query_norm = np.linalg.norm(query_vector)\n"
            "    if query_norm == 0:\n"
            "        return np.zeros(doc_matrix.shape[0])\n"
            "    doc_norms = np.linalg.norm(doc_matrix, axis=1)\n"
            "    doc_norms[doc_norms == 0] = 1.0\n"
            "    dot_products = np.dot(doc_matrix, query_vector)\n"
            "    return dot_products / (query_norm * doc_norms)\n"
        ),
    },
    {
        "id": "code-08",
        "doc_name": "sse_parser.ts",
        "page": 1,
        "doc_type": "code",
        "text": (
            "export async function* parseSSEStream(reader: ReadableStreamDefaultReader<Uint8Array>): AsyncGenerator<SSEMessage> {\n"
            "  const decoder = new TextDecoder('utf-8');\n"
            "  let buffer = '';\n"
            "  while (true) {\n"
            "    const { value, done } = await reader.read();\n"
            "    if (done) break;\n"
            "    buffer += decoder.decode(value, { stream: true });\n"
            "    const lines = buffer.split('\\n\\n');\n"
            "    buffer = lines.pop() || '';\n"
            "    for (const chunk of lines) {\n"
            "      if (chunk.startsWith('data: ')) yield JSON.parse(chunk.slice(6));\n"
            "    }\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "id": "code-09",
        "doc_name": "groq_provider.py",
        "page": 1,
        "doc_type": "code",
        "text": (
            "async for line in response.aiter_lines():\n"
            "    if not line.startswith('data: '): continue\n"
            "    raw = line[6:].strip()\n"
            "    if raw == '[DONE]': break\n"
            "    delta = json.loads(raw)['choices'][0]['delta']\n"
            "    text = delta.get('content')\n"
            "    if text:\n"
            "        while text:\n"
            "            if not in_thinking:\n"
            "                if '<think>' in text:\n"
            "                    before, text = text.split('<think>', 1)\n"
            "                    if before: yield before\n"
            "                    in_thinking = True\n"
            "                else: yield text; break\n"
            "            else:\n"
            "                if '</think>' in text: _, text = text.split('</think>', 1); in_thinking = False\n"
            "                else: break\n"
        ),
    },
    {
        "id": "code-10",
        "doc_name": "ArtifactContext.tsx",
        "page": 1,
        "doc_type": "code",
        "text": (
            "const updateArtifact = (newContent: string) => {\n"
            "  setActiveArtifact(prev => {\n"
            "    if (!prev) return null;\n"
            "    const newVersionNum = prev.versions.length + 1;\n"
            "    const newVersionItem: ArtifactVersion = {\n"
            "      versionNumber: newVersionNum,\n"
            "      content: newContent,\n"
            "      timestamp: new Date().toISOString()\n"
            "    };\n"
            "    return {\n"
            "      ...prev,\n"
            "      content: newContent,\n"
            "      activeVersionIndex: newVersionNum - 1,\n"
            "      versions: [...prev.versions, newVersionItem]\n"
            "    };\n"
            "  });\n"
            "};\n"
        ),
    },

    # --- 10 MARKDOWN SPECIFICATION PASSAGES ---
    {
        "id": "md-01",
        "doc_name": "api_problem_details_rfc7807.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## RFC 7807 Problem Details Standard\n"
            "HTTP error responses in JRU conform to RFC 7807 problem details specification:\n"
            "- `type`: URI identifying the error problem type (e.g. `https://errors.jru.ai/rate-limit-exceeded`)\n"
            "- `title`: Short human-readable summary of problem (e.g. `Too Many Requests`)\n"
            "- `status`: HTTP status code matching transport response (e.g. `429`)\n"
            "- `detail`: Context-specific explanation of what triggered the failure\n"
            "- `instance`: Unique request tracing URI for audit correlation\n"
        ),
    },
    {
        "id": "md-02",
        "doc_name": "database_schema_specification.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## SQLite Chat Storage Schema\n"
            "The chat persistence layer maintains two relational tables:\n"
            "1. `chat_sessions`: `session_id` (TEXT PK), `title` (TEXT), `created_at` (TIMESTAMP), `updated_at` (TIMESTAMP)\n"
            "2. `chat_messages`: `message_id` (TEXT PK), `session_id` (TEXT FK REFERENCES chat_sessions),\n"
            "   `role` (TEXT CHECK(role IN ('user', 'assistant', 'system'))),\n"
            "   `content` (TEXT NOT NULL), `model` (TEXT), `provider` (TEXT),\n"
            "   `tokens_per_sec` (REAL), `duration_sec` (REAL), `created_at` (TIMESTAMP)\n"
        ),
    },
    {
        "id": "md-03",
        "doc_name": "security_isolation_policy.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## Client-Side Sandbox CSP Directives\n"
            "Artifact rendering iframes enforce strict isolation policies:\n"
            "- `sandbox=\"allow-scripts allow-modals\"`: Blocks top-level navigation, same-origin access, and popups\n"
            "- Omission of `allow-same-origin`: Ensures iframe runs in an opaque origin (`null`), preventing access\n"
            "  to host `window.parent.localStorage`, `window.parent.document.cookie`, or host session tokens\n"
            "- Content Security Policy header: `default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline'`\n"
        ),
    },
    {
        "id": "md-04",
        "doc_name": "rate_limiting_tier_quotas.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## Provider Rate Limit Safeguards\n"
            "To prevent free-tier developer quota exhaustion, upstream rate caps are enforced:\n"
            "| Provider | Max Requests / Min | Max Tokens / Min | Daily Quota |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| Groq (Llama 3.3 70B) | 30 req/min | 6,000 tok/min | 14,400 req/day |\n"
            "| Groq (GPT-OSS 120B) | 30 req/min | 8,000 tok/min | 14,400 req/day |\n"
            "| Google Gemini 3.7 Flash | 15 req/min | 1,000,000 tok/min | 1,500 req/day |\n"
            "| Local Ollama | Unlimited | Unlimited | Unlimited |\n"
        ),
    },
    {
        "id": "md-05",
        "doc_name": "sse_streaming_protocol_spec.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## Server-Sent Events Protocol Specifications\n"
            "Chat completion stream emits three structured event types:\n"
            "1. `event: message` with payload `{\"type\":\"token\",\"content\":\"str\",\"provider\":\"groq\",\"failover\":false}`\n"
            "2. `event: message` with payload `{\"type\":\"citation\",\"title\":\"str\",\"url\":\"str\"}` (when web grounding active)\n"
            "3. `event: done` with payload `{\"type\":\"finish\",\"finish_reason\":\"stop\",\"total_tokens\":124}`\n"
            "Clients must send `Accept: text/event-stream` and handle reconnection on network drops.\n"
        ),
    },
    {
        "id": "md-06",
        "doc_name": "docker_production_deployment.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## Multi-Stage Production Dockerfile\n"
            "The container image uses a 2-stage build for minimal surface area:\n"
            "- Stage 1 (Builder): `python:3.12-slim`, installs GCC, compiles C extensions, builds wheels in `/wheels`\n"
            "- Stage 2 (Runner): `python:3.12-slim`, non-root user `appuser` (UID 10001), installs wheels from `/wheels`\n"
            "- Runtime command: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop`\n"
            "- Healthcheck: `CMD curl --fail http://localhost:8000/api/v1/health || exit 1`\n"
        ),
    },
    {
        "id": "md-07",
        "doc_name": "ai_persona_guidelines.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## Persona Behavioral Guidelines\n"
            "Personas modulate prompt engineering behavior via dynamic system instruction injection:\n"
            "- **Senior Systems Architect**: Mandates end-to-end component diagrams, ASCII sequence charts, and failure mode tables\n"
            "- **Elite Competitive Coder**: Enforces idiomatic zero-fluff code, exact Big-O space and time complexity annotations\n"
            "- **Research Scientist**: Enforces academic tone, empirical citations, and LaTeX mathematical formalisms\n"
            "- **General Assistant**: Balanced, polite brevity with structured markdown headings\n"
        ),
    },
    {
        "id": "md-08",
        "doc_name": "incident_response_runbook.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## Disaster Recovery & Failover Runbook\n"
            "When primary provider reports HTTP 429 quota exhaustion or 503 outage:\n"
            "1. Circuit breaker trips from `CLOSED` to `OPEN` after 3 consecutive failures\n"
            "2. Automated failover routes all subsequent traffic directly to secondary provider (Gemini Flash)\n"
            "3. Cooldown probe timer (15 seconds) enters `HALF_OPEN` state\n"
            "4. Canary probes verify recovery; 2 consecutive successes transition state back to `CLOSED`\n"
            "5. If all providers fail, an RFC 7807 503 response is returned to the client\n"
        ),
    },
    {
        "id": "md-09",
        "doc_name": "artifact_versioning_specification.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## Multi-Turn Artifact Versioning Architecture\n"
            "Artifacts maintain immutable revision trees across multi-turn conversational edits:\n"
            "- Every code or UI update appends a new `vN` version object to the `versions` array\n"
            "- Revision history strip in `ArtifactCanvas.tsx` renders version selector pills (`v1`, `v2`, `v3`...)\n"
            "- Active version index allows viewing historical revisions without overwriting current head\n"
            "- Rollback action (`Restore vX`) promotes historical revision as a new head revision\n"
        ),
    },
    {
        "id": "md-10",
        "doc_name": "frontend_design_tokens.md",
        "page": 1,
        "doc_type": "markdown",
        "text": (
            "## Glassmorphism UI Design System\n"
            "Frontend styling utilizes custom CSS custom properties for dark/light themes:\n"
            "- `--bg-workspace`: Base background `#0a0a0c` (dark) / `#f8fafc` (light)\n"
            "- `--bg-surface`: Card background `rgba(255, 255, 255, 0.05)` with `backdrop-filter: blur(20px)`\n"
            "- `--border-hairline`: Subtle border `rgba(255, 255, 255, 0.08)`\n"
            "- `--btn-send`: Accent blue `#0071e3` with transition `all 150ms cubic-bezier(0.16, 1, 0.3, 1)`\n"
            "- User message bubble: `border-radius: 18px; border-bottom-right-radius: 4px; padding: 7px 13px;`\n"
        ),
    },
]

# 30 Ground-Truth Questions with target chunk IDs
EVAL_QUESTIONS: List[Dict[str, Any]] = [
    # PDF Questions (1-10)
    {
        "id": "q-01",
        "doc_type": "pdf",
        "query": "What is the randomized election timer range used by followers in Raft consensus?",
        "target_chunk_id": "pdf-01",
        "expected_answer_keywords": ["150ms", "300ms", "election timer", "AppendEntries"],
    },
    {
        "id": "q-02",
        "doc_type": "pdf",
        "query": "What does the efConstruction parameter control in HNSW vector index graphs?",
        "target_chunk_id": "pdf-02",
        "expected_answer_keywords": ["efConstruction", "candidate list", "graph indexing", "HNSW"],
    },
    {
        "id": "q-03",
        "doc_type": "pdf",
        "query": "At what capacity threshold is an LSM MemTable frozen and flushed as a Level-0 SSTable?",
        "target_chunk_id": "pdf-03",
        "expected_answer_keywords": ["64MB", "MemTable", "SSTable", "Level-0"],
    },
    {
        "id": "q-04",
        "doc_type": "pdf",
        "query": "Which kernel routine handles page fault interrupts to allocate physical frames in Linux x86-64?",
        "target_chunk_id": "pdf-04",
        "expected_answer_keywords": ["handle_mm_fault", "page fault", "#PF", "PTE"],
    },
    {
        "id": "q-05",
        "doc_type": "pdf",
        "query": "How does TLS 1.3 achieve 1-RTT handshake latency with ephemeral Diffie-Hellman?",
        "target_chunk_id": "pdf-05",
        "expected_answer_keywords": ["ClientHello", "ServerHello", "key share", "ECDHE"],
    },
    {
        "id": "q-06",
        "doc_type": "pdf",
        "query": "What triggers a circuit breaker to trip from CLOSED to OPEN in microservices?",
        "target_chunk_id": "pdf-06",
        "expected_answer_keywords": ["failure rate", "sliding window", "threshold", "trip"],
    },
    {
        "id": "q-07",
        "doc_type": "pdf",
        "query": "How are dead row versions identified and reclaimed in PostgreSQL MVCC?",
        "target_chunk_id": "pdf-07",
        "expected_answer_keywords": ["xmin", "xmax", "VACUUM", "Snapshot Isolation"],
    },
    {
        "id": "q-08",
        "doc_type": "pdf",
        "query": "What happens when a CPU core writes to a cache line in Shared state under MESI?",
        "target_chunk_id": "pdf-08",
        "expected_answer_keywords": ["Invalidating", "Modified", "Shared", "Invalid"],
    },
    {
        "id": "q-09",
        "doc_type": "pdf",
        "query": "What are the four fields comprising the W3C traceparent HTTP header in OpenTelemetry?",
        "target_chunk_id": "pdf-09",
        "expected_answer_keywords": ["version", "trace-id", "parent-span-id", "trace-flags"],
    },
    {
        "id": "q-10",
        "doc_type": "pdf",
        "query": "How does Product Quantization compress 1024-dimensional float32 embeddings to 64 bytes?",
        "target_chunk_id": "pdf-10",
        "expected_answer_keywords": ["sub-vectors", "centroids", "k-means", "quantized"],
    },

    # Code Questions (11-20)
    {
        "id": "q-11",
        "doc_type": "code",
        "query": "How does the CircuitBreaker class check if cooldown has elapsed to enter HALF_OPEN?",
        "target_chunk_id": "code-01",
        "expected_answer_keywords": ["cooldown_seconds", "HALF_OPEN", "allow_request", "elapsed"],
    },
    {
        "id": "q-12",
        "doc_type": "code",
        "query": "What SQLite PRAGMA journal mode is set when initializing the session store database?",
        "target_chunk_id": "code-02",
        "expected_answer_keywords": ["journal_mode=WAL", "synchronous=NORMAL", "busy_timeout"],
    },
    {
        "id": "q-13",
        "doc_type": "code",
        "query": "Where in stream_response is fast-path failover triggered if primary breaker is open?",
        "target_chunk_id": "code-03",
        "expected_answer_keywords": ["allow_request", "backup.stream_response", "QuotaExhaustedException"],
    },
    {
        "id": "q-14",
        "doc_type": "code",
        "query": "How does computeMyersDiff build the dynamic programming table for line diffs in TypeScript?",
        "target_chunk_id": "code-04",
        "expected_answer_keywords": ["oldLines", "newLines", "dp", "computeMyersDiff"],
    },
    {
        "id": "q-15",
        "doc_type": "code",
        "query": "How are tokens refilled based on elapsed time in TokenBucketLimiter acquire?",
        "target_chunk_id": "code-05",
        "expected_answer_keywords": ["time.monotonic", "tokens", "elapsed", "capacity"],
    },
    {
        "id": "q-16",
        "doc_type": "code",
        "query": "How is DuckDuckGo URL unquoted from uddg parameter in extract_ddg_results?",
        "target_chunk_id": "code-06",
        "expected_answer_keywords": ["uddg=", "unquote", "result__a", "result__snippet"],
    },
    {
        "id": "q-17",
        "doc_type": "code",
        "query": "How does batch_cosine_similarity handle zero vector norms in NumPy?",
        "target_chunk_id": "code-07",
        "expected_answer_keywords": ["np.linalg.norm", "dot_products", "doc_norms", "query_norm"],
    },
    {
        "id": "q-18",
        "doc_type": "code",
        "query": "How does parseSSEStream split multi-line chunks in the buffer?",
        "target_chunk_id": "code-08",
        "expected_answer_keywords": ["split('\\n\\n')", "data: ", "JSON.parse", "TextDecoder"],
    },
    {
        "id": "q-19",
        "doc_type": "code",
        "query": "How does the groq streaming parser suppress internal <think> blocks across tokens?",
        "target_chunk_id": "code-09",
        "expected_answer_keywords": ["<think>", "</think>", "in_thinking", "split"],
    },
    {
        "id": "q-20",
        "doc_type": "code",
        "query": "How is a new version appended to the artifact versions array in updateArtifact?",
        "target_chunk_id": "code-10",
        "expected_answer_keywords": ["versionNumber", "newVersionNum", "versions", "ArtifactVersion"],
    },

    # Markdown Questions (21-30)
    {
        "id": "q-21",
        "doc_type": "markdown",
        "query": "What five attributes define an RFC 7807 problem details error payload?",
        "target_chunk_id": "md-01",
        "expected_answer_keywords": ["type", "title", "status", "detail", "instance"],
    },
    {
        "id": "q-22",
        "doc_type": "markdown",
        "query": "What columns and constraints define the SQLite chat_messages table schema?",
        "target_chunk_id": "md-02",
        "expected_answer_keywords": ["message_id", "session_id", "role", "tokens_per_sec"],
    },
    {
        "id": "q-23",
        "doc_type": "markdown",
        "query": "Why is allow-same-origin deliberately omitted from the iframe sandbox policy?",
        "target_chunk_id": "md-03",
        "expected_answer_keywords": ["allow-same-origin", "opaque origin", "localStorage", "cookie"],
    },
    {
        "id": "q-24",
        "doc_type": "markdown",
        "query": "What are the rate limit quotas per minute for Groq and Google Gemini 3.7 Flash?",
        "target_chunk_id": "md-04",
        "expected_answer_keywords": ["30 req/min", "15 req/min", "6,000 tok/min", "1,000,000"],
    },
    {
        "id": "q-25",
        "doc_type": "markdown",
        "query": "What are the three event types emitted by the JRU Server-Sent Events chat completion stream?",
        "target_chunk_id": "md-05",
        "expected_answer_keywords": ["event: message", "event: done", "token", "citation"],
    },
    {
        "id": "q-26",
        "doc_type": "markdown",
        "query": "What non-root user UID is configured in stage 2 of the production Dockerfile?",
        "target_chunk_id": "md-06",
        "expected_answer_keywords": ["appuser", "10001", "python:3.12-slim", "uvicorn"],
    },
    {
        "id": "q-27",
        "doc_type": "markdown",
        "query": "What are the behavioral mandates of the Senior Systems Architect persona?",
        "target_chunk_id": "md-07",
        "expected_answer_keywords": ["component diagrams", "ASCII", "failure mode", "Architect"],
    },
    {
        "id": "q-28",
        "doc_type": "markdown",
        "query": "What is the cooldown probe delay before transitioning from OPEN to HALF_OPEN in the incident runbook?",
        "target_chunk_id": "md-08",
        "expected_answer_keywords": ["15 seconds", "HALF_OPEN", "canary", "CLOSED"],
    },
    {
        "id": "q-29",
        "doc_type": "markdown",
        "query": "What user action promotes a historical revision back to the latest head in ArtifactCanvas?",
        "target_chunk_id": "md-09",
        "expected_answer_keywords": ["Restore vX", "activeVersionIndex", "versions", "revision"],
    },
    {
        "id": "q-30",
        "doc_type": "markdown",
        "query": "What CSS border-radius and padding values are assigned to user message bubbles?",
        "target_chunk_id": "md-10",
        "expected_answer_keywords": ["18px", "border-bottom-right-radius: 4px", "7px 13px"],
    },
]
