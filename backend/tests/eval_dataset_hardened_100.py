"""100-Query Hardened Evaluation Dataset for JRU RAG Quality & Ablation Studies.

Contains 50 diverse document passages spanning:
  1. Distributed Systems & Core OS Architecture
  2. Jruva AI Platform Engineering & Telemetry Specs
  3. Jharkhand Rai University Academic Domain Guidelines
  4. Adversarial Distractor Passages (Lexical traps)

Contains 100 annotated test queries categorized across:
  - direct_factual (25)
  - terminology_mismatch (15)
  - multi_section (15)
  - code_documentation (15)
  - ambiguous (10)
  - unanswerable (10)
  - adversarial_distractor (10)
"""
from typing import Any, Dict, List, Optional

PASSAGES: List[Dict[str, Any]] = [
    # --- Distributed Systems & OS Internals (1-15) ---
    {
        "id": "doc-sys-01",
        "doc_name": "raft_consensus_specification.md",
        "doc_type": "systems",
        "text": (
            "Raft consensus protocol governs leader election through randomized election timers "
            "ranging between 150ms and 300ms. When a follower node detects a heartbeat timeout "
            "due to absence of AppendEntries RPCs, it transitions to candidate state, increments term, "
            "and requests votes. A candidate wins an election upon receiving votes from a majority of "
            "cluster nodes for that specific term counter."
        ),
    },
    {
        "id": "doc-sys-02",
        "doc_name": "hnsw_vector_indexing.md",
        "doc_type": "systems",
        "text": (
            "Hierarchical Navigable Small World (HNSW) graphs construct multi-layered geometric graphs "
            "for vector approximate nearest neighbor (ANN) search. Upper layers maintain sparse long-range "
            "skip-links for logarithmic routing, while layer 0 hosts the dense k-nearest neighbors graph. "
            "Indexing complexity is governed by M (maximum degree per vertex) and efConstruction (dynamic "
            "candidate exploration list depth during insertion)."
        ),
    },
    {
        "id": "doc-sys-03",
        "doc_name": "lsm_tree_storage_engine.md",
        "doc_type": "systems",
        "text": (
            "Log-Structured Merge-tree storage architectures transform random disk writes into sequential "
            "disk I/O. Incoming mutations append to a Write-Ahead Log (WAL) and an in-memory MemTable. Once "
            "the MemTable reaches capacity (configured at 64MB by default), it flushes to disk as an immutable "
            "Level-0 Sorted String Table (SSTable). Background leveled compaction merges overlapping key ranges."
        ),
    },
    {
        "id": "doc-sys-04",
        "doc_name": "linux_virtual_memory_paging.md",
        "doc_type": "systems",
        "text": (
            "Linux x86-64 virtual memory management uses 4-level or 5-level hierarchical page tables consisting "
            "of PGD, P4D, PUD, PMD, and PTE structures. Page faults (#PF exception 14) invoke handle_mm_fault() "
            "to allocate physical frames via the buddy allocator and establish PTE mappings with appropriate "
            "read/write/user protection bits, followed by TLB invalidation."
        ),
    },
    {
        "id": "doc-sys-05",
        "doc_name": "tls_13_handshake_crypto.md",
        "doc_type": "systems",
        "text": (
            "TLS 1.3 deprecates static RSA and custom Diffie-Hellman parameters, requiring Ephemeral "
            "Diffie-Hellman (ECDHE or X25519) to guarantee forward secrecy. The client sends its key share "
            "directly in the ClientHello, achieving a 1-RTT handshake. Zero round-trip resumption (0-RTT) "
            "uses pre-shared keys (PSK) with early data, albeit vulnerable to replay attacks."
        ),
    },
    {
        "id": "doc-sys-06",
        "doc_name": "circuit_breaker_resilience.md",
        "doc_type": "systems",
        "text": (
            "The Circuit Breaker pattern prevents cascade failure across downstream microservices. "
            "In CLOSED state, error counters track consecutive failures or sliding window error percentages. "
            "When the trip threshold (e.g., 3 consecutive failures) is reached, the breaker trips to OPEN, "
            "failing fast immediately without dispatching upstream calls. After a cooldown delay (e.g., 15s), "
            "it enters HALF_OPEN to dispatch probe requests."
        ),
    },
    {
        "id": "doc-sys-07",
        "doc_name": "mvcc_postgresql_concurrency.md",
        "doc_type": "systems",
        "text": (
            "PostgreSQL implements Multi-Version Concurrency Control (MVCC) via heap tuple headers containing "
            "xmin (inserting transaction ID) and xmax (deleting/updating transaction ID). Snapshot isolation "
            "evaluates tuple visibility against active transaction snapshots. Dead tuples invisible to all "
            "active snapshots are pruned and reclaimed into free-space maps by the autovacuum daemon."
        ),
    },
    {
        "id": "doc-sys-08",
        "doc_name": "mesi_cache_coherence_protocol.md",
        "doc_type": "systems",
        "text": (
            "The MESI cache coherence protocol defines four states for multicore CPU cache lines: "
            "Modified (dirty, exclusive), Exclusive (clean, single core), Shared (clean, multiple cores), "
            "and Invalid (stale). When a core attempts to write to a line in Shared state, it must broadcast "
            "a Read-with-Intent-to-Modify (RWITM) bus transaction, forcing other cores to transition to Invalid."
        ),
    },
    {
        "id": "doc-sys-09",
        "doc_name": "opentelemetry_w3c_trace_context.md",
        "doc_type": "systems",
        "text": (
            "The W3C Trace Context standard standardizes distributed tracing through the traceparent HTTP header. "
            "Format: `version-traceid-parentid-traceflags` (e.g., `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`). "
            "The 16-byte trace-id identifies the distributed transaction, the 8-byte parent-id identifies the parent span, "
            "and trace-flags bit 0 indicates whether the trace is sampled."
        ),
    },
    {
        "id": "doc-sys-10",
        "doc_name": "product_quantization_compression.md",
        "doc_type": "systems",
        "text": (
            "Product Quantization (PQ) compresses dense high-dimensional vectors by decomposing a d-dimensional "
            "space into m orthogonal sub-vectors of dimension d/m. Each sub-vector is quantized against k centroids "
            "computed via k-means. For d=1024 and m=64 with k=256, a 4096-byte float32 vector is compressed into "
            "64 bytes of 8-bit centroid indices, enabling asymmetric distance computation via lookup tables."
        ),
    },
    {
        "id": "doc-sys-11",
        "doc_name": "epoll_io_multiplexing.md",
        "doc_type": "systems",
        "text": (
            "Linux epoll provides scalable I/O event notification with O(1) performance using an internal red-black "
            "tree for watched file descriptors and a doubly-linked ready list for active events. Edge-triggered "
            "(EPOLLET) mode delivers events only when file descriptor state changes, requiring consumers to read "
            "or write until receiving EAGAIN or EWOULDBLOCK."
        ),
    },
    {
        "id": "doc-sys-12",
        "doc_name": "sqlite_wal_locking_internals.md",
        "doc_type": "systems",
        "text": (
            "In SQLite Write-Ahead Logging (WAL) mode, readers do not block writers and writers do not block "
            "readers. Modifications append to a separate `-wal` file. Readers use a shared-memory `-shm` index "
            "to find the latest valid frame for their transaction snapshot. Checkpointing transfers frames from the "
            "WAL back into the database file, governed by PRAGMA wal_autocheckpoint."
        ),
    },
    {
        "id": "doc-sys-13",
        "doc_name": "b_tree_database_indexing.md",
        "doc_type": "systems",
        "text": (
            "B+ trees store all key-payload associations exclusively in leaf nodes linked sequentially for range scans, "
            "while internal nodes store navigation keys and child pointers. Page splits occur when an insertion "
            "exceeds the 8KB or 16KB page capacity, dividing the node evenly and propagating the median key upwards."
        ),
    },
    {
        "id": "doc-sys-14",
        "doc_name": "paxos_distributed_consensus.md",
        "doc_type": "systems",
        "text": (
            "Classic Paxos achieves consensus across asynchronous networks through two phases: Phase 1 (Prepare/Promise) "
            "and Phase 2 (Accept/Accepted). A proposer must acquire promises from a majority quorum with proposal number n "
            "before it can submit value v in the Accept phase. Multi-Paxos optimizes this by electing a stable leader to "
            "bypass Phase 1 for subsequent log instances."
        ),
    },
    {
        "id": "doc-sys-15",
        "doc_name": "consistent_hashing_ring.md",
        "doc_type": "systems",
        "text": (
            "Consistent hashing maps nodes and keys onto a circular keyspace ring (typically 2^32 or 2^128). "
            "When a cluster node joins or departs, only k/N keys require migration where k is total keys and N is "
            "node count. Virtual nodes (vnodes) distribute each physical server across dozens of points on the ring "
            "to prevent non-uniform data skew and hotspotting."
        ),
    },

    # --- Jruva AI Platform Engineering & Specs (16-30) ---
    {
        "id": "doc-jruva-01",
        "doc_name": "router_architecture_spec.md",
        "doc_type": "jruva_spec",
        "text": (
            "Jruva's model router classifies user prompts into task categories: general_chat, deep_coding, "
            "mathematical_reasoning, document_vision, and academic_rag. The router orchestrates tiered "
            "provider dispatch (Tier A: Groq Llama-3.3-70B / Gemini-2.0-Flash; Tier B: OpenRouter LFM-2.5 / "
            "Qwen specialist) coupled with circuit breakers and token-bucket rate limiters."
        ),
    },
    {
        "id": "doc-jruva-02",
        "doc_name": "fastpath_optimization_rules.md",
        "doc_type": "jruva_spec",
        "text": (
            "Deterministic fastpaths bypass upstream LLM inference for canned queries, greetings, help commands, "
            "and identity queries (e.g., 'who are you', 'what is jruva'). Fastpath responses execute in sub-10ms "
            "with zero token cost and record `cache_hit=False`, `fastpath=True`, and `provider=fastpath` in telemetry."
        ),
    },
    {
        "id": "doc-jruva-03",
        "doc_name": "response_cache_lru_invalidation.md",
        "doc_type": "jruva_spec",
        "text": (
            "Jruva employs a dual-layer caching strategy: an exact-match LRU cache with SHA-256 normalized prompt keys "
            "and a semantic similarity cache. Cached responses return instant SSE streams with `cache_hit=True`. "
            "Cache entries expire after a TTL of 3600 seconds or upon explicit manual invalidation."
        ),
    },
    {
        "id": "doc-jruva-04",
        "doc_name": "rag_hybrid_retrieval_pipeline.md",
        "doc_type": "jruva_spec",
        "text": (
            "The hybrid RAG retrieval pipeline merges dense vector embeddings (BAAI/bge-small-en-v1.5) with "
            "sparse BM25 lexical token rankings using Reciprocal Rank Fusion (RRF, constant k=60). Candidate "
            "passages (top 15) are passed to the FlashRank neural cross-encoder (ms-marco-TinyBERT-L-2-v2) for "
            "precision re-scoring, selecting top-k=3 context passages for LLM synthesis."
        ),
    },
    {
        "id": "doc-jruva-05",
        "doc_name": "session_authorization_tenant_model.md",
        "doc_type": "jruva_spec",
        "text": (
            "User sessions are bound to authenticated JWT user identities (`user_id`). Anonymous callers are "
            "assigned an ephemeral server-generated UUID. All session turn mutations, reads, and deletes verify "
            "ownership via `session.user_id == current_user.id`. Any unauthorized access attempt terminates "
            "immediately with HTTP 403 Forbidden."
        ),
    },
    {
        "id": "doc-jruva-06",
        "doc_name": "qdrant_tenant_isolation_filter.md",
        "doc_type": "jruva_spec",
        "text": (
            "Every vector search in Qdrant enforces an explicit tenant filter on the `owner_id` keyword payload index. "
            "Queries from tenant T are scoped strictly to `owner_id == T` OR `owner_id == 'public'`. Private document "
            "vectors from tenant A cannot be retrieved or scored by tenant B, preventing cross-tenant information leakage."
        ),
    },
    {
        "id": "doc-jruva-07",
        "doc_name": "code_execution_security_policy.md",
        "doc_type": "jruva_spec",
        "text": (
            "Arbitrary Python `exec()` execution is strictly disabled in public production mode. The code intelligence "
            "service returns `status='execution_unavailable'` for runtime execution requests while preserving safe "
            "static analysis (AST parsing, cyclomatic complexity calculation, and algorithmic template suggestions). "
            "Unsafe execution is restricted to explicit local development (`ENABLE_UNSAFE_LOCAL_CODE_EXECUTION=true`)."
        ),
    },
    {
        "id": "doc-jruva-08",
        "doc_name": "sse_streaming_protocol_specification.md",
        "doc_type": "jruva_spec",
        "text": (
            "Server-Sent Events (SSE) stream tokens using standard event schemas: `event: token` containing "
            "`{\"content\": \"...\", \"provider\": \"...\"}`, followed by optional `event: citation` payloads, "
            "and terminated by `event: done` with finish reasons and token usage metrics. Unclosed `<think>` tags "
            "are sanitized before emission to prevent chain-of-thought prompt leakage."
        ),
    },
    {
        "id": "doc-jruva-09",
        "doc_name": "provider_failover_telemetry.md",
        "doc_type": "jruva_spec",
        "text": (
            "When the primary LLM provider raises HTTP 429 (Rate Limit Exceeded) or 503 (Service Unavailable), "
            "the circuit breaker increments failure count. If open, the router instantly switches to the secondary "
            "provider. Telemetry records `fallback_occurred=True`, `fallback_reason='http_429'`, and latency metrics."
        ),
    },
    {
        "id": "doc-jruva-10",
        "doc_name": "image_generation_service_urls.md",
        "doc_type": "jruva_spec",
        "text": (
            "Image generation outputs and upscaled variations are saved to disk under `static/generated/` and "
            "`static/demo/`. APIs return environment-aware relative URLs (e.g., `/static/generated/img_123.webp`) "
            "rather than hardcoded localhost ports, ensuring correct resolution across local dev, Docker, and jruva.in."
        ),
    },
    {
        "id": "doc-jruva-11",
        "doc_name": "jwt_authentication_token_security.md",
        "doc_type": "jruva_spec",
        "text": (
            "Authentication uses HS256 JWT tokens with 15-minute access token expiry and 7-day refresh tokens. "
            "Production startup mandates a cryptographically secure `JWT_SECRET_KEY` env var; missing keys raise "
            "a fatal `RuntimeError`. Passwords are hashed with bcrypt (cost factor 12)."
        ),
    },
    {
        "id": "doc-jruva-12",
        "doc_name": "calculus_engine_symbolic_math.md",
        "doc_type": "jruva_spec",
        "text": (
            "The calculus engine parses mathematical expressions into SymPy AST representations, executing symbolic "
            "derivatives, definite/indefinite integrals, Taylor series expansions, and limit calculations. Output "
            "is formatted as LaTeX math blocks with step-by-step explanatory prose."
        ),
    },
    {
        "id": "doc-jruva-13",
        "doc_name": "ocr_multimodal_vision_service.md",
        "doc_type": "jruva_spec",
        "text": (
            "Multimodal document vision accepts image uploads (marks cards, fee receipts, identity cards) and "
            "extracts structured tabular data using easyocr and Gemini vision. OCR outputs are validated against "
            "academic schemas to identify student enrollment numbers, grades, SGPA, and semester status."
        ),
    },
    {
        "id": "doc-jruva-14",
        "doc_name": "telemetry_metrics_collector.md",
        "doc_type": "jruva_spec",
        "text": (
            "The telemetry collector records per-request metrics: request ID, timestamp, task type, selected provider, "
            "selected model, TTFT (time-to-first-token), total latency, prompt tokens, completion tokens, estimated cost, "
            "and HTTP status code. All logs are structured JSON without secrets or authorization bearer tokens."
        ),
    },
    {
        "id": "doc-jruva-15",
        "doc_name": "cloudflare_tunnel_production_ingress.md",
        "doc_type": "jruva_spec",
        "text": (
            "Cloudflare Zero Trust tunnel (`cloudflared`) connects the local Windows host to `jruva.in`. Traffic from "
            "the internet terminates at Cloudflare edge with DDoS mitigation, TLS termination, and Web Application "
            "Firewall (WAF) filtering before routing via encrypted tunnel to local port 3121 (frontend) and 8001 (backend)."
        ),
    },

    # --- JRU Academic & University Regulations (31-45) ---
    {
        "id": "doc-jru-01",
        "doc_name": "attendance_examination_eligibility_ordinance.md",
        "doc_type": "university",
        "text": (
            "Under Jharkhand Rai University Academic Ordinance 12, students must maintain a minimum attendance of 75% "
            "in aggregate across all registered theory and practical courses in a semester to be eligible for end-semester "
            "examinations. The Dean of Student Affairs may condone up to 10% on valid medical grounds upon submission "
            "of hospital admission certificates within 7 days of recovery."
        ),
    },
    {
        "id": "doc-jru-02",
        "doc_name": "cbcs_grading_and_evaluation_system.md",
        "doc_type": "university",
        "text": (
            "JRU follows the Choice Based Credit System (CBCS) 10-point scale: O (Outstanding, 10.0), A+ (Excellent, 9.0), "
            "A (Very Good, 8.0), B+ (Good, 7.0), B (Above Average, 6.0), C (Average, 5.0), P (Pass, 4.0), and F (Fail, 0.0). "
            "To qualify for graduation, an undergraduate student must achieve a cumulative CGPA of not less than 4.5."
        ),
    },
    {
        "id": "doc-jru-03",
        "doc_name": "btech_cse_fee_structure_and_eligibility.md",
        "doc_type": "university",
        "text": (
            "Eligibility for B.Tech Computer Science & Engineering requires 10+2 with Physics and Mathematics as "
            "compulsory subjects along with Chemistry/CS, securing at least 45% aggregate (40% for reserved categories). "
            "The annual tuition fee is Rs. 75,000 payable in two equal semester installments, with a one-time caution "
            "deposit of Rs. 5,000 refundable at degree completion."
        ),
    },
    {
        "id": "doc-jru-04",
        "doc_name": "btech_semester_backlog_promotion_criteria.md",
        "doc_type": "university",
        "text": (
            "To be promoted from 2nd year to 3rd year (Semester 5) in the B.Tech program, a student must have cleared "
            "at least 60% of total credits prescribed for the 1st and 2nd years combined. A maximum of 4 active backlogs "
            "is permitted at the time of Semester 5 registration."
        ),
    },
    {
        "id": "doc-jru-05",
        "doc_name": "transcript_and_degree_certificate_procedure.md",
        "doc_type": "university",
        "text": (
            "Graduates can apply for official consolidated transcripts and degree certificates through the University "
            "Examination Cell. The prescribed application form requires no-dues clearance from the library, accounts, "
            "and department. Processing fee is Rs. 1,000 per transcript set with a standard turnaround time of 15 business days."
        ),
    },
    {
        "id": "doc-jru-06",
        "doc_name": "department_mining_engineering_placements.md",
        "doc_type": "university",
        "text": (
            "The Department of Mining Engineering has dedicated MoUs with Coal India subsidiaries (CCL, BCCL) and "
            "private mining conglomerates. Key recruiting partners include Tata Steel Mining, Adani Enterprises, and "
            "JSW. Highest CTC recorded in recent placement cycles was 12 LPA with an average package of 5.2 LPA."
        ),
    },
    {
        "id": "doc-jru-07",
        "doc_name": "tuition_fee_refund_cancellation_policy.md",
        "doc_type": "university",
        "text": (
            "Tuition fee refund requests following admission cancellation adhere strictly to UGC notifications: "
            "100% refund if formally withdrawn 15 days or more before orientation (less Rs. 1,000 processing fee); "
            "90% within 15 days before orientation; 80% within 15 days after orientation; 50% between 16 and 30 days "
            "after orientation; and 0% refund after 30 days post-orientation."
        ),
    },
    {
        "id": "doc-jru-08",
        "doc_name": "merit_and_domicile_scholarships_jharkhand.md",
        "doc_type": "university",
        "text": (
            "JRU offers scholarships for domicile holders of Jharkhand state: 25% tuition fee concession for candidates "
            "with Jharkhand resident certificates securing >70% in qualifying examinations. Female students receive "
            "an additional 10% gender diversity waiver across all engineering, pharmacy, and management courses."
        ),
    },
    {
        "id": "doc-jru-09",
        "doc_name": "examination_malpractice_disciplinary_ordinance.md",
        "doc_type": "university",
        "text": (
            "Examination Malpractice Ordinance 14 penalizes unauthorized notes or mobile devices during exams. "
            "First-time possession leads to cancellation of the specific paper. Impersonation or smuggling answer "
            "booklets results in debarment from the university for two consecutive semesters."
        ),
    },
    {
        "id": "doc-jru-10",
        "doc_name": "continuous_internal_assessment_weightage.md",
        "doc_type": "university",
        "text": (
            "Course evaluation distributes 100 total marks into Continuous Internal Assessment (CIA, 40 marks) and "
            "End-Semester Examination (ESE, 60 marks). CIA comprises mid-term exam (20 marks), assignment submissions "
            "(10 marks), class presentation (5 marks), and attendance regularity (5 marks)."
        ),
    },
    {
        "id": "doc-jru-11",
        "doc_name": "hostel_regulations_and_curfew_policy.md",
        "doc_type": "university",
        "text": (
            "University residence halls mandate night curfew at 8:30 PM for female hostels and 9:30 PM for male hostels. "
            "Night-out permissions require formal written consent from registered parents or guardians via the parent portal "
            "at least 24 hours in advance. Outside visitors are restricted to the common reception foyer between 4 PM and 7 PM."
        ),
    },
    {
        "id": "doc-jru-12",
        "doc_name": "library_borrowing_rules_and_fines.md",
        "doc_type": "university",
        "text": (
            "Undergraduate students can borrow up to 3 textbooks simultaneously for a duration of 14 calendar days. "
            "Overdue books incur a fine of Rs. 5 per book per day. Reference volumes, journals, and project dissertations "
            "are strictly restricted to the reading room and cannot be checked out."
        ),
    },
    {
        "id": "doc-jru-13",
        "doc_name": "anti_ragging_committee_ugc_compliance.md",
        "doc_type": "university",
        "text": (
            "In compliance with UGC Regulations 2009, JRU enforces zero tolerance for ragging. The Anti-Ragging Committee "
            "operates a 24x7 toll-free helpline and surprise inspection squads across campus and hostels. Verified offenders "
            "face immediate suspension, cancellation of admission, and FIR lodgment with local law enforcement."
        ),
    },
    {
        "id": "doc-jru-14",
        "doc_name": "supplementary_remedial_examination_policy.md",
        "doc_type": "university",
        "text": (
            "Students obtaining an 'F' grade in an end-semester exam may register for Supplementary Examinations conducted "
            "during the summer recess. A maximum grade of 'B+' (7.0) is awarded on supplementary clearance, and examination "
            "fees are Rs. 500 per backlog subject."
        ),
    },
    {
        "id": "doc-jru-15",
        "doc_name": "innovation_incubation_cell_support.md",
        "doc_type": "university",
        "text": (
            "The JRU Innovation & Incubation Cell (IIC) provides student entrepreneurs with seed funding up to Rs. 1,00,000, "
            "dedicated co-working spaces, high-speed fiber internet, and mentorship from industry executives. Startups incubated "
            "at IIC receive legal assistance for patent filing and trademark registration."
        ),
    },

    # --- Adversarial Distractors & Lexical Look-alikes (46-50) ---
    {
        "id": "doc-dist-01",
        "doc_name": "raft_nautical_equipment_guide.md",
        "doc_type": "distractor",
        "text": (
            "Whitewater raft navigation requires specialized inflation equipment and safety gear. A standard commercial "
            "inflatable raft features multi-chamber hypalon tubes with military-grade valves. In turbulent Class IV rapids, "
            "the raft leader maintains downstream boat trajectory by shouting directional commands to rowers."
        ),
    },
    {
        "id": "doc-dist-02",
        "doc_name": "circuit_breaker_electrical_panel.md",
        "doc_type": "distractor",
        "text": (
            "Residential electrical circuit breakers protect household wiring from overload and short circuits. "
            "When current exceeding 15 or 20 Amperes passes through a bimetallic strip, thermal expansion causes "
            "the breaker to trip into the open position, breaking the physical circuit to prevent fire hazards."
        ),
    },
    {
        "id": "doc-dist-03",
        "doc_name": "mining_cryptocurrency_hardware.md",
        "doc_type": "distractor",
        "text": (
            "Cryptocurrency mining harnesses ASIC processors and GPU clusters to solve proof-of-work cryptographic puzzles. "
            "Bitcoin miners calculate SHA-256 hashes to find a nonce satisfying target difficulty. Modern mining facilities "
            "require industrial HVAC cooling and megawatts of electrical supply."
        ),
    },
    {
        "id": "doc-dist-04",
        "doc_name": "tree_biology_forest_canopy.md",
        "doc_type": "distractor",
        "text": (
            "Temperate forest trees utilize xylem and phloem vascular bundles to transport water and photosynthates. "
            "Deciduous trees shed foliage during autumn in response to reduced photoperiod and temperature drops. "
            "Root systems extend laterally to anchor the tree against gale force winds."
        ),
    },
    {
        "id": "doc-dist-05",
        "doc_name": "btech_fashion_designing_course.md",
        "doc_type": "distractor",
        "text": (
            "The Bachelor of Design in Fashion Technology emphasizes textile chemistry, garment construction, and CAD pattern "
            "making. Students create haute couture collections evaluated by fashion industry designers during annual runway shows."
        ),
    },
]

# 100 Hardened Evaluation Questions
QUESTIONS: List[Dict[str, Any]] = [
    # 1. Direct Factual Retrieval (25 items)
    {"id": "q-df-01", "category": "direct_factual", "difficulty": "easy",
     "query": "What is the randomized election timer range in Raft consensus?",
     "target_chunk_id": "doc-sys-01", "expected_evidence": "150ms and 300ms"},
    {"id": "q-df-02", "category": "direct_factual", "difficulty": "easy",
     "query": "What does efConstruction parameter control in HNSW indexing?",
     "target_chunk_id": "doc-sys-02", "expected_evidence": "dynamic candidate exploration list depth"},
    {"id": "q-df-03", "category": "direct_factual", "difficulty": "easy",
     "query": "At what capacity is an LSM MemTable flushed to disk as an SSTable?",
     "target_chunk_id": "doc-sys-03", "expected_evidence": "64MB"},
    {"id": "q-df-04", "category": "direct_factual", "difficulty": "easy",
     "query": "Which kernel routine handles page faults in Linux x86-64?",
     "target_chunk_id": "doc-sys-04", "expected_evidence": "handle_mm_fault()"},
    {"id": "q-df-05", "category": "direct_factual", "difficulty": "easy",
     "query": "What key exchange is mandated by TLS 1.3 to guarantee forward secrecy?",
     "target_chunk_id": "doc-sys-05", "expected_evidence": "Ephemeral Diffie-Hellman (ECDHE or X25519)"},
    {"id": "q-df-06", "category": "direct_factual", "difficulty": "easy",
     "query": "What state does a circuit breaker transition to after cooldown elapses?",
     "target_chunk_id": "doc-sys-06", "expected_evidence": "HALF_OPEN"},
    {"id": "q-df-07", "category": "direct_factual", "difficulty": "easy",
     "query": "Which header fields identify inserting and deleting transactions in PostgreSQL MVCC?",
     "target_chunk_id": "doc-sys-07", "expected_evidence": "xmin and xmax"},
    {"id": "q-df-08", "category": "direct_factual", "difficulty": "easy",
     "query": "What bus transaction is broadcast when writing to a Shared cache line under MESI?",
     "target_chunk_id": "doc-sys-08", "expected_evidence": "Read-with-Intent-to-Modify (RWITM)"},
    {"id": "q-df-09", "category": "direct_factual", "difficulty": "easy",
     "query": "What are the four components of the W3C traceparent header?",
     "target_chunk_id": "doc-sys-09", "expected_evidence": "version-traceid-parentid-traceflags"},
    {"id": "q-df-10", "category": "direct_factual", "difficulty": "easy",
     "query": "What compression size is achieved by Product Quantization for a 1024-d float32 vector?",
     "target_chunk_id": "doc-sys-10", "expected_evidence": "64 bytes"},
    {"id": "q-df-11", "category": "direct_factual", "difficulty": "easy",
     "query": "What internal data structure does Linux epoll use for registered file descriptors?",
     "target_chunk_id": "doc-sys-11", "expected_evidence": "red-black tree"},
    {"id": "q-df-12", "category": "direct_factual", "difficulty": "easy",
     "query": "What file extension stores the index used by readers in SQLite WAL mode?",
     "target_chunk_id": "doc-sys-12", "expected_evidence": "-shm"},
    {"id": "q-df-13", "category": "direct_factual", "difficulty": "easy",
     "query": "Where are key-payload associations stored in a B+ tree index?",
     "target_chunk_id": "doc-sys-13", "expected_evidence": "exclusively in leaf nodes"},
    {"id": "q-df-14", "category": "direct_factual", "difficulty": "easy",
     "query": "What are the two phases of classic Paxos consensus?",
     "target_chunk_id": "doc-sys-14", "expected_evidence": "Prepare/Promise and Accept/Accepted"},
    {"id": "q-df-15", "category": "direct_factual", "difficulty": "easy",
     "query": "Why are virtual nodes utilized in consistent hashing rings?",
     "target_chunk_id": "doc-sys-15", "expected_evidence": "prevent non-uniform data skew and hotspotting"},
    {"id": "q-df-16", "category": "direct_factual", "difficulty": "easy",
     "query": "What models belong to Tier A in Jruva's router specification?",
     "target_chunk_id": "doc-jruva-01", "expected_evidence": "Groq Llama-3.3-70B / Gemini-2.0-Flash"},
    {"id": "q-df-17", "category": "direct_factual", "difficulty": "easy",
     "query": "What is the typical execution latency of deterministic fastpaths in Jruva?",
     "target_chunk_id": "doc-jruva-02", "expected_evidence": "sub-10ms"},
    {"id": "q-df-18", "category": "direct_factual", "difficulty": "easy",
     "query": "What is the time-to-live (TTL) for Jruva's response cache?",
     "target_chunk_id": "doc-jruva-03", "expected_evidence": "3600 seconds"},
    {"id": "q-df-19", "category": "direct_factual", "difficulty": "easy",
     "query": "What cross-encoder model does Jruva use for FlashRank neural reranking?",
     "target_chunk_id": "doc-jruva-04", "expected_evidence": "ms-marco-TinyBERT-L-2-v2"},
    {"id": "q-df-20", "category": "direct_factual", "difficulty": "easy",
     "query": "What HTTP status code is returned when a user attempts unauthorized session access?",
     "target_chunk_id": "doc-jruva-05", "expected_evidence": "HTTP 403 Forbidden"},
    {"id": "q-df-21", "category": "direct_factual", "difficulty": "easy",
     "query": "What minimum attendance percentage is required at JRU to sit for end-semester exams?",
     "target_chunk_id": "doc-jru-01", "expected_evidence": "75%"},
    {"id": "q-df-22", "category": "direct_factual", "difficulty": "easy",
     "query": "What grade point corresponds to an 'A+' grade under JRU CBCS?",
     "target_chunk_id": "doc-jru-02", "expected_evidence": "9.0"},
    {"id": "q-df-23", "category": "direct_factual", "difficulty": "easy",
     "query": "What is the annual tuition fee for B.Tech Computer Science at JRU?",
     "target_chunk_id": "doc-jru-03", "expected_evidence": "Rs. 75,000"},
    {"id": "q-df-24", "category": "direct_factual", "difficulty": "easy",
     "query": "What is the maximum allowed backlog count for promotion to Semester 5 in B.Tech?",
     "target_chunk_id": "doc-jru-04", "expected_evidence": "maximum of 4 active backlogs"},
    {"id": "q-df-25", "category": "direct_factual", "difficulty": "easy",
     "query": "What is the fee per transcript set and standard turnaround time at JRU?",
     "target_chunk_id": "doc-jru-05", "expected_evidence": "Rs. 1,000 per transcript set with 15 business days"},

    # 2. Terminology Mismatch (15 items)
    {"id": "q-tm-01", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "How are vote collisions avoided when multiple followers wake up simultaneously in cluster voting?",
     "target_chunk_id": "doc-sys-01", "expected_evidence": "randomized election timers between 150ms and 300ms"},
    {"id": "q-tm-02", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "What parameter governs the beam search breadth during vector graph building?",
     "target_chunk_id": "doc-sys-02", "expected_evidence": "efConstruction dictates dynamic candidate exploration"},
    {"id": "q-tm-03", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "How is non-sequential disk storage avoided when writing data to persistence?",
     "target_chunk_id": "doc-sys-03", "expected_evidence": "converts random disk writes into sequential append"},
    {"id": "q-tm-04", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "What happens when an application accesses unallocated or protected memory addresses in x86?",
     "target_chunk_id": "doc-sys-04", "expected_evidence": "page fault interrupt #PF invokes handle_mm_fault"},
    {"id": "q-tm-05", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "How do cryptographic connections establish session encryption keys without waiting for server responses?",
     "target_chunk_id": "doc-sys-05", "expected_evidence": "client sends key share directly in ClientHello for 1-RTT"},
    {"id": "q-tm-06", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "How is downstream service degradation prevented from spreading upstream in distributed architectures?",
     "target_chunk_id": "doc-sys-06", "expected_evidence": "Circuit breaker trips to OPEN, failing fast"},
    {"id": "q-tm-07", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "How are obsolete record copies cleaned up from disk in relational databases?",
     "target_chunk_id": "doc-sys-07", "expected_evidence": "autovacuum daemon reclaims dead tuples into free-space maps"},
    {"id": "q-tm-08", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "How is hardware cache consistency synchronized when multiple cores modify identical memory addresses?",
     "target_chunk_id": "doc-sys-08", "expected_evidence": "MESI protocol broadcasts RWITM bus transaction"},
    {"id": "q-tm-09", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "How are user prompts categorized into specialized execution tiers in the AI engine?",
     "target_chunk_id": "doc-jruva-01", "expected_evidence": "router classifies prompts into task categories and tiers"},
    {"id": "q-tm-10", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "How are common conversational greetings processed without consuming external API credits?",
     "target_chunk_id": "doc-jruva-02", "expected_evidence": "Deterministic fastpaths bypass upstream LLM inference"},
    {"id": "q-tm-11", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "How does the platform ensure one client cannot inspect another person's uploaded knowledge vectors?",
     "target_chunk_id": "doc-jruva-06", "expected_evidence": "explicit tenant filter on owner_id keyword payload"},
    {"id": "q-tm-12", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "What security measure stops malicious users from triggering arbitrary code on the production server?",
     "target_chunk_id": "doc-jruva-07", "expected_evidence": "Arbitrary Python exec() execution is strictly disabled"},
    {"id": "q-tm-13", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "What recourse is available for students hospitalized during the academic semester regarding class attendance?",
     "target_chunk_id": "doc-jru-01", "expected_evidence": "Dean of Student Affairs may condone up to 10% on medical grounds"},
    {"id": "q-tm-14", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "What compensation is provided if a candidate decides to withdraw enrollment prior to the start of classes?",
     "target_chunk_id": "doc-jru-07", "expected_evidence": "100% refund if formally withdrawn 15 days or more before orientation"},
    {"id": "q-tm-15", "category": "terminology_mismatch", "difficulty": "medium",
     "query": "What financial grants can young founders receive to build prototypes on campus?",
     "target_chunk_id": "doc-jru-15", "expected_evidence": "seed funding up to Rs. 1,00,000 from Innovation Cell"},

    # 3. Multi-Section Retrieval (15 items)
    {"id": "q-ms-01", "category": "multi_section", "difficulty": "hard",
     "query": "Compare the consensus quorum requirements of Raft versus classic Paxos during leader proposals.",
     "target_chunk_id": "doc-sys-01", "expected_evidence": "Raft majority of cluster nodes vs Paxos majority quorum promises"},
    {"id": "q-ms-02", "category": "multi_section", "difficulty": "hard",
     "query": "Contrast how HNSW and Product Quantization handle dimensional vector scaling in large databases.",
     "target_chunk_id": "doc-sys-02", "expected_evidence": "HNSW multi-layer geometric graph vs PQ vector compression"},
    {"id": "q-ms-03", "category": "multi_section", "difficulty": "hard",
     "query": "How do LSM trees and B+ trees differ in managing disk write locality and page splits?",
     "target_chunk_id": "doc-sys-03", "expected_evidence": "LSM sequential append MemTable vs B+ tree page splits"},
    {"id": "q-ms-04", "category": "multi_section", "difficulty": "hard",
     "query": "How does SQLite WAL mode prevent reader-writer lock contention compared to PostgreSQL MVCC?",
     "target_chunk_id": "doc-sys-12", "expected_evidence": "readers do not block writers using -wal and -shm"},
    {"id": "q-ms-05", "category": "multi_section", "difficulty": "hard",
     "query": "Explain how circuit breaker state changes trigger provider failover and telemetry logging in Jruva.",
     "target_chunk_id": "doc-jruva-09", "expected_evidence": "circuit breaker trips, router switches provider, records fallback"},
    {"id": "q-ms-06", "category": "multi_section", "difficulty": "hard",
     "query": "Describe the lifecycle of a user query through fastpath, semantic cache, and hybrid RAG retrieval.",
     "target_chunk_id": "doc-jruva-04", "expected_evidence": "RRF combines dense and sparse, FlashRank reranks top candidates"},
    {"id": "q-ms-07", "category": "multi_section", "difficulty": "hard",
     "query": "How does tenant authorization enforce isolation in both session endpoints and vector searches?",
     "target_chunk_id": "doc-jruva-05", "expected_evidence": "session.user_id check and Qdrant owner_id filter"},
    {"id": "q-ms-08", "category": "multi_section", "difficulty": "hard",
     "query": "What steps are required to calculate continuous internal assessment marks and final CBCS grades at JRU?",
     "target_chunk_id": "doc-jru-10", "expected_evidence": "CIA 40 marks and ESE 60 marks mapped to 10-point scale"},
    {"id": "q-ms-09", "category": "multi_section", "difficulty": "hard",
     "query": "How do attendance condonation and semester backlog restrictions influence eligibility for graduation?",
     "target_chunk_id": "doc-jru-01", "expected_evidence": "75% attendance and credit clearance requirements"},
    {"id": "q-ms-10", "category": "multi_section", "difficulty": "hard",
     "query": "Explain the interaction between Cloudflare tunnels, TLS termination, and local FastAPI server endpoints.",
     "target_chunk_id": "doc-jruva-15", "expected_evidence": "Cloudflare edge tunnel routes encrypted traffic to local ports"},
    {"id": "q-ms-11", "category": "multi_section", "difficulty": "hard",
     "query": "How does Jruva balance token streaming efficiency, reasoning tag sanitization, and citation delivery?",
     "target_chunk_id": "doc-jruva-08", "expected_evidence": "SSE events sanitize <think> tags and emit citation payloads"},
    {"id": "q-ms-12", "category": "multi_section", "difficulty": "hard",
     "query": "What are the rules regarding campus curfew, night outs, and visitor entry in the university residences?",
     "target_chunk_id": "doc-jru-11", "expected_evidence": "curfew 8:30/9:30 PM, parent consent for night outs"},
    {"id": "q-ms-13", "category": "multi_section", "difficulty": "hard",
     "query": "Describe the end-to-end process from getting an 'F' grade to clearing the subject via supplementary exam.",
     "target_chunk_id": "doc-jru-14", "expected_evidence": "register for supplementary exam, max B+ grade, Rs. 500 fee"},
    {"id": "q-ms-14", "category": "multi_section", "difficulty": "hard",
     "query": "How do scholarships and fee concessions assist female students from Jharkhand enrolled in B.Tech?",
     "target_chunk_id": "doc-jru-08", "expected_evidence": "25% domicile concession plus 10% gender diversity waiver"},
    {"id": "q-ms-15", "category": "multi_section", "difficulty": "hard",
     "query": "Compare epoll I/O multiplexing event notifications with W3C distributed tracing context propagation.",
     "target_chunk_id": "doc-sys-11", "expected_evidence": "epoll ready list vs W3C traceparent header propagation"},

    # 4. Code & Documentation Retrieval (15 items)
    {"id": "q-cd-01", "category": "code_documentation", "difficulty": "medium",
     "query": "What is the exact constant k used for Reciprocal Rank Fusion in Jruva's hybrid RAG?",
     "target_chunk_id": "doc-jruva-04", "expected_evidence": "constant k=60"},
    {"id": "q-cd-02", "category": "code_documentation", "difficulty": "medium",
     "query": "What status string is returned when code execution is invoked in production mode?",
     "target_chunk_id": "doc-jruva-07", "expected_evidence": "status='execution_unavailable'"},
    {"id": "q-cd-03", "category": "code_documentation", "difficulty": "medium",
     "query": "What environment variable explicitly enables unsafe local Python code execution?",
     "target_chunk_id": "doc-jruva-07", "expected_evidence": "ENABLE_UNSAFE_LOCAL_CODE_EXECUTION=true"},
    {"id": "q-cd-04", "category": "code_documentation", "difficulty": "medium",
     "query": "What directory paths are used to store generated and demo image files?",
     "target_chunk_id": "doc-jruva-10", "expected_evidence": "static/generated/ and static/demo/"},
    {"id": "q-cd-05", "category": "code_documentation", "difficulty": "medium",
     "query": "What algorithm and cost factor are used for password hashing in Jruva?",
     "target_chunk_id": "doc-jruva-11", "expected_evidence": "bcrypt (cost factor 12)"},
    {"id": "q-cd-06", "category": "code_documentation", "difficulty": "medium",
     "query": "Which library powers the symbolic derivative and integral calculations in Jruva?",
     "target_chunk_id": "doc-jruva-12", "expected_evidence": "SymPy AST representations"},
    {"id": "q-cd-07", "category": "code_documentation", "difficulty": "medium",
     "query": "What optical character recognition libraries are employed for multimodal marks card parsing?",
     "target_chunk_id": "doc-jruva-13", "expected_evidence": "easyocr and Gemini vision"},
    {"id": "q-cd-08", "category": "code_documentation", "difficulty": "medium",
     "query": "What metrics are logged by the telemetry collector for each AI request?",
     "target_chunk_id": "doc-jruva-14", "expected_evidence": "request ID, provider, model, TTFT, latency, tokens, cost"},
    {"id": "q-cd-09", "category": "code_documentation", "difficulty": "medium",
     "query": "What local ports are targeted by the Cloudflare tunnel for frontend and backend traffic?",
     "target_chunk_id": "doc-jruva-15", "expected_evidence": "port 3121 (frontend) and 8001 (backend)"},
    {"id": "q-cd-10", "category": "code_documentation", "difficulty": "medium",
     "query": "What exception is raised on production startup if JWT_SECRET_KEY is not defined?",
     "target_chunk_id": "doc-jruva-11", "expected_evidence": "fatal RuntimeError"},
    {"id": "q-cd-11", "category": "code_documentation", "difficulty": "medium",
     "query": "How are unclosed reasoning blocks sanitized during token streaming?",
     "target_chunk_id": "doc-jruva-08", "expected_evidence": "Unclosed <think> tags are sanitized before emission"},
    {"id": "q-cd-12", "category": "code_documentation", "difficulty": "medium",
     "query": "What payload filter is applied to prevent cross-tenant vector retrieval in Qdrant?",
     "target_chunk_id": "doc-jruva-06", "expected_evidence": "owner_id == T OR owner_id == 'public'"},
    {"id": "q-cd-13", "category": "code_documentation", "difficulty": "medium",
     "query": "What hashing algorithm generates exact-match keys for the response cache?",
     "target_chunk_id": "doc-jruva-03", "expected_evidence": "SHA-256 normalized prompt keys"},
    {"id": "q-cd-14", "category": "code_documentation", "difficulty": "medium",
     "query": "What model embedding checkpoint is utilized in Jruva's dense retrieval stage?",
     "target_chunk_id": "doc-jruva-04", "expected_evidence": "BAAI/bge-small-en-v1.5"},
    {"id": "q-cd-15", "category": "code_documentation", "difficulty": "medium",
     "query": "What telemetry fields indicate that a deterministic shortcut was triggered?",
     "target_chunk_id": "doc-jruva-02", "expected_evidence": "fastpath=True, provider=fastpath"},

    # 5. Ambiguous Queries (10 items)
    {"id": "q-amb-01", "category": "ambiguous", "difficulty": "hard",
     "query": "What are the rules regarding timers in system state transitions?",
     "target_chunk_id": "doc-sys-01", "expected_evidence": "randomized election timers between 150ms and 300ms"},
    {"id": "q-amb-02", "category": "ambiguous", "difficulty": "hard",
     "query": "How does flushing work when memory fills up?",
     "target_chunk_id": "doc-sys-03", "expected_evidence": "MemTable reaches capacity and flushes as Level-0 SSTable"},
    {"id": "q-amb-03", "category": "ambiguous", "difficulty": "hard",
     "query": "Explain how the system handles exceptions when addressing fails.",
     "target_chunk_id": "doc-sys-04", "expected_evidence": "page faults invoke handle_mm_fault()"},
    {"id": "q-amb-04", "category": "ambiguous", "difficulty": "hard",
     "query": "What happens when things break and we need to cut off connections?",
     "target_chunk_id": "doc-sys-06", "expected_evidence": "Circuit breaker trips to OPEN"},
    {"id": "q-amb-05", "category": "ambiguous", "difficulty": "hard",
     "query": "What is the penalty for violating regulations during tests?",
     "target_chunk_id": "doc-jru-09", "expected_evidence": "cancellation of specific paper or debarment"},
    {"id": "q-amb-06", "category": "ambiguous", "difficulty": "hard",
     "query": "How much does it cost to study here per year?",
     "target_chunk_id": "doc-jru-03", "expected_evidence": "Rs. 75,000 for B.Tech CSE"},
    {"id": "q-amb-07", "category": "ambiguous", "difficulty": "hard",
     "query": "How do we route things between different tiers?",
     "target_chunk_id": "doc-jruva-01", "expected_evidence": "router classifies prompts into task categories and tiers"},
    {"id": "q-amb-08", "category": "ambiguous", "difficulty": "hard",
     "query": "Where do we put the files after creating them?",
     "target_chunk_id": "doc-jruva-10", "expected_evidence": "static/generated/ and static/demo/"},
    {"id": "q-amb-09", "category": "ambiguous", "difficulty": "hard",
     "query": "What is the policy for students who fail exams?",
     "target_chunk_id": "doc-jru-14", "expected_evidence": "supplementary examinations during summer recess"},
    {"id": "q-amb-10", "category": "ambiguous", "difficulty": "hard",
     "query": "Who helps students turn their ideas into companies?",
     "target_chunk_id": "doc-jru-15", "expected_evidence": "JRU Innovation & Incubation Cell"},

    # 6. Unanswerable Queries (No Answer Exists - Refusal Benchmark) (10 items)
    {"id": "q-unans-01", "category": "unanswerable", "difficulty": "hard",
     "query": "What is the quantum annealing coherence time for the D-Wave processor installed at JRU?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},
    {"id": "q-unans-02", "category": "unanswerable", "difficulty": "hard",
     "query": "What is the flight schedule for the university helicopter shuttle between Ranchi and Delhi?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},
    {"id": "q-unans-03", "category": "unanswerable", "difficulty": "hard",
     "query": "How many nuclear submarines are maintained by Jharkhand Rai University's naval division?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},
    {"id": "q-unans-04", "category": "unanswerable", "difficulty": "hard",
     "query": "What is the exact API endpoint for controlling humanoid robotics on Mars from Jruva?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},
    {"id": "q-unans-05", "category": "unanswerable", "difficulty": "hard",
     "query": "What is the tuition fee discount for Martian citizens applying to B.Tech Mining?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},
    {"id": "q-unans-06", "category": "unanswerable", "difficulty": "hard",
     "query": "What algorithm does Jruva use to predict tomorrow's winning lottery numbers?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},
    {"id": "q-unans-07", "category": "unanswerable", "difficulty": "hard",
     "query": "How many Michelin stars has the JRU campus dining hall received in 2026?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},
    {"id": "q-unans-08", "category": "unanswerable", "difficulty": "hard",
     "query": "What is the formula for synthesizing room-temperature superconductors in the chemistry lab?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},
    {"id": "q-unans-09", "category": "unanswerable", "difficulty": "hard",
     "query": "What is the required GPA for admission into JRU's School of Hogwarts Wizardry?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},
    {"id": "q-unans-10", "category": "unanswerable", "difficulty": "hard",
     "query": "What is the maximum warp speed permissible when parking spacecraft on the university roof?",
     "target_chunk_id": None, "expected_evidence": "NO_EVIDENCE"},

    # 7. Adversarial Distractor Queries (10 items)
    {"id": "q-adv-01", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "What directional commands does the raft leader shout to rowers in Class IV rapids?",
     "target_chunk_id": "doc-dist-01", "expected_evidence": "shouting directional commands to rowers"},
    {"id": "q-adv-02", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "What bimetallic strip amperage triggers a residential electrical circuit breaker?",
     "target_chunk_id": "doc-dist-02", "expected_evidence": "current exceeding 15 or 20 Amperes"},
    {"id": "q-adv-03", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "How do cryptocurrency mining rigs use ASIC processors to solve SHA-256 puzzles?",
     "target_chunk_id": "doc-dist-03", "expected_evidence": "ASIC processors and GPU clusters calculate SHA-256"},
    {"id": "q-adv-04", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "How do xylem and phloem vascular bundles transport nutrients in forest trees?",
     "target_chunk_id": "doc-dist-04", "expected_evidence": "xylem and phloem vascular bundles"},
    {"id": "q-adv-05", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "What runway shows evaluate haute couture collections in Fashion Designing?",
     "target_chunk_id": "doc-dist-05", "expected_evidence": "evaluated by fashion industry designers during annual runway shows"},
    {"id": "q-adv-06", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "How does Raft leader election handle hypalon tubes and military-grade valves?",
     "target_chunk_id": "doc-dist-01", "expected_evidence": "multi-chamber hypalon tubes with military-grade valves"},
    {"id": "q-adv-07", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "How does the microservices circuit breaker prevent household electrical panel fires?",
     "target_chunk_id": "doc-dist-02", "expected_evidence": "breaking the physical circuit to prevent fire hazards"},
    {"id": "q-adv-08", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "How do Mining Engineering placement students mine Bitcoin using ASIC rigs?",
     "target_chunk_id": "doc-dist-03", "expected_evidence": "ASIC processors and GPU clusters"},
    {"id": "q-adv-09", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "How does B+ tree indexing prune deciduous leaves during autumn photoperiod drops?",
     "target_chunk_id": "doc-dist-04", "expected_evidence": "Deciduous trees shed foliage during autumn"},
    {"id": "q-adv-10", "category": "adversarial_distractor", "difficulty": "hard",
     "query": "What CAD pattern making techniques are required for B.Tech Computer Science algorithms?",
     "target_chunk_id": "doc-dist-05", "expected_evidence": "textile chemistry, garment construction, and CAD pattern making"},
]
