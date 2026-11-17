"""Held-Out Benchmark Dataset for RAG Generation-Quality Evaluation.

Maintains strict separation from training/fine-tuning and retrieval benchmarks:
1. 5 Reference Documents across academic, curriculum, and distributed systems domains
2. 16 Evaluation Cases partitioned into 4 distinct slices:
   - simple_factual (4 queries)
   - ambiguous (4 queries)
   - multi_part (4 queries)
   - noisy_typo (4 queries)
3. For every evaluation case:
   - User Query and Slice Category
   - Target Reference Document ID & Ground-Truth Reference Facts
   - Expected Entities and Key Phrases
   - Pre-computed Grounded Answer (Faithful to context)
   - Synthetic Hallucinated Answer (Contains unsupported claims / altered numbers)
   - Irrelevant Answer (Off-topic or non-responsive)
"""
from typing import Any, Dict, List

# ==============================================================================
# 1. HELD-OUT REFERENCE DOCUMENTS
# ==============================================================================
HELD_OUT_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "doc_01",
        "doc_name": "university_fee_structure.txt",
        "text": (
            "JHARKHAND RAI UNIVERSITY ANNUAL FEE STRUCTURE\n"
            "Academic Year 2025-2026 Regulations.\n"
            "1. B.Tech Computer Science and Engineering: Annual tuition fee is INR 85,000 per academic year.\n"
            "2. Admission Registration Fee: A one-time non-refundable fee of INR 5,000 payable at enrollment.\n"
            "3. Examination Fee: INR 2,500 per semester payable before end-semester examinations.\n"
            "4. Hostel Accommodation and Mess Charges: INR 60,000 annually for double-occupancy rooms.\n"
            "5. Caution Deposit: Refundable security deposit of INR 5,000 upon course completion."
        ),
    },
    {
        "id": "doc_02",
        "doc_name": "cbcs_examination_ordinance.txt",
        "text": (
            "CHOICE BASED CREDIT SYSTEM (CBCS) EXAMINATION ORDINANCE\n"
            "1. Grading Scale: 10-point scale where 'O' represents Outstanding (Grade Point 10),\n"
            "   'A+' represents Excellent (Grade Point 9), and 'P' represents Pass (Grade Point 4).\n"
            "2. Passing Standard: A student must secure a minimum of 40% marks in Continuous Internal Assessment (CIA)\n"
            "   and 40% marks in End Semester Examination (ESE) independently to clear each course.\n"
            "3. Attendance Requirement: Mandatory minimum of 75% attendance in lectures and practicals.\n"
            "   Dean may condone up to 10% on genuine medical grounds upon submission of certified documents."
        ),
    },
    {
        "id": "doc_03",
        "doc_name": "btech_cse_syllabus_regulations.txt",
        "text": (
            "B.TECH COMPUTER SCIENCE & ENGINEERING CURRICULUM REGULATIONS\n"
            "1. Degree Requirements: Total graduation requirement is 160 credits across 8 semesters.\n"
            "2. Core Course Load: Semester 3 covers Data Structures (4 credits) and Digital Logic (3 credits).\n"
            "3. Semester 5 covers Operating Systems (4 credits) and Database Management Systems (4 credits).\n"
            "4. Final Year Project: Capstone Project Part I (2 credits in Sem 7) and Part II (6 credits in Sem 8).\n"
            "5. Industrial Internship: Mandatory 6-week summer internship between Sem 6 and Sem 7."
        ),
    },
    {
        "id": "doc_04",
        "doc_name": "distributed_consensus_raft_spec.txt",
        "text": (
            "RAFT DISTRIBUTED CONSENSUS PROTOCOL SPECIFICATION\n"
            "1. Leader Election: Follower election timeout is randomized between 150ms and 300ms.\n"
            "2. Transition: If no heartbeat is received before timeout, follower increments term counter,\n"
            "   transitions to Candidate state, votes for itself, and broadcasts RequestVote RPCs.\n"
            "3. Quorum: A candidate must receive votes from a strict majority (N/2 + 1) of cluster nodes to win.\n"
            "4. Heartbeats: The elected leader periodically sends empty AppendEntries RPCs every 50ms."
        ),
    },
    {
        "id": "doc_05",
        "doc_name": "fastapi_async_backend_architecture.txt",
        "text": (
            "CHATBOT ASYNC BACKEND ARCHITECTURE\n"
            "1. Server Framework: Built with FastAPI running on Uvicorn with uvloop event loop.\n"
            "2. Database Engine: SQLite configured with WAL (Write-Ahead Logging) journal mode and 15-second busy timeout.\n"
            "3. Circuit Breaker: Stateful circuit breakers per provider with 3-failure threshold and 15-second cooldown.\n"
            "4. Rate Limiting: SlowAPI sliding-window rate limiting configured per client device identifier.\n"
            "5. Vector Retrieval: Hybrid in-memory BM25 keyword matching combined with dense vector indexing."
        ),
    },
]

# ==============================================================================
# 2. HELD-OUT EVALUATION CASES ACROSS 4 SLICES
# ==============================================================================
HELD_OUT_EVAL_CASES: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------------
    # SLICE 1: SIMPLE FACTUAL QUERIES (4)
    # --------------------------------------------------------------------------
    {
        "id": "gen-01",
        "slice": "simple_factual",
        "query": "What is the annual tuition fee for B.Tech Computer Science and Engineering?",
        "target_doc_id": "doc_01",
        "ground_truth_facts": ["85,000", "per academic year", "annual tuition fee"],
        "expected_entities": ["B.Tech", "85,000", "INR 85,000"],
        "grounded_answer": "The annual tuition fee for B.Tech Computer Science and Engineering is INR 85,000 per academic year.",
        "hallucinated_answer": "The annual tuition fee for B.Tech CSE is INR 1,50,000 per semester with a 20% early bird discount.",
        "irrelevant_answer": "Raft leader election uses randomized timers between 150ms and 300ms to avoid split votes.",
    },
    {
        "id": "gen-02",
        "slice": "simple_factual",
        "query": "What is the minimum attendance percentage required under the CBCS examination ordinance?",
        "target_doc_id": "doc_02",
        "ground_truth_facts": ["75%", "attendance", "minimum"],
        "expected_entities": ["75%", "CBCS"],
        "grounded_answer": "Under the CBCS examination ordinance, students must maintain a mandatory minimum of 75% attendance in lectures and practicals.",
        "hallucinated_answer": "The minimum attendance required is 60%, and students can self-certify their absence without medical documentation.",
        "irrelevant_answer": "SQLite uses Write-Ahead Logging mode with a 15-second busy timeout.",
    },
    {
        "id": "gen-03",
        "slice": "simple_factual",
        "query": "How many total credits are required for graduation in B.Tech CSE?",
        "target_doc_id": "doc_03",
        "ground_truth_facts": ["160 credits", "8 semesters", "graduation requirement"],
        "expected_entities": ["160", "credits"],
        "grounded_answer": "The total graduation requirement for B.Tech Computer Science & Engineering is 160 credits across 8 semesters.",
        "hallucinated_answer": "Students must complete 200 credits including 40 mandatory non-academic physical education credits.",
        "irrelevant_answer": "Hostel accommodation and mess charges are INR 60,000 annually for double-occupancy rooms.",
    },
    {
        "id": "gen-04",
        "slice": "simple_factual",
        "query": "What is the election timer range for followers in Raft distributed consensus?",
        "target_doc_id": "doc_04",
        "ground_truth_facts": ["150ms", "300ms", "randomized", "election timer"],
        "expected_entities": ["150ms", "300ms", "Raft"],
        "grounded_answer": "In Raft distributed consensus, follower election timeouts are randomized between 150ms and 300ms.",
        "hallucinated_answer": "Follower election timeouts in Raft are strictly fixed at 5000 milliseconds for all cluster peers.",
        "irrelevant_answer": "The one-time admission registration fee is INR 5,000 payable at enrollment.",
    },

    # --------------------------------------------------------------------------
    # SLICE 2: AMBIGUOUS QUERIES (4)
    # --------------------------------------------------------------------------
    {
        "id": "gen-05",
        "slice": "ambiguous",
        "query": "What are the passing rules?",
        "target_doc_id": "doc_02",
        "ground_truth_facts": ["40%", "Continuous Internal Assessment", "End Semester Examination", "independently"],
        "expected_entities": ["40%", "CIA", "ESE"],
        "grounded_answer": "According to the CBCS ordinance, to pass each course a student must secure a minimum of 40% in Continuous Internal Assessment (CIA) and 40% in End Semester Examination (ESE) independently.",
        "hallucinated_answer": "Passing requires an aggregate 50% across all subjects combined, with automatic moderation for failing grades.",
        "irrelevant_answer": "The elected leader sends heartbeat AppendEntries RPCs every 50ms.",
    },
    {
        "id": "gen-06",
        "slice": "ambiguous",
        "query": "Tell me about the project work.",
        "target_doc_id": "doc_03",
        "ground_truth_facts": ["Capstone Project", "Part I", "2 credits", "Part II", "6 credits"],
        "expected_entities": ["Capstone Project", "Sem 7", "Sem 8"],
        "grounded_answer": "The curriculum specifies Capstone Project Part I in Semester 7 (2 credits) and Capstone Project Part II in Semester 8 (6 credits).",
        "hallucinated_answer": "Project work is an optional 1-credit assignment completed individually in Semester 1.",
        "irrelevant_answer": "A refundable caution deposit of INR 5,000 is returned upon course completion.",
    },
    {
        "id": "gen-07",
        "slice": "ambiguous",
        "query": "How does the circuit breaker work?",
        "target_doc_id": "doc_05",
        "ground_truth_facts": ["circuit breaker", "3-failure threshold", "15-second cooldown"],
        "expected_entities": ["3-failure threshold", "15-second", "circuit breaker"],
        "grounded_answer": "The backend implements stateful circuit breakers per provider with a 3-failure threshold and a 15-second cooldown period.",
        "hallucinated_answer": "Circuit breakers trip after 50 consecutive timeouts and require manual administrator CLI reset.",
        "irrelevant_answer": "Grade 'O' represents Outstanding with Grade Point 10 on the 10-point scale.",
    },
    {
        "id": "gen-08",
        "slice": "ambiguous",
        "query": "What happens when a node times out?",
        "target_doc_id": "doc_04",
        "ground_truth_facts": ["increments term", "Candidate state", "votes for itself", "RequestVote RPCs"],
        "expected_entities": ["Candidate", "term", "RequestVote"],
        "grounded_answer": "When an election timer elapses without heartbeats, the follower increments its term counter, transitions to Candidate state, votes for itself, and broadcasts RequestVote RPCs.",
        "hallucinated_answer": "When a node times out, it permanently shuts down and removes itself from the cluster configuration file.",
        "irrelevant_answer": "A mandatory 6-week summer internship is conducted between Semester 6 and Semester 7.",
    },

    # --------------------------------------------------------------------------
    # SLICE 3: MULTI-PART QUERIES (4)
    # --------------------------------------------------------------------------
    {
        "id": "gen-09",
        "slice": "multi_part",
        "query": "What is the B.Tech tuition fee AND what is the admission registration fee?",
        "target_doc_id": "doc_01",
        "ground_truth_facts": ["85,000", "5,000", "tuition fee", "admission registration"],
        "expected_entities": ["85,000", "5,000", "tuition", "registration"],
        "grounded_answer": "The annual tuition fee for B.Tech is INR 85,000 per academic year, and the one-time admission registration fee is INR 5,000 payable at enrollment.",
        "hallucinated_answer": "The tuition fee is INR 85,000 per year, but the admission registration fee is INR 25,000 with an additional processing charge of INR 10,000.",
        "irrelevant_answer": "In Raft, a candidate must receive votes from a strict majority (N/2 + 1) of cluster peers.",
    },
    {
        "id": "gen-10",
        "slice": "multi_part",
        "query": "What are the passing marks for CIA and ESE, and what attendance is mandatory?",
        "target_doc_id": "doc_02",
        "ground_truth_facts": ["40%", "CIA", "ESE", "75%", "attendance"],
        "expected_entities": ["40%", "75%", "CIA", "ESE"],
        "grounded_answer": "Students must secure at least 40% in CIA and 40% in ESE independently to pass, alongside maintaining a mandatory minimum of 75% attendance.",
        "hallucinated_answer": "Passing marks are 50% in CIA and 30% in ESE, and attendance is completely optional if marks exceed 80%.",
        "irrelevant_answer": "Semester 3 covers Data Structures (4 credits) and Digital Logic (3 credits).",
    },
    {
        "id": "gen-11",
        "slice": "multi_part",
        "query": "How many credits are Data Structures and Digital Logic in Semester 3?",
        "target_doc_id": "doc_03",
        "ground_truth_facts": ["Data Structures", "4 credits", "Digital Logic", "3 credits"],
        "expected_entities": ["Data Structures", "Digital Logic", "4 credits", "3 credits"],
        "grounded_answer": "In Semester 3, Data Structures is 4 credits and Digital Logic is 3 credits.",
        "hallucinated_answer": "Data Structures is 6 credits with mandatory laboratory thesis, and Digital Logic is 1 credit.",
        "irrelevant_answer": "Examination fee is INR 2,500 per semester payable before end-semester examinations.",
    },
    {
        "id": "gen-12",
        "slice": "multi_part",
        "query": "What quorum is required to win election in Raft, and how often are heartbeats sent?",
        "target_doc_id": "doc_04",
        "ground_truth_facts": ["strict majority", "N/2 + 1", "50ms", "heartbeat", "AppendEntries"],
        "expected_entities": ["majority", "N/2 + 1", "50ms"],
        "grounded_answer": "A candidate must receive votes from a strict majority (N/2 + 1) of cluster nodes to win, and the elected leader sends heartbeat AppendEntries RPCs every 50ms.",
        "hallucinated_answer": "Election requires unanimous 100% agreement, and heartbeats are sent every 10 minutes.",
        "irrelevant_answer": "The backend server uses FastAPI running on Uvicorn with the uvloop event loop.",
    },

    # --------------------------------------------------------------------------
    # SLICE 4: NOISY / TYPO QUERIES (4)
    # --------------------------------------------------------------------------
    {
        "id": "gen-13",
        "slice": "noisy_typo",
        "query": "could u tell me b.tech annual tuition fees pls?",
        "target_doc_id": "doc_01",
        "ground_truth_facts": ["85,000", "B.Tech", "tuition fee"],
        "expected_entities": ["85,000", "B.Tech"],
        "grounded_answer": "The annual tuition fee for B.Tech Computer Science and Engineering is INR 85,000 per academic year.",
        "hallucinated_answer": "B.Tech fees are INR 2,00,000 annually with an extra laboratory surcharge.",
        "irrelevant_answer": "Dean may condone up to 10% attendance on genuine medical grounds.",
    },
    {
        "id": "gen-14",
        "slice": "noisy_typo",
        "query": "plz explain cbcs grading scale for O and A+ grades??",
        "target_doc_id": "doc_02",
        "ground_truth_facts": ["10-point scale", "'O'", "Outstanding", "Grade Point 10", "'A+'", "Grade Point 9"],
        "expected_entities": ["10-point", "O", "A+", "10", "9"],
        "grounded_answer": "Under the CBCS 10-point grading scale, 'O' represents Outstanding with Grade Point 10, and 'A+' represents Excellent with Grade Point 9.",
        "hallucinated_answer": "CBCS uses a 4-point American GPA scale where 'O' is 4.0 and 'A+' is 3.7.",
        "irrelevant_answer": "Semester 5 covers Operating Systems (4 credits) and Database Management Systems (4 credits).",
    },
    {
        "id": "gen-15",
        "slice": "noisy_typo",
        "query": "WHAT IS RAFT ELECTION TIMER RANGE???",
        "target_doc_id": "doc_04",
        "ground_truth_facts": ["150ms", "300ms", "election timer", "randomized"],
        "expected_entities": ["150ms", "300ms"],
        "grounded_answer": "In Raft consensus, follower election timeouts are randomized between 150ms and 300ms.",
        "hallucinated_answer": "Raft election timer is fixed between 10 seconds and 30 seconds for all servers.",
        "irrelevant_answer": "Admission registration fee is INR 5,000 one-time non-refundable.",
    },
    {
        "id": "gen-16",
        "slice": "noisy_typo",
        "query": "plz tell database journal mode & busy timeout in backend",
        "target_doc_id": "doc_05",
        "ground_truth_facts": ["WAL", "Write-Ahead Logging", "15-second", "busy timeout"],
        "expected_entities": ["WAL", "15-second", "SQLite"],
        "grounded_answer": "The SQLite database is configured with WAL (Write-Ahead Logging) journal mode and a 15-second busy timeout.",
        "hallucinated_answer": "The database uses MEMORY journal mode with zero timeout and no locking safeguards.",
        "irrelevant_answer": "Mandatory 6-week summer internship between Sem 6 and Sem 7.",
    },
]
