"""Adversarial Routing Regression Dataset & Benchmark Harness.

Contains 80+ realistic prompts spanning valid introductions, near-confusables,
location/food queries, coding, web search, reasoning, and institutional questions.
Used to measure precision, false-positive fastpath triggers, and fall-through fidelity.
"""
from typing import Dict, List, Any, Optional
import sys
import os
import re

# Ensure backend path is loaded
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ADVERSARIAL_ROUTING_DATASET: List[Dict[str, Any]] = [
    # =========================================================================
    # Group 1: Valid Name Introductions (MUST trigger name fastpath or safe greeting)
    # =========================================================================
    {
        "id": "NAME-01",
        "prompt": "I am Jasir",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Jasir",
        "must_not_trigger": ["model_routing", "web_search", "coding"],
    },
    {
        "id": "NAME-02",
        "prompt": "I'm Jasir",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Jasir",
        "must_not_trigger": ["model_routing", "web_search", "coding"],
    },
    {
        "id": "NAME-03",
        "prompt": "My name is Jasir",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Jasir",
        "must_not_trigger": ["model_routing", "web_search"],
    },
    {
        "id": "NAME-04",
        "prompt": "I am John",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "John",
        "must_not_trigger": ["model_routing"],
    },
    {
        "id": "NAME-05",
        "prompt": "My name is Priya Sharma",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Priya Sharma",
        "must_not_trigger": ["model_routing"],
    },
    {
        "id": "NAME-06",
        "prompt": "Call me Alex",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Alex",
        "must_not_trigger": ["model_routing"],
    },
    {
        "id": "NAME-07",
        "prompt": "im Rahul",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Rahul",
        "must_not_trigger": ["model_routing"],
    },
    {
        "id": "NAME-08",
        "prompt": "my ame is Alan Turing",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Alan Turing",
        "must_not_trigger": ["model_routing"],
    },

    # =========================================================================
    # Group 2: NOT Name Introductions: "I am..." location / action / state
    # (MUST NOT trigger name fastpath! MUST fall through to model / search)
    # =========================================================================
    {
        "id": "NOT-NAME-01",
        "prompt": "i am at chagramkulam valayamkulam which is food spot to visit near",
        "expected_intent": "location_food_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-02",
        "prompt": "I am from Kerala",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-03",
        "prompt": "I am looking for restaurants",
        "expected_intent": "location_food_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-04",
        "prompt": "I am learning machine learning",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-05",
        "prompt": "I am working on a FastAPI project",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-06",
        "prompt": "I am getting an error in python",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-07",
        "prompt": "I am near Kuttippuram",
        "expected_intent": "location_food_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-08",
        "prompt": "I am hungry",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-09",
        "prompt": "I am trying to deploy my app",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-10",
        "prompt": "I am searching for a monitor",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-11",
        "prompt": "I am at Bangalore airport looking for a cab",
        "expected_intent": "location_food_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-12",
        "prompt": "I am feeling tired today",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-13",
        "prompt": "I am writing a blog post about artificial intelligence",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-14",
        "prompt": "I am preparing for an interview tomorrow",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-15",
        "prompt": "I am interested in quantum computing",
        "expected_intent": "reasoning",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-16",
        "prompt": "I'm travelling to Delhi next week, what are top attractions?",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-17",
        "prompt": "im building an AI agent with langchain",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-18",
        "prompt": "call me if you need more information",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-19",
        "prompt": "call me tomorrow afternoon",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-20",
        "prompt": "I am wondering how gradient descent works",
        "expected_intent": "reasoning",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },

    # =========================================================================
    # Group 3: Coding Intent & Interception Tests
    # =========================================================================
    {
        "id": "CODE-01",
        "prompt": "In a python script how do I get current date",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_date", "fastpath_name"],
    },
    {
        "id": "CODE-02",
        "prompt": "Write a SQL query to get today date and filter by status",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_date", "fastpath_name"],
    },
    {
        "id": "CODE-03",
        "prompt": "Write a Python script to sort a dictionary by key",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name", "fastpath_greeting"],
    },
    {
        "id": "CODE-04",
        "prompt": "def reverse_linked_list(head): how to implement this in C++?",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "CODE-05",
        "prompt": "Fix this bug: TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "CODE-06",
        "prompt": "How to create a Dockerfile for a FastAPI backend?",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "CODE-07",
        "prompt": "explain useEffect dependency array in React",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "CODE-08",
        "prompt": "git commit -m 'initial' gives error rejected non-fast-forward",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },

    # =========================================================================
    # Group 4: Web Search Intent vs Name / Greeting Interception
    # =========================================================================
    {
        "id": "SEARCH-01",
        "prompt": "latest news about SpaceX Starship launch today",
        "expected_intent": "web_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name", "fastpath_greeting"],
    },
    {
        "id": "SEARCH-02",
        "prompt": "current president of India",
        "expected_intent": "web_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "SEARCH-03",
        "prompt": "I am looking for recent updates on gold price in India",
        "expected_intent": "web_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "SEARCH-04",
        "prompt": "who won the cricket world cup 2024 final?",
        "expected_intent": "web_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "SEARCH-05",
        "prompt": "weather forecast for tomorrow in Mumbai",
        "expected_intent": "web_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "SEARCH-06",
        "prompt": "breaking news stock market crash today",
        "expected_intent": "web_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },

    # =========================================================================
    # Group 5: External Institutional Questions (Must NOT trigger JRU institutional fastpaths)
    # =========================================================================
    {
        "id": "INST-EXT-01",
        "prompt": "What are the fees for Stanford BTech?",
        "expected_intent": "general_qa",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_jru_fee", "fastpath_name"],
    },
    {
        "id": "INST-EXT-02",
        "prompt": "Tell me about tuition fees in Germany for BCA",
        "expected_intent": "general_qa",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_jru_fee", "fastpath_name"],
    },
    {
        "id": "INST-EXT-03",
        "prompt": "What is caution deposit in apartment lease agreement?",
        "expected_intent": "general_qa",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_jru_fee", "fastpath_name"],
    },
    {
        "id": "INST-EXT-04",
        "prompt": "What are MIT computer science course offerings?",
        "expected_intent": "general_qa",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_jru_courses", "fastpath_name"],
    },

    # =========================================================================
    # Group 6: Genuine JRU Institutional Fastpaths
    # =========================================================================
    {
        "id": "INST-JRU-01",
        "prompt": "/fee",
        "expected_intent": "institutional_jru",
        "expected_fastpath": "fastpath_jru_fee",
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "INST-JRU-02",
        "prompt": "jru btech fee structure",
        "expected_intent": "institutional_jru",
        "expected_fastpath": "fastpath_jru_fee",
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "INST-JRU-03",
        "prompt": "/courses",
        "expected_intent": "institutional_jru",
        "expected_fastpath": "fastpath_jru_courses",
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "INST-JRU-04",
        "prompt": "/attendance",
        "expected_intent": "institutional_jru",
        "expected_fastpath": "fastpath_jru_attendance",
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "INST-JRU-05",
        "prompt": "/helpline",
        "expected_intent": "institutional_jru",
        "expected_fastpath": "fastpath_jru_helpline",
        "must_not_trigger": ["fastpath_name"],
    },

    # =========================================================================
    # Group 7: Valid Pure Greetings & Small Talk Fastpaths
    # =========================================================================
    {
        "id": "GREET-01",
        "prompt": "hi",
        "expected_intent": "greeting",
        "expected_fastpath": "fastpath_greeting",
        "must_not_trigger": ["model_routing", "fastpath_name"],
    },
    {
        "id": "GREET-02",
        "prompt": "hello",
        "expected_intent": "greeting",
        "expected_fastpath": "fastpath_greeting",
        "must_not_trigger": ["model_routing", "fastpath_name"],
    },
    {
        "id": "GREET-03",
        "prompt": "good morning",
        "expected_intent": "greeting",
        "expected_fastpath": "fastpath_greeting",
        "must_not_trigger": ["model_routing", "fastpath_name"],
    },
    {
        "id": "GREET-04",
        "prompt": "who are you",
        "expected_intent": "bot_identity",
        "expected_fastpath": "fastpath_identity",
        "must_not_trigger": ["model_routing", "fastpath_name"],
    },
    {
        "id": "GREET-05",
        "prompt": "what can you do",
        "expected_intent": "bot_capabilities",
        "expected_fastpath": "fastpath_capabilities",
        "must_not_trigger": ["model_routing", "fastpath_name"],
    },
    {
        "id": "GREET-06",
        "prompt": "how are you",
        "expected_intent": "smalltalk",
        "expected_fastpath": "fastpath_smalltalk",
        "must_not_trigger": ["model_routing", "fastpath_name"],
    },
    {
        "id": "GREET-07",
        "prompt": "thank you very much",
        "expected_intent": "gratitude",
        "expected_fastpath": "fastpath_gratitude",
        "must_not_trigger": ["model_routing", "fastpath_name"],
    },
    {
        "id": "GREET-08",
        "prompt": "goodbye",
        "expected_intent": "farewell",
        "expected_fastpath": "fastpath_farewell",
        "must_not_trigger": ["model_routing", "fastpath_name"],
    },

    # =========================================================================
    # Group 8: Math & Reasoning Intent
    # =========================================================================
    {
        "id": "MATH-01",
        "prompt": "If today is Monday, what day will it be in 100 days?",
        "expected_intent": "reasoning_day_riddle",
        "expected_fastpath": "fastpath_day_riddle",
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "MATH-02",
        "prompt": "solve 2x + 5 = 15",
        "expected_intent": "reasoning",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name", "fastpath_greeting"],
    },
    {
        "id": "MATH-03",
        "prompt": "Calculate derivative of f(x) = x^3 - 4x + 1",
        "expected_intent": "reasoning",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "MATH-04",
        "prompt": "A train travels 120 km in 2 hours. What is its average speed in km/h?",
        "expected_intent": "reasoning",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "MATH-05",
        "prompt": "Prove that the sum of angles in a triangle is 180 degrees",
        "expected_intent": "reasoning",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },

    # =========================================================================
    # Group 9: Pure Date Query
    # =========================================================================
    {
        "id": "DATE-01",
        "prompt": "what is today's date",
        "expected_intent": "date_query",
        "expected_fastpath": "fastpath_date",
        "must_not_trigger": ["fastpath_name", "coding"],
    },
    {
        "id": "DATE-02",
        "prompt": "today's date",
        "expected_intent": "date_query",
        "expected_fastpath": "fastpath_date",
        "must_not_trigger": ["fastpath_name", "coding"],
    },

    # =========================================================================
    # Group 10: Location & Nearby Queries
    # =========================================================================
    {
        "id": "LOC-01",
        "prompt": "best food spots near Edappal",
        "expected_intent": "location_food_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name", "fastpath_greeting"],
    },
    {
        "id": "LOC-02",
        "prompt": "good cafes near MG Road Bangalore",
        "expected_intent": "location_food_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name", "fastpath_greeting"],
    },
    {
        "id": "LOC-03",
        "prompt": "where can I find vegetarian food near airport?",
        "expected_intent": "location_food_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "LOC-04",
        "prompt": "I am in Kochi, recommend seafood dinner places",
        "expected_intent": "location_food_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "LOC-05",
        "prompt": "I am near Central Railway Station, where is the cloak room?",
        "expected_intent": "location_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },

    # =========================================================================
    # Group 11: More "I am..." near-confusables (Verbs, States, Inquiries)
    # =========================================================================
    {
        "id": "NOT-NAME-21",
        "prompt": "I am feeling sick today",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-22",
        "prompt": "I am trying to learn C++",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-23",
        "prompt": "I am looking for hotels near Kozhikode beach",
        "expected_intent": "location_food_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-24",
        "prompt": "I am at home right now",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-25",
        "prompt": "I am an engineering student",
        "expected_intent": "academic_profile",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-26",
        "prompt": "I am in semester 4 of BCA",
        "expected_intent": "academic_profile",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-27",
        "prompt": "I am planning a trip to Manali next month",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-28",
        "prompt": "I am cooking chicken biryani for dinner",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-29",
        "prompt": "I am wondering if it will rain in Pune today",
        "expected_intent": "web_search",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-30",
        "prompt": "I am writing code in Python to scrape tables",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-31",
        "prompt": "call me later tonight",
        "expected_intent": "general_chat",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },
    {
        "id": "NOT-NAME-32",
        "prompt": "call me crazy but I think vim is better than emacs",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_name"],
    },

    # =========================================================================
    # Group 12: Additional Valid Human Introductions
    # =========================================================================
    {
        "id": "NAME-09",
        "prompt": "Hello my name is Alex",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Alex",
        "must_not_trigger": ["model_routing"],
    },
    {
        "id": "NAME-10",
        "prompt": "Hi, I am David",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "David",
        "must_not_trigger": ["model_routing"],
    },
    {
        "id": "NAME-11",
        "prompt": "I am Maya",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Maya",
        "must_not_trigger": ["model_routing"],
    },
    {
        "id": "NAME-12",
        "prompt": "My name is Sarah Connor",
        "expected_intent": "name_introduction",
        "expected_fastpath": "fastpath_name",
        "expected_name": "Sarah Connor",
        "must_not_trigger": ["model_routing"],
    },

    # =========================================================================
    # Group 13: More Coding & Institutional Interceptions
    # =========================================================================
    {
        "id": "CODE-09",
        "prompt": "How to get current date in JavaScript?",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_date", "fastpath_name"],
    },
    {
        "id": "CODE-10",
        "prompt": "SQL query to get today date and time in PostgreSQL",
        "expected_intent": "coding",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_date", "fastpath_name"],
    },
    {
        "id": "INST-EXT-05",
        "prompt": "Tell me about Harvard university tuition fee",
        "expected_intent": "general_qa",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_jru_fee", "fastpath_name"],
    },
    {
        "id": "INST-EXT-06",
        "prompt": "Oxford university fee structure for international students",
        "expected_intent": "general_qa",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_jru_fee", "fastpath_name"],
    },
    {
        "id": "INST-EXT-07",
        "prompt": "How much is security deposit for house rental in Bangalore?",
        "expected_intent": "general_qa",
        "expected_fastpath": None,
        "must_not_trigger": ["fastpath_jru_fee", "fastpath_name"],
    },
]



def classify_fastpath_type(reply: Optional[str]) -> Optional[str]:
    """Map deterministic reply text to a semantic fastpath label."""
    if not reply:
        return None
    r_lower = reply.lower()
    if "nice to meet you" in r_lower:
        return "fastpath_name"
    if "hello! how can i help you today?" in r_lower:
        return "fastpath_greeting"
    if "today's date is" in r_lower:
        return "fastpath_date"
    if "derivation" in r_lower and "mod 7" in r_lower:
        return "fastpath_day_riddle"
    if "official university fee information" in r_lower:
        return "fastpath_jru_fee"
    if "academic programs" in r_lower and "minimum eligibility" in r_lower:
        return "fastpath_jru_courses"
    if "academic attendance regulations" in r_lower:
        return "fastpath_jru_attendance"
    if "campus helplines" in r_lower:
        return "fastpath_jru_helpline"
    if "who are you" in r_lower or "i am jruva, an advanced ai assistant" in r_lower:
        return "fastpath_identity"
    if "ai assistant capabilities" in r_lower:
        return "fastpath_capabilities"
    if "doing well and ready to assist" in r_lower:
        return "fastpath_smalltalk"
    if "you're very welcome" in r_lower:
        return "fastpath_gratitude"
    if "goodbye" in r_lower:
        return "fastpath_farewell"
    return "fastpath_other"


def evaluate_routing_dataset() -> Dict[str, Any]:
    """Run all dataset cases against classify_conversational_intent and analyze failures."""
    from app.services.chat_service import classify_conversational_intent

    total = len(ADVERSARIAL_ROUTING_DATASET)
    passed = 0
    failed = 0
    failures: List[Dict[str, Any]] = []

    for item in ADVERSARIAL_ROUTING_DATASET:
        prompt = item["prompt"]
        expected_fp = item.get("expected_fastpath")
        must_not = item.get("must_not_trigger", [])

        actual_reply = classify_conversational_intent(prompt)
        actual_fp = classify_fastpath_type(actual_reply)

        # Check conditions
        is_pass = True
        reason = ""

        # 1. Check must_not_trigger
        for forbidden in must_not:
            if actual_fp == forbidden:
                is_pass = False
                reason = f"Triggered forbidden fastpath '{forbidden}' (reply: {actual_reply[:60]}...)"
                break

        # 2. Check expected_fastpath
        if is_pass and expected_fp is not None:
            if actual_fp != expected_fp:
                is_pass = False
                reason = f"Expected '{expected_fp}', got '{actual_fp}' (reply: {actual_reply[:60] if actual_reply else 'None'})"

        # 3. Check fall-through expectation (when expected_fastpath is None, actual_fp should be None)
        if is_pass and expected_fp is None and actual_fp is not None:
            is_pass = False
            reason = f"Expected fall-through (None), but triggered fastpath '{actual_fp}' (reply: {actual_reply[:60]}...)"

        if is_pass:
            passed += 1
        else:
            failed += 1
            failures.append({
                "id": item["id"],
                "prompt": prompt,
                "expected_fastpath": expected_fp,
                "actual_fastpath": actual_fp,
                "reason": reason,
            })

    accuracy = (passed / total) * 100 if total > 0 else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy_pct": round(accuracy, 2),
        "failures": failures,
    }


def test_adversarial_routing_dataset():
    """Verify that all 92 adversarial routing test cases pass without regressions."""
    results = evaluate_routing_dataset()
    assert results["failed"] == 0, f"Adversarial routing test failures: {results['failures']}"
    assert results["accuracy_pct"] == 100.0


if __name__ == "__main__":
    results = evaluate_routing_dataset()
    print("=" * 70)
    print(f"ADVERSARIAL ROUTING BENCHMARK: {results['passed']}/{results['total']} PASSED ({results['accuracy_pct']}%)")
    print(f"FAILED CASES: {results['failed']}")
    print("=" * 70)
    for f in results["failures"]:
        safe_reason = f['reason'].encode('ascii', 'replace').decode('ascii')
        print(f"[{f['id']}] Prompt: '{f['prompt']}'")
        print(f"       Expected: {f['expected_fastpath']} | Actual: {f['actual_fastpath']}")
        print(f"       Reason: {safe_reason}\n")
