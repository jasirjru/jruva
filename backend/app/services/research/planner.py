"""Bounded Research Planner: Decomposes complex queries into focused sub-queries."""

import re
from typing import List


class ResearchPlanner:
    """Generates a strictly bounded set of 2-3 search queries for deep research."""

    MAX_SUBQUERIES = 3

    @classmethod
    def plan_subqueries(cls, topic: str) -> List[str]:
        clean_topic = topic.strip()
        # Clean question punctuation
        clean_topic = re.sub(r"[?!.]+$", "", clean_topic)

        # 1. Base query
        queries = [clean_topic]

        # 2. Check for comparison ("vs", "compare", "difference between")
        compare_match = re.search(r"(\b[\w\s]+?)\s+(?:vs\.?|versus|compare|and)\s+([\w\s]+)", clean_topic, re.IGNORECASE)
        if compare_match and len(compare_match.groups()) == 2:
            left = compare_match.group(1).strip()
            right = compare_match.group(2).strip()
            if left and right:
                queries.append(f"{left} features architecture")
                queries.append(f"{right} features architecture")

        # 3. Check for technical topic
        elif any(w in clean_topic.lower() for w in ["how to", "architecture", "explained", "best practices", "performance"]):
            queries.append(f"{clean_topic} overview documentation")
            queries.append(f"{clean_topic} benchmarks analysis")

        # 4. Default expansion
        else:
            queries.append(f"{clean_topic} recent developments")

        # Enforce strict bound
        return queries[:cls.MAX_SUBQUERIES]
