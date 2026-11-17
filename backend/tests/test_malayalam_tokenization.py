"""Malayalam text must stay whole when Jruva splits text into words.

Python's ``\\w`` does not match combining marks such as the Malayalam vowel sign "ി"
or the virama "്", so a plain ``\\w`` or ASCII regex cut "കേരളത്തിന്റെ" into
pieces (or dropped it completely). Keyword search (BM25) and answer grounding then
saw no Malayalam words at all.
"""
from app.services.chat_service import has_semantic_grounding
from app.services.rag.document_rag import NON_WORD_PATTERN, BM25Index, tokenize

KERALA_CAPITAL = "കേരളത്തിന്റെ തലസ്ഥാനം തിരുവനന്തപുരം ആണ്."
ML_DOCS = [
    "കൊച്ചി ഒരു തുറമുഖ നഗരമാണ്. കപ്പലുകൾ ഇവിടെ വരുന്നു.",
    KERALA_CAPITAL + " നിയമസഭ അവിടെയാണ്.",
    "മൂന്നാർ തേയിലത്തോട്ടങ്ങൾക്ക് പ്രസിദ്ധമാണ്.",
]


def test_tokenize_keeps_malayalam_words_whole():
    tokens = tokenize(KERALA_CAPITAL)

    assert "കേരളത്തിന്റെ" in tokens
    assert "തലസ്ഥാനം" in tokens
    assert "തിരുവനന്തപുരം" in tokens


def test_tokenize_keeps_chillu_written_with_zero_width_joiner():
    word = "അവന്‍"  # older chillu spelling: na + virama + ZWJ

    assert tokenize(f"{word} വന്നു") == [word, "വന്നു"]


def test_tokenize_english_behaviour_is_unchanged():
    text = "The B.Tech fee-structure, e.g. 3.14 and X-ray... is in hostel_rules."

    assert tokenize(text) == ["b.tech", "fee-structure", "e.g", "3.14", "x-ray", "hostel_rules"]


def test_bm25_ranks_the_matching_malayalam_document_first():
    index = BM25Index()
    index.fit([tokenize(doc) for doc in ML_DOCS])

    scores = index.score(tokenize("കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?"))

    assert scores[1] > 0
    assert scores[1] == max(scores)


def test_non_word_pattern_does_not_split_malayalam_words():
    cleaned = NON_WORD_PATTERN.sub(" ", "കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?").split()

    assert cleaned == ["കേരളത്തിന്റെ", "തലസ്ഥാനം", "ഏതാണ്"]


def test_semantic_grounding_matches_malayalam_query_to_malayalam_passage():
    assert has_semantic_grounding("തിരുവനന്തപുരം എവിടെയാണ്?", KERALA_CAPITAL)
    assert not has_semantic_grounding("മൂന്നാർ എവിടെയാണ്?", KERALA_CAPITAL)
