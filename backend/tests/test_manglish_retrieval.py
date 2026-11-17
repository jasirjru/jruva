"""Manglish (Malayalam typed in Latin letters) must find Malayalam-script passages.

Users often type "keralathinte thalasthanam ethanu" instead of
"കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്". Keyword search compared Latin query words with
Malayalam-script document words, found no shared word, and ranked passages by
chance (3/15 top-1 in the 10 Nov audit). The fix romanizes Malayalam document words,
compares them with the query words by sound, and adds the closest Malayalam words to
the query. English and Malayalam queries must rank exactly as before.

The documents and queries below are the synthetic set from the 10 Nov audit probe
(scratch/audit-2026-11-10/compare_attachment_retrieval.py).
"""
import base64

import pytest
from app.schemas.chat import AttachmentItem
from app.services.rag import document_rag
from app.services.rag.document_rag import DocumentRAGService
from app.services.rag.manglish import (
    expand_manglish_query,
    phonetic_key,
    romanize_malayalam,
)

DOCS = [
    ("a.txt", "മൺപാത്ര പരിശീലനം ബുധനാഴ്ചയാണ്. കളിമൺ ശിൽപശാലയുടെ ഫീസ് 240 രൂപയാണ്."),
    ("b.txt", "താരനിരീക്ഷണം ശനിയാഴ്ച നടക്കും. ദൂരദർശിനി ഉപയോഗിക്കാൻ അനുമതി ആവശ്യമാണ്."),
    ("c.txt", "മഴവെള്ള സംഭരണിയുടെ ശേഷി 850 ലിറ്ററാണ്. കുടിവെള്ള പരിശോധന തിങ്കളാഴ്ചയാണ്."),
    ("d.txt", "Ceramic kiln maintenance uses a copper latch. The equipment tag is KESTREL-14."),
    ("e.txt", "The seed archive borrows a silver trolley. Its return deadline is Tuesday afternoon."),
    ("f.txt", "കേരളത്തിന്റെ തലസ്ഥാനം തിരുവനന്തപുരം ആണ്. നിയമസഭ അവിടെയാണ്."),
]

MANGLISH_CASES = [
    ("manpaathra parisheelanam eppozhaanu?", "a.txt"),
    ("dooradarshini upayogikkan enthu venam?", "b.txt"),
    ("mazhavella sambharaniyude sheshi ethra?", "c.txt"),
    ("keralathinte thalasthanam ethanu", "f.txt"),
    ("kalimann shilpashalayude fees ethra?", "a.txt"),
    ("kudivella parishodhana eppol?", "c.txt"),
]

UNCHANGED_CASES = [
    ("കളിമൺ ശിൽപശാലയുടെ ഫീസ് എത്രയാണ്?", "a.txt"),
    ("മൺപാത്ര പരിശീലനം ഏതു ദിവസമാണ്?", "a.txt"),
    ("താരനിരീക്ഷണം എപ്പോഴാണ്?", "b.txt"),
    ("ദൂരദർശിനി ഉപയോഗിക്കാൻ എന്താണ് ആവശ്യം?", "b.txt"),
    ("മഴവെള്ള സംഭരണിയുടെ ശേഷി എത്ര?", "c.txt"),
    ("കുടിവെള്ള പരിശോധന എപ്പോഴാണ്?", "c.txt"),
    ("What is the ceramic kiln's equipment tag?", "d.txt"),
    ("Which latch does ceramic kiln maintenance use?", "d.txt"),
    ("When is the seed archive's silver trolley due back?", "e.txt"),
    ("What does the seed archive borrow?", "e.txt"),
]


@pytest.fixture(autouse=True)
def _low_memory_retrieval(monkeypatch):
    """Use the production (512 MiB) path: BM25 + n-gram fusion, no dense model."""
    from app.services.rag import qdrant_vector_store

    monkeypatch.setattr(
        qdrant_vector_store, "is_embedding_enabled", lambda: (False, "disabled_for_test")
    )


def _attachments(docs):
    return [
        AttachmentItem(
            name=name,
            mime_type="text/plain",
            data=base64.b64encode(text.encode()).decode(),
        )
        for name, text in docs
    ]


def _first_doc_in_every_rotation(query):
    service = DocumentRAGService()
    attachments = _attachments(DOCS)
    firsts = []
    for shift in range(len(attachments)):
        ordered = attachments[shift:] + attachments[:shift]
        chunks, _ = service.retrieve(ordered, query, top_k=4)
        firsts.append(chunks[0]["doc_name"] if chunks else None)
    return firsts


def test_romanize_follows_common_manglish_spelling():
    assert romanize_malayalam("തിരുവനന്തപുരം") == "thiruvananthapuram"
    assert romanize_malayalam("കേരളത്തിന്റെ") == "keralaththinte"
    assert romanize_malayalam("ഏതാണ്") == "ethaanu"
    assert romanize_malayalam("മൺപാത്ര") == "manpaathra"
    assert romanize_malayalam("അവന്‍") == "avan"


def test_phonetic_key_matches_manglish_and_romanized_spellings():
    pairs = [
        ("keralathinte", "കേരളത്തിന്റെ"),
        ("thalasthanam", "തലസ്ഥാനം"),
        ("mazhavella", "മഴവെള്ള"),
        ("parisheelanam", "പരിശീലനം"),
        ("upayogikkan", "ഉപയോഗിക്കാൻ"),
    ]
    for manglish, malayalam in pairs:
        assert phonetic_key(manglish) == phonetic_key(romanize_malayalam(malayalam))


@pytest.mark.parametrize("query,expected", MANGLISH_CASES)
def test_manglish_query_ranks_matching_malayalam_passage_first(query, expected):
    assert _first_doc_in_every_rotation(query) == [expected] * len(DOCS)


@pytest.mark.parametrize("query,expected", UNCHANGED_CASES)
def test_english_and_malayalam_queries_still_rank_first(query, expected):
    assert _first_doc_in_every_rotation(query) == [expected] * len(DOCS)


def test_expansion_adds_matching_malayalam_words_and_keeps_the_query():
    vocabulary = {"കേരളത്തിന്റെ", "തലസ്ഥാനം", "തിരുവനന്തപുരം", "നിയമസഭ"}

    expanded = expand_manglish_query("keralathinte thalasthanam ethanu", vocabulary)

    assert expanded.startswith("keralathinte thalasthanam ethanu")
    assert "കേരളത്തിന്റെ" in expanded
    assert "തലസ്ഥാനം" in expanded
    assert "നിയമസഭ" not in expanded


def test_expansion_skips_malayalam_queries_and_latin_only_documents():
    malayalam_query = "കേരളത്തിന്റെ തലസ്ഥാനം"
    assert expand_manglish_query(malayalam_query, {"കേരളത്തിന്റെ"}) == malayalam_query
    assert expand_manglish_query("ceramic kiln", {"ceramic", "kiln"}) == "ceramic kiln"
    assert expand_manglish_query("ceramic kiln", set()) == "ceramic kiln"


def test_retrieve_does_not_expand_english_queries_over_english_documents(monkeypatch):
    calls = []
    real = document_rag.expand_manglish_query

    def spy(query, vocabulary):
        calls.append(query)
        return real(query, vocabulary)

    monkeypatch.setattr(document_rag, "expand_manglish_query", spy)
    service = DocumentRAGService()
    service.retrieve(_attachments(DOCS[3:5]), "What does the seed archive borrow?")

    assert calls == []
