"""Cross-script retrieval for Manglish: Malayalam typed in Latin letters.

Users often type "keralathinte thalasthanam ethanu" for "കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്".
Keyword search cannot match a Latin word with a Malayalam-script word, so:

1. ``romanize_malayalam`` writes each Malayalam document word in Latin letters, close to
   common Manglish spelling (ISO 15919-like, with "th" for ത and a final "u" for a
   word-final virama).
2. ``phonetic_key`` reduces both spellings to a rough sound key (drops aspiration and
   voicing, vowel length, doubled letters), because Manglish spelling is not fixed.
3. ``expand_manglish_query`` appends the Malayalam document words whose key is close
   to a query word's key. The original query is kept, so English words still match.

Pure Python, no model, no network; memory is a small LRU cache of word keys.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

_MALAYALAM = re.compile(r"[\u0D00-\u0D7F]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_LATIN_WORD = re.compile(r"[a-z]+")

_VIRAMA = "\u0D4D"
_ZWJ = "\u200D"
_ZWNJ = "\u200C"
# Private-use stand-ins for clusters Manglish writes differently from their parts.
_NTA = "\uE000"  # ന്റ -> "nt" (as in "ente")
_TTA = "\uE001"  # റ്റ -> "tt"
_CLUSTERS = (("ന്റ", _NTA), ("റ്റ", _TTA))

_CONSONANTS = {
    "ക": "k", "ഖ": "kh", "ഗ": "g", "ഘ": "gh", "ങ": "ng",
    "ച": "ch", "ഛ": "chh", "ജ": "j", "ഝ": "jh", "ഞ": "nj",
    "ട": "t", "ഠ": "th", "ഡ": "d", "ഢ": "dh", "ണ": "n",
    "ത": "th", "ഥ": "th", "ദ": "d", "ധ": "dh", "ന": "n",
    "പ": "p", "ഫ": "ph", "ബ": "b", "ഭ": "bh", "മ": "m",
    "യ": "y", "ര": "r", "ല": "l", "വ": "v", "ശ": "sh",
    "ഷ": "sh", "സ": "s", "ഹ": "h", "ള": "l", "ഴ": "zh",
    "റ": "r", _NTA: "nt", _TTA: "tt",
}
_VOWELS = {
    "അ": "a", "ആ": "aa", "ഇ": "i", "ഈ": "ee", "ഉ": "u", "ഊ": "oo",
    "ഋ": "ru", "എ": "e", "ഏ": "e", "ഐ": "ai", "ഒ": "o", "ഓ": "o", "ഔ": "au",
}
_VOWEL_SIGNS = {
    "ാ": "aa", "ി": "i", "ീ": "ee", "ു": "u", "ൂ": "oo", "ൃ": "ru",
    "െ": "e", "േ": "e", "ൈ": "ai", "ൊ": "o", "ോ": "o", "ൌ": "au", "ൗ": "au",
}
_OTHER = {
    "ൺ": "n", "ൻ": "n", "ർ": "r", "ൽ": "l", "ൾ": "l", "ൿ": "k",
    "ം": "m", "ഃ": "h", _ZWJ: "", _ZWNJ: "",
}

# Spelling variants that Manglish writers use interchangeably, applied in order.
_KEY_REPLACEMENTS = (
    ("zh", "l"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "t"), ("kh", "k"),
    ("gh", "k"), ("ph", "p"), ("bh", "p"), ("jh", "j"), ("nj", "n"), ("ng", "n"),
    ("ee", "i"), ("oo", "u"), ("w", "v"), ("f", "p"), ("z", "s"), ("q", "k"),
    ("x", "ks"), ("d", "t"), ("g", "k"), ("b", "p"),
)
_REPEATS = re.compile(r"(.)\1+")

# Question and filler words that would otherwise pull in unrelated Malayalam words.
_MANGLISH_STOPWORDS = (
    "aanu", "anu", "ano", "aano", "alle", "undo", "und", "illa", "ethu", "ethanu",
    "enthu", "entha", "enthanu", "ethra", "ethrayanu", "eppol", "eppozhanu",
    "eppozhaanu", "evide", "evideyanu", "aaru", "aaranu", "engane", "enganeyanu",
    "venam", "ennu", "ethokke", "oru", "ee", "aa",
)

MIN_QUERY_WORD_LEN = 3
MATCH_THRESHOLD = 0.6
PREFIX_MATCH_SCORE = 0.75
MIN_PREFIX_LEN = 5
MATCHES_PER_WORD = 2


def contains_malayalam(text: str) -> bool:
    return bool(_MALAYALAM.search(text))


def is_latin_query(text: str) -> bool:
    """True when the query has Latin letters and no Malayalam script."""
    return bool(_LATIN_LETTER.search(text)) and not contains_malayalam(text)


def romanize_malayalam(word: str) -> str:
    """Write a Malayalam word in Latin letters, close to common Manglish spelling."""
    for cluster, stand_in in _CLUSTERS:
        word = word.replace(cluster, stand_in)
    out: list[str] = []
    i = 0
    while i < len(word):
        char = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if char not in _CONSONANTS:
            out.append(_VOWELS.get(char) or _OTHER.get(char, char))
            i += 1
            continue
        base = _CONSONANTS[char]
        if nxt in _VOWEL_SIGNS:
            out.append(base + _VOWEL_SIGNS[nxt])
            i += 2
        elif nxt == _VIRAMA:
            after = word[i + 2] if i + 2 < len(word) else ""
            if after == _ZWJ:  # old chillu spelling: consonant + virama + ZWJ
                out.append(base)
                i += 3
            else:
                # A word-final virama is spoken (and written in Manglish) as a short "u".
                out.append(base + ("u" if not after else ""))
                i += 2
        else:
            out.append(base + "a")
            i += 1
    return "".join(out)


def phonetic_key(latin_word: str) -> str:
    """Reduce a Latin spelling to a rough sound key shared by Manglish variants."""
    key = "".join(_LATIN_WORD.findall(latin_word.lower()))
    for old, new in _KEY_REPLACEMENTS:
        key = key.replace(old, new)
    return _REPEATS.sub(r"\1", key)


_STOP_KEYS = frozenset(phonetic_key(w) for w in _MANGLISH_STOPWORDS)


def _bigrams(key: str) -> frozenset[str]:
    padded = f"^{key}$"
    return frozenset(padded[i:i + 2] for i in range(len(padded) - 1))


@lru_cache(maxsize=32768)  # short strings only: a few MiB when full
def _malayalam_word_key(word: str) -> str:
    return phonetic_key(romanize_malayalam(word))


def _similarity(query_key: str, query_grams: frozenset[str], doc_key: str) -> float:
    doc_grams = _bigrams(doc_key)
    dice = 2 * len(query_grams & doc_grams) / (len(query_grams) + len(doc_grams))
    shorter, longer = sorted((query_key, doc_key), key=len)
    if len(shorter) >= MIN_PREFIX_LEN and longer.startswith(shorter):
        return max(dice, PREFIX_MATCH_SCORE)
    return dice


def _keys_by_first_letter(vocabulary: Iterable[str]) -> dict[str, list[tuple[str, str]]]:
    buckets: dict[str, list[tuple[str, str]]] = {}
    for word in vocabulary:
        key = _malayalam_word_key(word)
        if key:
            buckets.setdefault(key[0], []).append((key, word))
    return buckets


def _closest_words(query_key: str, buckets: dict[str, list[tuple[str, str]]]) -> list[str]:
    query_grams = _bigrams(query_key)
    max_length_gap = max(3, len(query_key) // 2)
    scored: list[tuple[float, str]] = []
    for doc_key, word in buckets.get(query_key[0], ()):
        if abs(len(doc_key) - len(query_key)) > max_length_gap:
            continue
        score = _similarity(query_key, query_grams, doc_key)
        if score >= MATCH_THRESHOLD:
            scored.append((score, word))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [word for _, word in scored[:MATCHES_PER_WORD]]


def malayalam_vocabulary(tokenized_docs: Iterable[Iterable[str]]) -> set[str]:
    """Distinct Malayalam-script tokens across the tokenized documents."""
    return {tok for doc in tokenized_docs for tok in doc if contains_malayalam(tok)}


def expand_manglish_query(query: str, vocabulary: set[str]) -> str:
    """Append Malayalam document words that sound like the query's Latin words.

    Returns the query unchanged when it contains Malayalam script, has no Latin
    letters, or the documents have no Malayalam words.
    """
    if not is_latin_query(query):
        return query
    buckets = _keys_by_first_letter(w for w in vocabulary if contains_malayalam(w))
    if not buckets:
        return query
    additions: list[str] = []
    seen: set[str] = set()
    for word in _LATIN_WORD.findall(query.lower()):
        key = phonetic_key(word)
        if len(key) < MIN_QUERY_WORD_LEN or key in _STOP_KEYS or key in seen:
            continue
        seen.add(key)
        additions.extend(w for w in _closest_words(key, buckets) if w not in additions)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"
