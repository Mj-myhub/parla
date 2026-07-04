"""Tool 1: hybrid grammatical error detection.

Layer 1 (deterministic string rules): a/an — high precision, no model needed.
Layer 2 (spaCy morphosyntax): subject-verb agreement + number agreement, written
    against observed en_core_web_sm tags (VBP/VBZ, Number, Person, nsubj, nummod).
Layer 3 (LLM, high recall): tense/aspect, register, and the long tail — needs context,
    so it is delegated to the model rather than faked with brittle rules.

detect() reconciles the layers (dedupe by span). Precision from rules, recall from LLM.
"""
from __future__ import annotations
import re
from functools import lru_cache
from parla.graph.state import DetectedError

_TAKES_A = {
    "university", "unicorn", "unique", "union", "united", "universe", "unit",
    "uniform", "user", "useful", "usual", "european", "euro", "ukulele", "one",
    "once", "ubiquitous", "unanimous", "utopia", "eulogy", "ewe",
}
_TAKES_AN = {"hour", "honest", "honestly", "honor", "honour", "honorable", "heir", "heirloom"}
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _starts_vowel_sound(word: str) -> bool:
    w = word.lower()
    if w in _TAKES_AN:
        return True
    if w in _TAKES_A:
        return False
    return w[:1] in "aeiou"


def _check_articles(text: str) -> list[DetectedError]:
    errs: list[DetectedError] = []
    tokens = list(_WORD.finditer(text))
    for i in range(len(tokens) - 1):
        art, nxt = tokens[i].group(), tokens[i + 1].group()
        if art.lower() not in {"a", "an"} or nxt.isupper():
            continue
        vowel = _starts_vowel_sound(nxt)
        if art.lower() == "a" and vowel:
            errs.append(DetectedError(span=f"{art} {nxt}", error_type="R:DET", suggestion=f"an {nxt}"))
        elif art.lower() == "an" and not vowel:
            errs.append(DetectedError(span=f"{art} {nxt}", error_type="R:DET", suggestion=f"a {nxt}"))
    return errs


_IRREGULAR_3SG = {"have": "has", "be": "is", "do": "does", "go": "goes"}


def third_person(lemma: str) -> str:
    lemma = lemma.lower()
    if lemma in _IRREGULAR_3SG:
        return _IRREGULAR_3SG[lemma]
    if lemma.endswith(("s", "x", "z", "ch", "sh", "o")):
        return lemma + "es"
    if len(lemma) > 1 and lemma.endswith("y") and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ies"
    return lemma + "s"


def pluralize(noun: str) -> str:
    n = noun.lower()
    if n.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if len(n) > 1 and n.endswith("y") and n[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


_QUANTIFIERS = {"many", "several", "few", "both"}


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def _subject_class(subj) -> str:
    morph = str(subj.morph)
    if subj.pos_ in {"NOUN", "PROPN"}:
        return "nonsg" if "Number=Plur" in morph else "3sg"
    person = next((p for p in ("1", "2", "3") if f"Person={p}" in morph), None)
    if "Number=Plur" in morph or person in {"1", "2"}:
        return "nonsg"
    if person == "3" and "Number=Sing" in morph:
        return "3sg"
    return "unknown"


def _spacy_checks(text: str) -> list[DetectedError]:
    try:
        doc = _nlp()(text)
    except OSError:
        return []
    errs: list[DetectedError] = []
    for tok in doc:
        if tok.tag_ in {"VBP", "VBZ"}:
            subj = next((c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
            if subj is not None:
                cls = _subject_class(subj)
                if cls == "3sg" and tok.tag_ == "VBP":
                    errs.append(DetectedError(span=f"{subj.text} {tok.text}", error_type="R:VERB:SVA",
                                              suggestion=f"{subj.text} {third_person(tok.lemma_)}"))
                elif cls == "nonsg" and tok.tag_ == "VBZ":
                    errs.append(DetectedError(span=f"{subj.text} {tok.text}", error_type="R:VERB:SVA",
                                              suggestion=f"{subj.text} {tok.lemma_}"))
        if tok.tag_ == "NN":
            for child in tok.children:
                q = child.lemma_.lower()
                if (child.dep_ == "nummod" and q not in {"one", "1", "a", "an"}) or q in _QUANTIFIERS:
                    errs.append(DetectedError(span=f"{child.text} {tok.text}", error_type="R:NOUN:NUM",
                                              suggestion=f"{child.text} {pluralize(tok.text)}"))
                    break
    return errs


def _llm_pass(text: str) -> list[DetectedError]:
    return []


def parse_llm_errors(payload: list[dict]) -> list[DetectedError]:
    out: list[DetectedError] = []
    for item in payload:
        span = str(item.get("span", "")).strip()
        if span:
            out.append(DetectedError(span=span,
                                     error_type=str(item.get("error_type", "R:OTHER")).strip() or "R:OTHER",
                                     suggestion=str(item.get("suggestion", "")).strip()))
    return out


def _dedupe(errors: list[DetectedError]) -> list[DetectedError]:
    seen: dict[str, DetectedError] = {}
    for e in errors:
        seen.setdefault(e.span.lower(), e)
    return list(seen.values())


def detect(text: str, use_spacy: bool = True, use_llm: bool = False) -> list[DetectedError]:
    errors = _check_articles(text)
    if use_spacy:
        errors += _spacy_checks(text)
    if use_llm:
        errors += _llm_pass(text)
    return _dedupe(errors)
