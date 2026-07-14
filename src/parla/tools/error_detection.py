"""Tool 1: hybrid grammatical error detection.

Layer 1 (deterministic string rules): a/an — high precision, no model needed.
Layer 2 (spaCy morphosyntax): subject-verb agreement + number agreement.
Layer 3 (LLM, high recall): tense/aspect, prepositions, context-dependent article errors,
    word form — the long tail rules can't safely catch. Enabled with use_llm=True.

detect() reconciles the layers (dedupe by span). Precision from rules, recall from LLM.
"""
from __future__ import annotations
import json
import re
from functools import lru_cache
from parla.graph.state import DetectedError

# ---- Layer 1: a/an ---------------------------------------------------------
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


# ---- morphology helpers (pure, unit-tested) --------------------------------
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


# ---- Layer 1b: spellcheck (deterministic; NON-WORD errors only) ------------
# Catches dictionary misses like "aventages"/"cointend" with near-perfect precision.
# Real-word misuse ("whit", "flash") needs context and is left to the LLM layer.
@lru_cache(maxsize=1)
def _spell():
    try:
        from spellchecker import SpellChecker
        return SpellChecker()
    except Exception:
        return None


def _check_spelling(text: str) -> list[DetectedError]:
    sp = _spell()
    if sp is None:
        return []
    errs: list[DetectedError] = []
    for m in _WORD.finditer(text):
        w = m.group()
        if len(w) < 4 or not w.islower() or "'" in w or "-" in w:
            continue                              # skip short words, proper nouns, contractions
        if w in sp.unknown([w]):
            corr = sp.correction(w)
            if corr and corr != w:
                errs.append(DetectedError(span=w, error_type="R:SPELL", suggestion=corr))
    return errs


# ---- Layer 2: spaCy agreement checks ---------------------------------------
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


# ---- Layer 3: LLM recall pass ----------------------------------------------
_ALLOWED_TAGS = (
    "M:DET, R:DET, U:DET, R:VERB:SVA, R:VERB:TENSE, R:VERB:FORM, M:PREP, R:PREP, U:PREP, "
    "R:NOUN:NUM, R:PRON, R:WO, R:SPELL, M:PUNCT, R:PUNCT, R:WORD"
)
_LLM_SYSTEM = (
    "You are an expert English grammar checker for language learners. Identify only GENUINE "
    "issues in the learner's text, and separate two kinds:\n"
    "1. GRAMMAR errors that are objectively wrong (agreement, tense, articles, prepositions, "
    "noun number, verb form, word order, spelling, punctuation, pronouns).\n"
    "2. STYLE / word-choice suggestions that are not wrong but could read better.\n\n"
    "For EACH item output an object with keys: 'span' (the exact incorrect substring, copied "
    "verbatim from the text), 'suggestion' (the corrected text for that span only), and "
    "'error_type'. For a GRAMMAR error, 'error_type' MUST be exactly one tag from this list: "
    f"{_ALLOWED_TAGS}. For a STYLE / word-choice suggestion, set 'error_type' to exactly "
    "'STYLE'. Pick the single most accurate tag. Do NOT flag correct text, and never label a "
    "style preference with a grammar tag. "
    "Respond with ONLY a JSON array of such objects and nothing else. If there are none, "
    "respond with []."
)


def _extract_json_array(raw: str) -> list:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*", "", s).rstrip("`").strip()
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j == -1 or j < i:
        return []
    try:
        data = json.loads(s[i:j + 1])
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def parse_llm_errors(payload: list[dict]) -> list[DetectedError]:
    out: list[DetectedError] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        span = str(item.get("span", "")).strip()
        if span:
            out.append(DetectedError(span=span,
                                     error_type=str(item.get("error_type", "R:OTHER")).strip() or "R:OTHER",
                                     suggestion=str(item.get("suggestion", "")).strip()))
    return out


@lru_cache(maxsize=1)
def _default_llm():
    from langchain.chat_models import init_chat_model
    from parla.config import settings
    return init_chat_model(settings.synth_model, model_provider=settings.llm_provider, temperature=0)


def _llm_pass(text: str, model=None) -> list[DetectedError]:
    """Layer 3, sentence-chunked: per-sentence LLM calls yield minimal, well-tagged
    edits (vs. clause-length rewrites on long essays), which also ground far better."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return _llm_pass_single(text, model=model)
    errs: list[DetectedError] = []
    for s in sentences:
        errs.extend(_llm_pass_single(s, model=model))
    return errs


def _llm_pass_single(text: str, model=None) -> list[DetectedError]:
    try:
        llm = model or _default_llm()
        resp = llm.invoke([("system", _LLM_SYSTEM), ("human", text)])
        content = resp.content if hasattr(resp, "content") else str(resp)
        return parse_llm_errors(_extract_json_array(content))
    except Exception:
        return []
def _dedupe(errors: list[DetectedError]) -> list[DetectedError]:
    seen: dict[str, DetectedError] = {}
    for e in errors:
        seen.setdefault(e.span.lower(), e)
    return list(seen.values())


def detect(text: str, use_spacy: bool = True, use_llm: bool = False, model=None) -> list[DetectedError]:
    errors = _check_articles(text)
    errors += _check_spelling(text)
    if use_spacy:
        errors += _spacy_checks(text)
    if use_llm:
        errors += _llm_pass(text, model=model)
    return _dedupe(errors)
