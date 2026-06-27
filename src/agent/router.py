"""
src/agent/router.py

Classifies an incoming user query into one of five intents, so the
orchestrator knows which tool to call:

    qa                 -> a specific question to answer with retrieval+LLM
    summarize           -> "summarize what people are saying about X"
    sentiment_insight    -> questions about positive/negative/neutral breakdown
    topic_insight        -> questions about what topics/themes show up
    entity_insight       -> questions about specific named things mentioned

Two strategies are provided:

  - `classify_intent_rules`: fast, deterministic, keyword/regex based. No
    LLM call needed, so it works even with no local model running, and is
    cheap enough to run on every query.
  - `classify_intent_llm`: uses the `getIntent` DSPy signature for a more
    flexible classification when the rules don't clearly match, or when
    `use_llm=True` is passed explicitly.

`classify_intent()` is the function the orchestrator should call: it tries
the rule-based classifier first (since it's free and usually right for this
kind of query set) and only falls back to the LLM when the rules are
unsure.
"""

import re

INTENTS = ["qa", "summarize", "sentiment_insight", "topic_insight", "entity_insight"]

_SENTIMENT_STRONG = [r"\bsentiment\b", r"\bsatisfied\b", r"\bhow (do|did) (people|users|customers) feel\b"]
_SENTIMENT_WEAK = [r"\bpositive\b", r"\bnegative\b", r"\bneutral\b", r"\bhappy\b", r"\bunhappy\b", r"\bcomplain", r"\bpraise"]

_TOPIC_STRONG = [r"\btopics?\b", r"\bthemes?\b", r"\bmost (discussed|common|frequent)\b", r"\bcategor"]
_TOPIC_WEAK = [r"\btrends?\b", r"\bwhat.*(talking|talk) about\b"]

_ENTITY_STRONG = [r"\bentit", r"\bwhich (feature|product|company|person|place)s?\b"]
_ENTITY_WEAK = [r"\bnamed\b", r"\bmentioned\b.*\b(most|often)\b", r"\bwho\b", r"\bwhat (feature|product)\b"]

_SUMMARIZE_STRONG = [r"\bsummar", r"\btl;?dr\b", r"\brecap\b"]
_SUMMARIZE_WEAK = [r"\bgive me an overview\b", r"\bin short\b", r"\bwhat are people saying\b"]

_SCORED_PATTERNS = {
    "summarize": (_SUMMARIZE_STRONG, _SUMMARIZE_WEAK),
    "sentiment_insight": (_SENTIMENT_STRONG, _SENTIMENT_WEAK),
    "topic_insight": (_TOPIC_STRONG, _TOPIC_WEAK),
    "entity_insight": (_ENTITY_STRONG, _ENTITY_WEAK),
}


def _score(text, strong, weak):
    return 2 * sum(bool(re.search(p, text)) for p in strong) + sum(bool(re.search(p, text)) for p in weak)


def classify_intent_rules(query):
    """Deterministic keyword-based classification using weighted scoring:
    explicit nouns naming a category (e.g. "topics", "sentiment", "entity")
    score higher than incidental adjectives (e.g. "negative", "positive")
    that often just qualify a topic/entity query rather than ask about
    sentiment overall -- e.g. "what are the main topics in negative
    comments?" should route to topic_insight, not sentiment_insight, even
    though it contains "negative". Returns an intent string, or None if no
    category scores above zero (caller should fall back to the LLM
    classifier or default to qa in that case)."""
    text = query.lower()
    scores = {intent: _score(text, strong, weak) for intent, (strong, weak) in _SCORED_PATTERNS.items()}
    best_intent, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score > 0:
        return best_intent
    if text.strip().endswith("?") or re.match(r"^(why|how|what|when|where|is|does|did|can)\b", text.strip()):
        return "qa"
    return None


def classify_intent_llm(query):
    """LLM-based fallback classifier using the getIntent DSPy signature.
    Imported lazily so router.py has no hard dependency on dspy/generator
    unless this path is actually used."""
    from src.rag.generator import _predict_safe, gen_intent

    raw = _predict_safe(gen_intent, "intent", "qa", query=query)
    raw = str(raw).strip().lower()
    for intent in INTENTS:
        if intent in raw:
            return intent
    return "qa"


def classify_intent(query, use_llm=False):
    """Main entry point for the orchestrator. Tries the cheap rule-based
    classifier first; only calls the LLM if the rules don't match anything
    or the caller explicitly asks for the LLM path."""
    if not use_llm:
        intent = classify_intent_rules(query)
        if intent is not None:
            return intent
    try:
        return classify_intent_llm(query)
    except Exception:
        # if even the LLM fallback can't run (e.g. no local model), default
        # to qa, which is the most generically useful tool
        return "qa"
