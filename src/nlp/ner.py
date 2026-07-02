"""
src/nlp/ner.py
--------------
Named Entity Recognition (NER) using spaCy.

Adds to each comment row:
  - entities      : JSON list of {text, label} dicts
  - entity_count  : integer count of extracted entities

Compatible with downstream RAG (entity_insight intent) and
Dashboard (entity frequency charts).

Author : Member 2 – NLP Enrichment Layer
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
import spacy
from spacy.language import Language

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s – %(message)s"))
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# spaCy entity labels to keep (tune as needed for YouTube comment context)
# ---------------------------------------------------------------------------
KEEP_LABELS: set[str] = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT",
    "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY",
    "NORP",   # nationalities, religious/political groups
    "FAC",    # buildings, airports, etc.
}


# ---------------------------------------------------------------------------
# Model loading (singleton pattern — load once per process)
# ---------------------------------------------------------------------------
_NLP_MODEL: Language | None = None
_LOADED_MODEL_NAME: str = ""


def load_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """
    Load (or return cached) spaCy model.

    Parameters
    ----------
    model_name : str
        Any spaCy model identifier.  Defaults to the small English model.
        For higher accuracy use ``en_core_web_trf`` (transformer-based).

    Returns
    -------
    spacy.language.Language
    """
    global _NLP_MODEL, _LOADED_MODEL_NAME
    if _NLP_MODEL is None or _LOADED_MODEL_NAME != model_name:
        logger.info("Loading spaCy model: %s", model_name)
        try:
            _NLP_MODEL = spacy.load(model_name)
            _LOADED_MODEL_NAME = model_name
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                f"Run:  python -m spacy download {model_name}"
            ) from exc
    return _NLP_MODEL


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def _extract_entities_from_doc(doc: Any) -> list[dict[str, str]]:
    """
    Return a deduplicated list of entity dicts from a spaCy Doc.

    Each dict has keys ``text`` and ``label``.
    Only entities whose label is in ``KEEP_LABELS`` are retained.
    """
    seen: set[tuple[str, str]] = set()
    entities: list[dict[str, str]] = []
    for ent in doc.ents:
        if ent.label_ not in KEEP_LABELS:
            continue
        key = (ent.text.strip(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        entities.append({"text": ent.text.strip(), "label": ent.label_})
    return entities


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str, nlp: Language) -> list[dict[str, str]]:
    """
    Run NER on a single comment string.

    Parameters
    ----------
    text : str
        Raw or cleaned comment text.
    nlp : Language
        Loaded spaCy pipeline.

    Returns
    -------
    list[dict]
        E.g. ``[{"text": "Apple", "label": "ORG"}, ...]``
    """
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        doc = nlp(text[:1_000_000])   # spaCy max-length guard
        return _extract_entities_from_doc(doc)
    except Exception as exc:           # noqa: BLE001
        logger.warning("NER failed for text snippet '%s…': %s", text[:60], exc)
        return []


def run_ner_on_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    model_name: str = "en_core_web_sm",
    batch_size: int = 64,
    n_process: int = 1,
) -> pd.DataFrame:
    """
    Vectorised NER over a DataFrame using spaCy's ``pipe`` for efficiency.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``text_col``.
    text_col : str
        Column to run NER on.
    model_name : str
        spaCy model to load.
    batch_size : int
        Texts per batch sent to spaCy (tune for memory/speed).
    n_process : int
        Parallel workers.  ``-1`` uses all CPU cores.
        NOTE: multiprocessing requires a forking-safe environment.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with two new columns appended:
        ``entities`` (JSON string) and ``entity_count`` (int).
    """
    nlp = load_spacy_model(model_name)

    texts: list[str] = (
        df[text_col].fillna("").astype(str).tolist()
    )

    logger.info(
        "Running NER on %d comments (batch_size=%d, n_process=%d) …",
        len(texts), batch_size, n_process,
    )

    all_entities: list[list[dict[str, str]]] = []
    try:
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            all_entities.append(_extract_entities_from_doc(doc))
    except Exception as exc:           # noqa: BLE001
        logger.error("Batch NER pipeline failed: %s — falling back to row-by-row.", exc)
        all_entities = [extract_entities(t, nlp) for t in texts]

    df = df.copy()
    df["entities"] = [json.dumps(e, ensure_ascii=False) for e in all_entities]
    df["entity_count"] = [len(e) for e in all_entities]

    logger.info("NER complete.  Mean entities/comment: %.2f", df["entity_count"].mean())
    return df


# ---------------------------------------------------------------------------
# Convenience helpers for downstream consumers (Member 3 / Member 4)
# ---------------------------------------------------------------------------

def parse_entities(entities_json: str) -> list[dict[str, str]]:
    """
    Deserialise the ``entities`` column value back to a Python list.

    Usage (in RAG / Dashboard):
    >>> ents = parse_entities(row["entities"])
    >>> [e["text"] for e in ents if e["label"] == "ORG"]
    """
    try:
        return json.loads(entities_json) if entities_json else []
    except json.JSONDecodeError:
        return []


def get_entities_by_label(
    entities_json: str,
    label: str,
) -> list[str]:
    """Return entity text values for a specific NER label."""
    return [
        e["text"]
        for e in parse_entities(entities_json)
        if e.get("label") == label
    ]
