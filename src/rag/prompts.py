"""
DSPy signatures used by the RAG layer.

`getAnswer` below is copied as-is from the professor's 7_Generator.py so the
generation behavior matches what was demonstrated in class. The other
signatures extend that same pattern to cover summarization (Task 9 option A:
"ask an LLM to summarize the text then use it as context"), HyDE-style
retrieval (the `retrievers.HyDE(...)` call referenced in 7_Generator.py), and
the natural-language phrasing needed by the agent's non-generative tools
(sentiment / topic / entity insight).

Every signature here is intentionally short and single-purpose, the same way
getAnswer is, since DSPy signatures are meant to describe one IO contract
each rather than a whole pipeline.
"""

import dspy


class getAnswer(dspy.Signature):
    """You are a YouTube audience analyst. Answer in 2-3 sentences using ONLY
    information from the provided comments. Reuse names and key phrases from
    the comments (e.g. "funny", "beautiful", "so good"). Describe the overall
    sentiment and themes viewers express. Do not invent jobs, skills, traits,
    or facts that are not stated or clearly implied in the comments."""

    question = dspy.InputField(desc="the user's question about viewer opinions")
    context = dspy.InputField(desc="YouTube comments from real viewers")
    answer = dspy.OutputField(desc="2-3 sentence answer grounded in comment wording and themes")


class getSummary(dspy.Signature):
    """Summarize the provided text faithfully. Only include information that
    appears in the text. Do not add opinions, interpretations, or facts that
    are not present in the source material."""

    text = dspy.InputField(desc="retrieved comments to summarize")
    summary = dspy.OutputField(desc="faithful summary that only uses information from the provided text")


class getHyDE(dspy.Signature):
    """write a short hypothetical answer to the question, to be used purely
    for embedding/retrieval purposes (Hypothetical Document Embeddings).
    The hypothetical answer does not need to be factually correct, it only
    needs to read like a real answer so its embedding lands near real
    relevant documents in vector space."""

    question = dspy.InputField(desc="question or a concept")
    hypothetical_answer = dspy.OutputField(
        desc="a plausible-sounding short answer to the question, used only for retrieval"
    )


class getIntent(dspy.Signature):
    """classify a user query into exactly one of five intents: qa,
    summarize, sentiment_insight, topic_insight, entity_insight"""

    query = dspy.InputField(desc="the user's natural language query")
    intent = dspy.OutputField(
        desc="one of: qa, summarize, sentiment_insight, topic_insight, entity_insight"
    )


class getInsight(dspy.Signature):
    """turn structured aggregate statistics (about sentiment, topics, or
    entities, pulled from the enriched comments dataset) into a short,
    natural-language insight that directly answers the user's question"""

    question = dspy.InputField(desc="the user's natural language query")
    stats = dspy.InputField(desc="structured aggregate statistics relevant to the query")
    insight = dspy.OutputField(desc="short natural-language answer grounded only in the given stats")
