"""
src/agent/orchestrator.py

The single entry point Member 4's Streamlit dashboard (and anyone else)
should call: takes a raw user query, routes it to the right intent, runs
the matching tool, and returns one consistent response shape with
citations back to the source comment_id(s) it used.

This is the "query -> tool -> response with citations" piece described in
the Member 3 task list.
"""

from src.agent.router import classify_intent
from src.agent.tools import TOOL_REGISTRY
from src.rag.generator import DEFAULT_DATA_PATH


class Orchestrator:
    def __init__(self, path=DEFAULT_DATA_PATH, use_llm_router=False):
        self.path = path
        self.use_llm_router = use_llm_router

    def handle(self, query, **tool_kwargs):
        import inspect
        intent = classify_intent(query, use_llm=self.use_llm_router)
        tool   = TOOL_REGISTRY[intent]

        # Strip kwargs the tool doesn't accept to prevent TypeError
        sig        = inspect.signature(tool)
        accepted   = set(sig.parameters)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_kw:
            tool_kwargs = {k: v for k, v in tool_kwargs.items() if k in accepted}

        try:
            result = tool(query, path=self.path, **tool_kwargs)
        except Exception as exc:
            result = {
                "intent":    intent,
                "answer":    f"⚠ Tool error ({intent}): {exc}",
                "citations": [],
                "raw":       {"error": str(exc)},
            }

        return {
            "query":     query,
            "intent":    result["intent"],
            "answer":    result["answer"],
            "citations": sorted(set(result.get("citations", []))),
            "raw":       result.get("raw", {}),
        }

    def format_response(self, response):
        cite_str = ", ".join(f"#{c}" for c in response["citations"]) or "none"
        return (
            f"Intent: {response['intent']}\n"
            f"Answer: {response['answer']}\n"
            f"Sources: {cite_str}"
        )


# module-level convenience function so callers don't need to instantiate
# the class for simple one-off queries
_default_orchestrator = None


def handle_query(query, path=DEFAULT_DATA_PATH, use_llm_router=False, **tool_kwargs):
    global _default_orchestrator
    if _default_orchestrator is None or _default_orchestrator.path != path:
        _default_orchestrator = Orchestrator(path=path, use_llm_router=use_llm_router)
    return _default_orchestrator.handle(query, **tool_kwargs)


if __name__ == "__main__":
    # one demo query per intent, same spirit as 7_Generator.py's
    # "set a question, run the pipeline, print the result" demo
    demo_queries = [
        "What do viewers say about Ryan's storytelling style?",
        "Summarize audience opinion on Ryan Trahan's recent videos.",
        "What is the overall sentiment in the comments?",
        "What are the main topics in negative comments?",
        "Which people and brands are most mentioned by viewers?",
    ]
    orch = Orchestrator()
    for q in demo_queries:
        response = orch.handle(q)
        print(orch.format_response(response))
        print("-" * 60)
