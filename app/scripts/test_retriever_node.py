"""Smoke-test retriever node: skip conversational; search+rerank technical."""

from app.agents import initial_agent_state, retriever_node
from app.agents.state import PlanState

# 1) Skip non-technical
conv = initial_agent_state("hi")
conv["intent"] = "conversational"
conv["plan"] = PlanState(
    intent="conversational",
    conversational_reply="Hello!",
)
conv["final_answer"] = "Hello!"
conv["status"] = "done"
out_skip = retriever_node(conv)
print("skip_docs", len(out_skip.get("documents") or []), "keys", sorted(out_skip.keys()))

# 2) Technical retrieval
tech = initial_agent_state("How do Kubernetes Jobs process a work queue?")
tech["intent"] = "technical"
tech["plan"] = PlanState(
    intent="technical",
    rewritten_query="Kubernetes Job Redis work queue parallel processing",
)
tech["status"] = "retrieving"
out = retriever_node(tech)
docs = out.get("documents") or []
print("status", out.get("status"), "error", out.get("error"))
print("docs", len(docs))
for d in docs[:5]:
    name = d.filename or d.source
    preview = (d.text or "")[:90].replace("\n", " ").encode("ascii", "replace").decode()
    print(f"  #{d.rank} score={d.score} file={name}")
    print(f"     {preview}")
