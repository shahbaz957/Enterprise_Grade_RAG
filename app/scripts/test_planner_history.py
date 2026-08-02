"""Smoke-test history-aware planner (DaemonSet → how do I monitor it?)."""

from langchain_core.messages import AIMessage, HumanMessage

from app.agents import initial_agent_state, planner_node

state = initial_agent_state("how do I monitor it?")
state["messages"] = [
    HumanMessage(content="What is a DaemonSet?"),
    AIMessage(
        content=(
            "A DaemonSet ensures a copy of a Pod runs on all (or selected) "
            "nodes in a Kubernetes cluster — often used for log collectors "
            "and node monitors."
        )
    ),
]

out = planner_node(state)
print("intent:", out.get("intent"))
print("status:", out.get("status"))
print("rewritten_query:", out.get("current_query") or (out.get("plan") and out["plan"].rewritten_query))
print("final_answer:", (out.get("final_answer") or "")[:80])
print("error:", out.get("error"))
