"""End-to-end agent smoke: session memory via thread_id."""

from app.agents.graph import invoke_agent

THREAD = "demo-session-daemonset"

r1 = invoke_agent(
    "How do Kubernetes Jobs process work from a Redis queue?",
    thread_id=THREAD,
)
print("T1", r1.get("intent"), r1.get("status"), "docs", len(r1.get("documents") or []))
print("A1", (r1.get("final_answer") or "")[:220].replace("\n", " "))
print("thread", r1.get("thread_id"))

r2 = invoke_agent("how do I monitor it?", thread_id=THREAD)
print("T2", r2.get("intent"), r2.get("status"), "docs", len(r2.get("documents") or []))
print("A2", (r2.get("final_answer") or "")[:220].replace("\n", " "))
print("query2", r2.get("current_query"))
