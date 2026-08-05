"""Smoke-test Neon session store + multi-turn agent memory."""

from app.config import get_settings
from app.db.session_store import create_session, load_messages
from app.agents.graph import invoke_agent

# Reload settings so DATABASE_URL is picked up after config edits.
get_settings.cache_clear()
from app.config import settings

print("has_database", settings.has_database)

sid = create_session()
print("session", sid)

r1 = invoke_agent(
    "How do Kubernetes Jobs process work from a Redis queue?",
    session_id=sid,
)
print("T1", r1["intent"], "msgs", len(r1.get("messages") or []))
print("A1", (r1.get("final_answer") or "")[:160].replace("\n", " "))

r2 = invoke_agent("how do I monitor it?", session_id=sid)
print("T2", r2["intent"], "query", r2.get("current_query"))
print("A2", (r2.get("final_answer") or "")[:160].replace("\n", " "))
print("db_messages", len(load_messages(sid)))
for m in load_messages(sid)[-4:]:
    print(f"  {m['role']}: {m['content'][:80].replace(chr(10), ' ')}")
