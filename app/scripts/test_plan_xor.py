from pydantic import ValidationError

from app.agents.nodes.planner import parse_planner_decision, planner_node
from app.agents.state import PlanState, initial_agent_state

# XOR: both filled must fail (DaemonSet scenario)
try:
    PlanState(
        intent="conversational",
        conversational_reply="You're welcome!",
        rewritten_query="what is a kubernetes daemonset",
    )
    print("FAIL: both allowed")
except ValidationError:
    print("block_both OK")

try:
    PlanState(intent="technical")
    print("FAIL: neither allowed")
except ValidationError:
    print("block_neither OK")

a = PlanState(intent="conversational", conversational_reply="Hi!")
b = PlanState(intent="technical", rewritten_query="daemonset explained")
print("ok", a.intent, b.intent)

try:
    parse_planner_decision(
        {
            "intent": "conversational",
            "conversational_reply": "You're welcome!",
            "rewritten_query": "what is a kubernetes daemonset",
        }
    )
    print("FAIL: decision allowed mixed")
except ValidationError:
    print("block_mixed_decision OK")

out = planner_node(initial_agent_state("thanks, and also what's a DaemonSet?"))
print("mixed_stub", out["intent"], out["status"], (out.get("current_query") or "")[:50])

out2 = planner_node(initial_agent_state("thanks"))
print("pure_thanks", out2["intent"], out2["status"])
