# router/Router.py

from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph, START, END
from llm.openRouter import call_llm
from agent_prompt.RouterQuery import ROUTER_PROMPT


# ----------------------------
# Graph State
# ----------------------------
class RouterState(TypedDict):
    input: str
    decision: Literal["langchain", "langgraph", "both"]


# ----------------------------
# Router node
# ----------------------------
def router_node(state: RouterState):
    decision = call_llm(
        prompt=state["input"],
        system_prompt=ROUTER_PROMPT,
    )
    print(f"Router decision: {decision.strip().lower()}")
    return {"decision": decision.strip().lower()}


# ----------------------------
# Build graph
# ----------------------------
builder = StateGraph(RouterState)
builder.add_node("router", router_node)
builder.add_edge(START, "router")
builder.add_edge("router", END)

router_graph = builder.compile()


# ----------------------------
# Public API
# ----------------------------
def route_query(question: str) -> str:
    result = router_graph.invoke({"input": question})
    return result["decision"]
