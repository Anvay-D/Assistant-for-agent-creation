# -------------------------------------------------
# 1️⃣  Imports – only the symbols that appear in the docs
# -------------------------------------------------
from langchain.chat_models import init_chat_model
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from llm.openRouter import call_llm


# -------------------------------------------------
# 2️⃣  Define the graph state (the object that flows between nodes)
# -------------------------------------------------
class GraphState(TypedDict):
    input: str
    decision: Literal["langchain", "langgraph", "both"]
    result: str = ""

    # any other fields you want to carry forward can be added here

# -------------------------------------------------
# 3️⃣  Build the routing function (the “router”)
# -------------------------------------------------
def llm_call_router(state: GraphState) -> dict:
    decision = call_llm(
        prompt=state.input,
        system_prompt=ROUTER_PROMPT
            )
    decision = decision.strip().lower()

    if decision not in {"langchain", "langgraph", "both"}:
        decision = "langchain"  # safe default

    return {"decision": decision}



# -------------------------------------------------
# 4️⃣  Define the downstream nodes (example placeholders)
# -------------------------------------------------
def langchain_node(state: GraphState) -> dict:
    """
    Query LangChain vector DB
    """
    # pseudo-code (replace with your retriever)
    answer = f"[LangChain DB] Answer for: {state.input}"

    return {"result": answer}



def langgraph_node(state: GraphState) -> dict:
    """
    Query LangGraph vector DB
    """
    # pseudo-code (replace with your retriever)
    answer = f"[LangGraph DB] Answer for: {state.input}"

    return {"result": answer}

def merge_node(state: GraphState) -> dict:
    merged = f"""
LangChain result:
{state.langchain_result}

LangGraph result:
{state.langgraph_result}
"""
    return {"result": merged}


# -------------------------------------------------
# 5️⃣  Wire the conditional edge – route without updating state
# -------------------------------------------------
def route_decision(state: GraphState):
    if state.decision == "both":
        return ["langchain", "langgraph"]
    return state.decision  # the value set by llm_call_router




# -------------------------------------------------
# 7️⃣  Assemble the graph (only the relevant parts are shown)
# -------------------------------------------------
builder = StateGraph(GraphState)

builder.add_node("LangChain", langchain_node)
builder.add_node("LangGraph", langgraph_node)
builder.add_node("Both", merge_node)

builder.add_edge(START, "LangChain")
builder.add_edge(START, "LangGraph")
builder.add_edge(START, "Both")

builder.add_conditional_edges("langchain", tools_condition)
builder.add_edge("tools", "langchain")
graph = builder.compile()


