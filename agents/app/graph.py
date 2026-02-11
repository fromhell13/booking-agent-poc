import os
import asyncio
from typing import TypedDict, List, Optional, Literal, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.mcp_client import get_mcp_tools

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "smallthinker")
llm = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL)

class State(TypedDict):
    messages: List[Dict[str, str]]
    text: str
    intent: Optional[str]
    tool_result: Optional[dict]
    answer: Optional[str]

def detect_intent(state: State) -> State:
    t = (state["text"] or "").lower()
    if any(k in t for k in ["menu", "price", "western", "asian", "beverage", "drink", "food"]):
        state["intent"] = "menu"
    elif any(k in t for k in ["book", "reserve", "reservation", "table", "availability", "cancel"]):
        state["intent"] = "booking"
    else:
        state["intent"] = "general"
    return state

async def _call_tool(tool_name: str, payload: dict) -> dict:
    tools = await get_mcp_tools()
    if tool_name not in tools:
        return {"error": "tool_not_found", "available": list(tools.keys())}
    return await tools[tool_name].ainvoke(payload)

def run_menu(state: State) -> State:
    state["tool_result"] = asyncio.run(_call_tool("query_menu", {"query": state["text"], "top_k": 3}))
    return state

def run_booking(state: State) -> State:
    # PoC: ask user for structured details if missing.
    # Later you can add an LLM extractor that calls booking_check_availability/booking_create.
    state["tool_result"] = {"note": "Booking intent detected. Implement entity extraction next (date/time/pax/name/phone)."}
    return state

def respond(state: State) -> State:
    msgs = [SystemMessage(content="You are a helpful booking assistant. Use tool_result as truth.")]
    for m in state.get("messages", [])[-10:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))

    msgs.append(HumanMessage(content=state["text"]))

    if state.get("tool_result"):
        msgs.append(SystemMessage(content=f"tool_result:\n{state['tool_result']}"))

    state["answer"] = llm.invoke(msgs).content
    return state

def build():
    g = StateGraph(State)
    g.add_node("detect", detect_intent)
    g.add_node("menu", run_menu)
    g.add_node("booking", run_booking)
    g.add_node("respond", respond)

    g.set_entry_point("detect")

    def route(s: State) -> Literal["menu", "booking", "respond"]:
        if s["intent"] == "menu":
            return "menu"
        if s["intent"] == "booking":
            return "booking"
        return "respond"

    g.add_conditional_edges("detect", route, {"menu": "menu", "booking": "booking", "respond": "respond"})
    g.add_edge("menu", "respond")
    g.add_edge("booking", "respond")
    g.add_edge("respond", END)
    return g.compile()

graph = build()
