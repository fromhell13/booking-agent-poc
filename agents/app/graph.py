"""
LangGraph workflow with split agents for security:
- Menu agent: RAG only (query_menu, menu_count)
- Booking-read agent: read-only DB (check_availability, list)
- Booking-write agent: create/cancel only
Each node uses only its scoped MCP tools.
"""
import json
import os
import re
import asyncio
import logging

import agentops
from agentops.sdk.decorators import tool

from datetime import datetime
from typing import TypedDict, List, Optional, Literal, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.mcp_client import get_mcp_tools, Scope

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv
load_dotenv()


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "smallthinker")
_ollama_http_timeout = float(os.getenv("OLLAMA_HTTP_TIMEOUT", "600"))
_ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "384"))
llm = ChatOllama(
    model=OLLAMA_CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    num_predict=_ollama_num_predict,
    client_kwargs={"timeout": _ollama_http_timeout},
)

# AgentOps init
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
agentops.init(
        api_key=AGENTOPS_API_KEY,
        default_tags=["langgraph"],
)
logger.info(f"AgentOps initialized with API key: {AGENTOPS_API_KEY}")

# Optional extracted slots for booking tools
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
TIME_RE = re.compile(r"\b(\d{1,2}:\d{2})\b")
_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}
PHONE_RE = re.compile(r"\+?[\d\s\-]{6,20}")
PAX_PATTERNS = [
    re.compile(r"\b(?:for|pax|people|persons|guests)\s*[:=]?\s*(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})\s*(?:pax|people|persons|guests)\b", re.IGNORECASE),
]
NAME_PATTERNS = [
    re.compile(r"\bname\s*(?:is|:)\s*([A-Za-z][A-Za-z\s\.'-]{0,118})\b", re.IGNORECASE),
    re.compile(r"\b(?:i am|i'm|this is)\s*([A-Za-z][A-Za-z\s\.'-]{0,118})\b", re.IGNORECASE),
]


def parse_date_from_text(text: str) -> Optional[str]:
    """Return YYYY-MM-DD or None. Handles ISO plus common English dates."""
    if not text:
        return None
    m = DATE_RE.search(text)
    if m:
        return m.group(1)
    tl = text.lower()
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s+(\d{4})\b", tl)
    if m:
        day, mon_s, year = int(m.group(1)), m.group(2), int(m.group(3))
        mon = _MONTH_NAMES.get(mon_s)
        if mon:
            try:
                return datetime(year, mon, day).date().isoformat()
            except ValueError:
                pass
    m = re.search(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", tl)
    if m:
        mon_s, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        mon = _MONTH_NAMES.get(mon_s)
        if mon:
            try:
                return datetime(year, mon, day).date().isoformat()
            except ValueError:
                pass
    return None


def parse_time_from_text(text: str) -> Optional[str]:
    """Return HH:MM 24-hour or None."""
    if not text:
        return None
    m = TIME_RE.search(text)
    if m:
        parts = m.group(1).split(":")
        return f"{int(parts[0]):02d}:{parts[1]}"
    m = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)\b", text, re.IGNORECASE)
    if m:
        h, mi, ap = int(m.group(1)), m.group(2), m.group(3).lower()
        ap = ap.replace(".", "")
        if ap in ("pm", "p"):
            if h != 12:
                h += 12
        elif ap in ("am", "a"):
            if h == 12:
                h = 0
        return f"{h:02d}:{mi}"
    m = re.search(r"\b(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)\b", text, re.IGNORECASE)
    if m:
        h, ap = int(m.group(1)), m.group(2).lower().replace(".", "")
        if ap in ("pm", "p") and h != 12:
            h += 12
        if ap in ("am", "a") and h == 12:
            h = 0
        return f"{h:02d}:00"
    return None


def _normalize_tool_result(raw: Any) -> dict:
    """Unwrap MCP streamable-http payloads that nest JSON in content[].text."""
    if isinstance(raw, list):
        raw = {"content": raw}
    if isinstance(raw, dict) and "content" in raw and isinstance(raw["content"], list):
        for block in raw["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            t = block.get("text")
            if not isinstance(t, str):
                continue
            try:
                parsed = json.loads(t)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return raw if isinstance(raw, dict) else {"raw": raw}
    if isinstance(raw, dict):
        return raw
    return {"content": raw}


class State(TypedDict):
    messages: List[Dict[str, str]]
    text: str
    booking_form: Optional[Dict[str, Any]]
    intent: Optional[str]
    tool_result: Optional[dict]
    answer: Optional[str]


def detect_intent(state: State) -> State:
    t = (state["text"] or "").lower()
    if any(k in t for k in ["menu", "price", "western", "asian", "beverage", "drink", "food", "dish"]):
        state["intent"] = "menu"
    elif any(k in t for k in ["availability", "available", "free", "slot", "check"]):
        state["intent"] = "booking_read"
    elif any(k in t for k in ["book", "reserve", "reservation", "table", "cancel", "create"]):
        state["intent"] = "booking_write"
    else:
        state["intent"] = "general"
    return state


async def _call_tool(scope: Scope, tool_name: str, payload: dict) -> dict:
    tools = await get_mcp_tools(scope)
    if tool_name not in tools:
        return {"error": "tool_not_found", "allowed": list(tools.keys())}
    raw = await tools[tool_name].ainvoke(payload)
    return _normalize_tool_result(raw)


@tool(name="query_menu")
def run_menu(state: State) -> State:
    result = asyncio.run(
        _call_tool("menu", "query_menu", {"query": state["text"], "top_k": 3})
    )
    state["tool_result"] = result
    return state

@tool(name="booking_check_availability")
def run_booking_read(state: State) -> State:
    text = state.get("text") or ""
    date = parse_date_from_text(text)
    time = parse_time_from_text(text)

    if date and time:
        result = asyncio.run(
            _call_tool(
                "booking_read",
                "booking_check_availability",
                {"date": date, "time": time, "max_tables": 10},
            )
        )
    else:
        # Do not call booking_list here — missing ISO date/time used to return unrelated rows.
        result = {
            "needs_datetime": True,
            "parsed_date": date,
            "parsed_time": time,
            "note": "Could not read date and time together. Use e.g. 10 April 2026 at 6:00 PM or 2026-04-10 and 18:00.",
        }
    state["tool_result"] = result
    return state

@tool(name="booking_create")
def run_booking_write(state: State) -> State:
    text = state.get("text") or ""
    form = state.get("booking_form") or {}

    # Extract booking details from all prior user turns so users can provide info gradually.
    user_texts = [m.get("content", "") for m in state.get("messages", []) if m.get("role") == "user"]
    combined = " ".join(user_texts + [text]).strip()

    form_name = (form.get("name") or "").strip() if isinstance(form.get("name"), str) else None
    form_phone = (form.get("phone") or "").strip() if isinstance(form.get("phone"), str) else None
    form_date = (form.get("date") or "").strip() if isinstance(form.get("date"), str) else None
    form_time = (form.get("time") or "").strip() if isinstance(form.get("time"), str) else None
    form_notes = (form.get("notes") or "").strip() if isinstance(form.get("notes"), str) else ""

    date = form_date or parse_date_from_text(combined)
    time = form_time or parse_time_from_text(combined)
    phone_matches = PHONE_RE.findall(combined)
    pax_match = None
    for p in PAX_PATTERNS:
        pax_match = p.search(combined)
        if pax_match:
            break
    name_match = None
    for p in NAME_PATTERNS:
        name_match = p.search(combined)
        if name_match:
            break

    phone = form_phone or (phone_matches[-1].strip() if phone_matches else None)
    name = form_name or (name_match.group(1).strip() if name_match else None)

    pax = None
    form_pax = form.get("pax")
    if isinstance(form_pax, int):
        pax = form_pax
    elif isinstance(form_pax, str) and form_pax.isdigit():
        pax = int(form_pax)
    elif pax_match:
        pax = int(pax_match.group(1))

    # Simple heuristic: "cancel" + number -> cancel; else create path
    if "cancel" in text.lower() and not form:
        ids = re.findall(r"\b(\d{1,6})\b", text)
        if ids:
            result = asyncio.run(
                _call_tool("booking_write", "booking_cancel", {"reservation_id": int(ids[0])})
            )
        else:
            result = {"error": "Provide reservation id to cancel (e.g. cancel 42)"}
    else:
        missing: list[str] = []
        if not name:
            missing.append("name")
        if not phone:
            missing.append("phone")
        if not date:
            missing.append("date (YYYY-MM-DD)")
        if not time:
            missing.append("time (HH:MM, 24-hour)")
        if not pax:
            missing.append("pax")

        # Never create a booking until all required fields are present.
        if missing:
            result = {
                "needs_booking_fields": True,
                "missing_fields": missing,
                "note": "Please provide the missing details in one message.",
                "example": "name: John Tan, phone: +60123456789, date: 2026-04-10, time: 18:00, pax: 2",
            }
        else:
            result = asyncio.run(
                _call_tool(
                    "booking_write",
                    "booking_create",
                    {"name": name, "phone": phone, "date": date, "time": time, "pax": pax, "notes": form_notes},
                )
            )
    state["tool_result"] = result
    return state

@tool(name="respond")
def respond(state: State) -> State:
    tr = state.get("tool_result")
    if state.get("intent") == "booking_read" and isinstance(tr, dict):
        if tr.get("needs_datetime"):
            parts = []
            if not tr.get("parsed_date"):
                parts.append("date")
            if not tr.get("parsed_time"):
                parts.append("time")
            need = " and ".join(parts) if parts else "date and time"
            state["answer"] = (
                f"Final Solution: Please provide the {need} "
                "(e.g. 10 April 2026 at 6:00 PM or 2026-04-10 and 18:00)."
            )
            return state
        if "available" in tr:
            avail = bool(tr.get("available"))
            rem = tr.get("remaining_tables", 0)
            d, tm = tr.get("date"), tr.get("time")
            state["answer"] = (
                f"Final Solution: {'Yes' if avail else 'No'} — {rem} table(s) still available on {d} at {tm}."
            )
            return state
    if state.get("intent") == "booking_write" and isinstance(tr, dict):
        if tr.get("needs_booking_fields"):
            missing = tr.get("missing_fields") or []
            if isinstance(missing, list) and missing:
                state["answer"] = (
                    "Final Solution: Please provide these missing details to complete your reservation: "
                    + ", ".join(str(x) for x in missing)
                    + "."
                )
            else:
                state["answer"] = "Final Solution: Please provide your name, phone, date, time, and pax."
            return state

        if tr.get("ok") is True and tr.get("id"):
            state["answer"] = f"Final Solution: Done. Reservation {tr.get('id')} has been cancelled."
            return state

        if tr.get("id") and tr.get("name") and tr.get("date") and tr.get("time"):
            state["answer"] = (
                f"Final Solution: Reservation confirmed for {tr.get('name')} on {tr.get('date')} at "
                f"{tr.get('time')} for {tr.get('pax')} pax. Booking ID: {tr.get('id')}."
            )
            return state

        if tr.get("error"):
            state["answer"] = f"Final Solution: {tr.get('error')}"
            return state

    msgs = [SystemMessage(content="""
        You are a booking and menu assistant. Use tool_result as the single source of truth.

        Global style rules:
        - Keep output concise and practical.
        - NEVER expose internal reasoning, thought process, analysis steps, or planning text.
        - Do not write phrases like "let me think", "first", "now", "to answer this", or "here's what I have so far".
        - Return only the final answer.
        - Do not repeat the user's request.
        - Never invent menu items, prices, or availability.

        Menu response rules (generic for any cuisine):
        - If user asks for a cuisine/category (e.g., asian, western, beverage), return ONLY items from that cuisine/category.
        - Exclude items from other cuisines/categories unless user asks for them.
        - If no matching items are found in tool_result, say that clearly and ask one short clarification.
        - Include price only when available in tool_result.
        - For general menu requests, return up to 6 items total.

        Booking response rules:
        - Be direct and action-oriented.
        - A reservation requires ALL fields: name, phone, date, time, pax.
        - Dietary restrictions/notes are optional; never ask them as required information.
        - Do not block or delay booking because dietary restrictions are missing.
        - If tool_result indicates missing booking fields, ask for those exact missing fields only.
        - Do not confirm booking unless booking_create has succeeded.

        Output format:
        - Keep total response under 120 words.
        - For menu responses, use this format exactly:
          Menu:
          - <item 1>
          - <item 2>
          - <item 3>
          (up to 6 items)
        - For booking/other responses, use one short paragraph.
        """)]
    for m in state.get("messages", [])[-10:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))

    if state.get("tool_result"):
        msgs.append(SystemMessage(content=f"tool_result:\n{state['tool_result']}"))

    state["answer"] = llm.invoke(msgs).content
    return state


def build():
    g = StateGraph(State)
    g.add_node("detect", detect_intent)
    g.add_node("menu", run_menu)
    g.add_node("booking_read", run_booking_read)
    g.add_node("booking_write", run_booking_write)
    g.add_node("respond", respond)

    g.set_entry_point("detect")

    def route(s: State) -> Literal["menu", "booking_read", "booking_write", "respond"]:
        if s["intent"] == "menu":
            return "menu"
        if s["intent"] == "booking_read":
            return "booking_read"
        if s["intent"] == "booking_write":
            return "booking_write"
        return "respond"

    g.add_conditional_edges(
        "detect",
        route,
        {"menu": "menu", "booking_read": "booking_read", "booking_write": "booking_write", "respond": "respond"},
    )
    g.add_edge("menu", "respond")
    g.add_edge("booking_read", "respond")
    g.add_edge("booking_write", "respond")
    g.add_edge("respond", END)
    return g.compile()


graph = build()
