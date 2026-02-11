import asyncio
import json
from typing import Dict, Any, List

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_mcp_adapters.client import MultiServerMCPClient


# ----------------------------
# Helpers
# ----------------------------
def is_menu_intent(text: str) -> bool:
    t = text.lower()
    keywords = ["menu", "cuisine", "western", "asian", "beverage", "drink", "price", "food"]
    return any(k in t for k in keywords)


async def get_mcp_tools(mcp_url: str):
    """
    Loads tools from your MCP server via Streamable HTTP.
    LangChain uses transport="http" for streamable-http MCP servers.
    """
    client = MultiServerMCPClient(
        {
            "booking-agent-tools": {
                "transport": "streamable_http",
                "url": mcp_url,
            }
        }
    )
    tools = await client.get_tools()
    # Return both for reuse
    return client, tools


async def call_tool_by_name(tools, name: str, tool_args: Dict[str, Any]) -> Any:
    tool_map = {t.name: t for t in tools}
    if name not in tool_map:
        return {"error": f"Tool '{name}' not found. Available: {list(tool_map)}"}
    # LangChain tools support ainvoke for async
    return await tool_map[name].ainvoke(tool_args)


def run(coro):
    """Run async code from Streamlit safely."""
    return asyncio.run(coro)


def format_hits(tool_result: Any) -> str:
    """
    Try to display query_menu output nicely.
    Tool result shape depends on your MCP adapter + server return.
    """
    # Usually tool_result is a dict already. If it's a string, try JSON.
    if isinstance(tool_result, str):
        try:
            tool_result = json.loads(tool_result)
        except Exception:
            return tool_result

    if isinstance(tool_result, dict) and "hits" in tool_result:
        hits = tool_result.get("hits") or []
        if not hits:
            return "No matching menu items found."
        lines = []
        for i, h in enumerate(hits, start=1):
            text = (h.get("text") or "").strip()
            md = h.get("metadata") or {}
            # Keep it short in UI
            text = text[:800]
            lines.append(f"{i}. {text}\n   metadata: {md}")
        return "\n\n".join(lines)

    # Fallback
    return json.dumps(tool_result, indent=2, ensure_ascii=False)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="MCP Chat Test (Streamable HTTP)", page_icon="🧪", layout="centered")
st.title("🧪 MCP Chat Test (Streamable HTTP)")

with st.sidebar:
    st.subheader("Connection")
    mcp_url = st.text_input("MCP URL", value="http://127.0.0.1:8000/mcp")
    st.caption("This should match your FastMCP streamable-http endpoint.")

    st.subheader("Model")
    ollama_base = st.text_input("Ollama base_url", value="http://127.0.0.1:11434")
    model_name = st.text_input("Ollama model", value="smallthinker")

    if st.button("Reload MCP tools"):
        st.session_state.pop("mcp_client", None)
        st.session_state.pop("mcp_tools", None)
        st.success("Cleared cached tools. They will reload on next message.")

# Cache MCP tools/client in session
if "mcp_tools" not in st.session_state:
    try:
        client, tools = run(get_mcp_tools(mcp_url))
        st.session_state.mcp_client = client
        st.session_state.mcp_tools = tools
    except Exception as e:
        st.error(f"Failed to load MCP tools from {mcp_url}: {e}")
        st.stop()

tools = st.session_state.mcp_tools
tool_names = [t.name for t in tools]

with st.sidebar:
    st.subheader("MCP tools discovered")
    st.write(tool_names)

    # Quick tool tests
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test: menu_count"):
            try:
                out = run(call_tool_by_name(tools, "menu_count", {}))
                st.json(out)
            except Exception as e:
                st.error(str(e))
    with col2:
        if st.button("Test: echo"):
            try:
                out = run(call_tool_by_name(tools, "echo", {"query": "hello"}))
                st.json(out)
            except Exception as e:
                st.error(str(e))

# LangChain LLM (Ollama)
llm = ChatOllama(model=model_name, base_url=ollama_base)

SYSTEM = (
    "You are a helpful assistant. If the user asks about the restaurant menu, "
    "use the provided tool results as the source of truth. "
    "If tool results are empty, say you couldn't find a match."
)

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hi! Ask me about the menu (e.g., 'show western cuisine').")]

# Render chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

prompt = st.chat_input("Type a message…")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        tool_context = ""
        raw_tool = None

        try:
            # If the user likely wants menu info, call MCP tool
            if is_menu_intent(prompt) and "query_menu" in tool_names:
                raw_tool = run(call_tool_by_name(tools, "query_menu", {"query": prompt}))
                tool_context = format_hits(raw_tool)

                st.markdown("**(MCP query_menu results)**")
                st.code(tool_context)

            # Now ask the LLM, injecting tool results as context
            msgs: List = [SystemMessage(content=SYSTEM)]
            # keep last ~10 messages for context
            history = st.session_state.messages[-10:]
            msgs.extend(history)

            if tool_context:
                msgs.append(SystemMessage(content=f"Tool results (menu search):\n{tool_context}"))

            answer = llm.invoke(msgs).content
            st.markdown(answer)
            st.session_state.messages.append(AIMessage(content=answer))

        except Exception as e:
            st.error(f"Error: {e}")
            # show raw tool output if we got any
            if raw_tool is not None:
                st.markdown("Raw tool output:")
                st.json(raw_tool)
