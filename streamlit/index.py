import os
import requests
import streamlit as st

AGENT_URL = os.getenv("AGENT_URL", "http://agent:8080")
# (connect_timeout_sec, read_timeout_sec) — local Ollama can exceed 5m on cold CPU
_AGENT_TIMEOUT = os.getenv("AGENT_REQUEST_TIMEOUT", "600")
try:
    _t = float(_AGENT_TIMEOUT)
    REQUEST_TIMEOUT = (15.0, _t)
except ValueError:
    REQUEST_TIMEOUT = (15.0, 600.0)

st.set_page_config(page_title="Booking Agent", page_icon="🍽️")
st.title("🍽️ Booking Agent")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask about the menu, or say you want to reserve a table."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        payload = {"messages": st.session_state.messages[-10:], "text": prompt}
        try:
            # Ollama cold-start / first request can exceed 60s on CPU.
            r = requests.post(f"{AGENT_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            answer = data.get("answer", "")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.expander("Debug (tool_result)"):
                st.json(data.get("tool_result"))
        except Exception as e:
            st.error(f"Agent error: {e}")
