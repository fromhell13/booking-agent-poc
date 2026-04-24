import os
import requests
import streamlit as st
from datetime import date, time

AGENT_URL = os.getenv("AGENT_URL", "http://agent:8080")
# (connect_timeout_sec, read_timeout_sec) — local Ollama can exceed 5m on cold CPU
_AGENT_TIMEOUT = os.getenv("AGENT_REQUEST_TIMEOUT", "600")
try:
    _t = float(_AGENT_TIMEOUT)
    REQUEST_TIMEOUT = (15.0, _t)
except ValueError:
    REQUEST_TIMEOUT = (15.0, 600.0)

st.set_page_config(page_title="Booking Agent", page_icon="🍽️")
st.title("Booking Agent")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask about the menu, or say you want to reserve a table."}]
if "show_reservation_form" not in st.session_state:
    st.session_state.show_reservation_form = False


def _looks_like_booking_request(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["reserve", "reservation", "book", "booking", "table"])

def _looks_like_cancel_request(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["cancel", "cancellation"])


def _send_to_agent(user_text: str, booking_form: dict | None = None) -> dict | None:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Agent is processing your request..."):
            payload = {"messages": st.session_state.messages[-10:], "text": user_text, "booking_form": booking_form}
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
                return data
            except Exception as e:
                st.error(f"Agent error: {e}")
                return None


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type here...")

if prompt:
    data = _send_to_agent(prompt)
    tool_result = (data or {}).get("tool_result") if isinstance(data, dict) else None
    should_show_form = False
    should_hide_form = _looks_like_cancel_request(prompt)

    if should_hide_form:
        if st.session_state.show_reservation_form:
            st.session_state.show_reservation_form = False
            st.rerun()
    else:
        if _looks_like_booking_request(prompt):
            should_show_form = True
        if isinstance(tool_result, dict) and tool_result.get("needs_booking_fields"):
            should_show_form = True
    if should_show_form and not st.session_state.show_reservation_form:
        st.session_state.show_reservation_form = True
        st.rerun()

if st.session_state.show_reservation_form:
    with st.expander("Reservation Form", expanded=True):
        with st.form("reservation_form", clear_on_submit=False):
            name = st.text_input("Name")
            phone = st.text_input("Phone")
            booking_date = st.date_input("Date", value=date.today())
            booking_time = st.time_input("Time", value=time(18, 0))
            pax = st.number_input("Pax", min_value=1, max_value=20, value=2, step=1)
            notes = st.text_area("Notes (optional)")
            submit_booking = st.form_submit_button("Submit Reservation")

        if submit_booking:
            form_payload = {
                "name": name,
                "phone": phone,
                "date": booking_date.isoformat(),
                "time": booking_time.strftime("%H:%M"),
                "pax": int(pax),
                "notes": notes,
            }
            summary = (
                "I want to reserve a table.\n"
                f"- name: {name}\n"
                f"- phone: {phone}\n"
                f"- date: {booking_date.isoformat()}\n"
                f"- time: {booking_time.strftime('%H:%M')}\n"
                f"- pax: {int(pax)}\n"
                f"- notes: {notes or '-'}"
            )
            data = _send_to_agent(summary, booking_form=form_payload)
            tool_result = (data or {}).get("tool_result") if isinstance(data, dict) else None
            # Close form after submit; reopen only if backend asks for missing fields.
            st.session_state.show_reservation_form = False
            if isinstance(tool_result, dict) and tool_result.get("needs_booking_fields"):
                st.session_state.show_reservation_form = True
            st.rerun()
