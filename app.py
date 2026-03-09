import streamlit as st
import sys
import os

# UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# API key — Streamlit Cloud secrets se, fallback .env se
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

from claude_crm import CRMManagerAgent

st.set_page_config(page_title="CRM Manager AI", page_icon="🗂", layout="centered")

st.title("CRM Manager AI")
st.caption("GreySpaces Retail Creative Strategist")

# ── Sidebar: Customer Context ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Customer Context")
    customer_name    = st.text_input("Customer",       value="Acme Corp")
    arr              = st.text_input("ARR",            value="$48,000")
    contact_name     = st.text_input("Contact Name",   value="Sarah Chen")
    last_contact     = st.text_input("Last Contact",   value="22 days ago")
    last_interaction = st.text_input("Last Interaction", value="Demo call — requested pricing")
    churn_risk       = st.selectbox("Churn Risk",      ["Low", "Medium", "High"], index=1)

    if st.button("Load Customer", use_container_width=True):
        st.session_state.customer_context = {
            "Customer":         customer_name,
            "ARR":              arr,
            "Contact name":     contact_name,
            "Last contact":     last_contact,
            "Last interaction": last_interaction,
            "Churn risk":       churn_risk,
        }
        # Reset agent with new context
        st.session_state.agent = CRMManagerAgent(api_key=api_key)
        st.session_state.agent.load_customer(st.session_state.customer_context)
        st.session_state.messages = []
        st.success("Customer loaded!")

    st.divider()
    if st.button("New Session", use_container_width=True):
        st.session_state.agent.new_session()
        st.session_state.messages = []
        st.rerun()

# ── Session State Init ────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = CRMManagerAgent(api_key=api_key)
    st.session_state.agent.load_customer({
        "Customer":         "Acme Corp",
        "ARR":              "$48,000",
        "Contact name":     "Sarah Chen",
        "Last contact":     "22 days ago",
        "Last interaction": "Demo call — requested pricing",
        "Churn risk":       "Medium",
    })

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = st.session_state.agent.chat(prompt)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
