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
    st.header("Project Context")
    brand_name       = st.text_input("Brand Name",         value="")
    project_name     = st.text_input("Project Name",       value="")
    authorised_person= st.text_input("Authorised Person",  value="")
    category         = st.text_input("Category",           value="")
    timeline         = st.text_input("Timeline",           value="")
    project_brief    = st.text_area("Project Brief / Objective", value="", height=100)

    if st.button("Load Project", use_container_width=True):
        st.session_state.customer_context = {
            "Brand Name":             brand_name,
            "Project Name":           project_name,
            "Authorised Person":      authorised_person,
            "Category":               category,
            "Timeline":               timeline,
            "Project Brief":          project_brief,
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
    st.session_state.agent.load_customer({})

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
