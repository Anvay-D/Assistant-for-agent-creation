import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Langchain RAG Chat", layout="centered")
st.title("💬 Agentic AI RAG Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
prompt = st.chat_input("Ask about Langchain API...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {
                "question": prompt,
                "session_id": st.session_state.session_id
            }
            res = requests.post(API_URL, json=payload)
            data = res.json()
            answer = data["answer"]
            # Sync session_id returned from backend (in case it changed)
            if "session_id" in data:
                st.session_state.session_id = data["session_id"]

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
