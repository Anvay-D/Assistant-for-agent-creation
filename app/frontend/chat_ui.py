import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Langchain RAG Chat", layout="centered")
st.title("💬 Agentic AI RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
            res = requests.post(API_URL, json={"question": prompt})
            data = res.json()
            answer = data["answer"]

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
