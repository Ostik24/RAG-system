"""
Streamlit app for querying The Batch using RAG (Retrieval-Augmented Generation).
"""

from rag.rag_chain import answer_query_with_rag, vector_db
import streamlit as st
import json
import re

if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "vector_database" not in st.session_state:
    st.session_state.vector_database = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def extract_json_from_response(text: str) -> dict:
    """
    Extracts JSON from the response text.
    Args:
        text (str): The response text containing JSON data.
    Returns:
        dict: The extracted JSON data.
    """
    if text.lower().strip().startswith("json"):
        text = text[text.lower().find("{"):]
    text = text.strip().strip("`")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("Failed to extract valid JSON from response.")

st.header("Welcome!")
st.write("Here you can freely ask for any information you need about The Batch.")

api_key = st.empty()
st.write("Please enter your API key below and wait for a few minutes:")
api_key = st.text_input("API Key", type="password", placeholder="Enter your API key here")

if api_key and not st.session_state.vector_database and st.session_state.api_key != api_key:
    with st.spinner("Initializing vector database..."):
        st.session_state.vector_database = vector_db(api_key)
        st.success("Vector database initialized successfully!")
    st.session_state.api_key = api_key
    api_key = st.empty()

st.write("Please enter your question below:")
query = st.chat_input("Ask The Batch…")

if query and st.session_state.api_key:
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            given_answer = answer_query_with_rag(query.lower().strip(), st.session_state.api_key, st.session_state.vector_database)
            st.session_state.chat_history.append({"role": "assistant", "content": given_answer})

if st.session_state.chat_history:
    st.write("Chat History:")
    for msg in st.session_state.chat_history:
        with st.chat_message(role :=msg["role"]):
            if role == "assistant":
                js = extract_json_from_response(msg["content"])
                if js.get("answers"):
                    for answer in js["answers"]:
                        st.markdown(f"**Answer {answer['number']}:** {answer['text']}")
                        st.markdown(f"[{answer['title']}]({answer['url']})")
                        if answer.get("image_url"):
                            st.image(answer["image_url"])
                else:
                    st.markdown(msg["content"])

            else:
                st.markdown(msg["content"])

if st.session_state.chat_history and st.button("Clear chat history"):
    st.session_state.chat_history = []
    st.rerun()
