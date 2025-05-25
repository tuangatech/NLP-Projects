import streamlit as st
import json
import requests

BACKEND_URL = "http://localhost:8000/query"
formatted_history = []
NO_ANSWER = "I don't know."

st.set_page_config(page_title="Tax Chatbot", layout="centered")
st.title("TaxBot - Ask Me About Tax!")

# Chat history in session state (each item is a full exchange)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "show_spinner" not in st.session_state:
    st.session_state.show_spinner = False

# Sends the user input and chat history to the backend and updates the chat history
def get_response():
    recent_history = st.session_state.chat_history[:3]  # most recent first
    formatted_history = [
        {"question": item["question"], "answer": item["answer"]}
        for item in recent_history
    ]

    try:
        response = requests.post(
            BACKEND_URL,
            json={
                "question": user_input,
                "chat_history": formatted_history
            },
            timeout=20
        )

        if response.status_code == 200:
            try:
                json_response = response.json()
                answer = json_response.get("answer", "No answer found.")
                source_docs = json_response.get("source_documents", [])

                if answer == NO_ANSWER:
                    source_docs = []        # If no answer, why do we need source docs?
            except requests.exceptions.JSONDecodeError:
                answer = "⚠️ Invalid response from backend."
                source_docs = []
        else:
            answer = f"⚠️ Error: {response.text}"
            source_docs = []

        # Add to chat history
        st.session_state.chat_history.insert(0, {
            "question": user_input,
            "answer": answer,
            "source_docs": source_docs
        })

    except Exception as e:
        st.error(f"Connection error: {str(e)}")
    finally:
        # Clear spinner and refresh
        st.session_state.show_spinner = False
        spinner_placeholder.empty()
        st.rerun()

# Chat input area + Send button at the top
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask a question about tax filing:",
        value="What can TAS help me with?", #  What information that LITC can provide? Tell me what are taxpayer rights?
        key="input_text"
    )
    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        send_button = st.form_submit_button("Send")
    with col2:
        # Placeholder for the spinner
        spinner_placeholder = st.empty()

    # Handle submission
    if send_button and user_input.strip():
        st.session_state.show_spinner = True
        # Display spinner immediately
        with spinner_placeholder:
            with st.spinner("Thinking..."):
                get_response()

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        # Do not show source docs if no answer
        if chat.get("source_docs") and chat["answer"] != NO_ANSWER:
            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(chat["source_docs"]):
                    st.markdown(f"**Source {i+1}:** `{doc.get('file_name', 'Unknown')}`")
                    st.markdown(f"- Page: `{doc.get('page_number', 'Unknown')}` " \
                                f"| Section: `{doc.get('section_title', 'Unknown')}`")
                    content = doc.get('page_content', '').strip()
                    if content:
                        with st.container():
                            st.text(content[:300] + "..." if len(content) > 300 else content)
                    else:
                        st.markdown("*No content preview available.*")