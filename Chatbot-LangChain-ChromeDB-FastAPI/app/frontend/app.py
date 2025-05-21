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

# Chat input area + Send button at the top
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask a question about tax filing:",
        value="What can Tax Accounting Service (TAS) help me with? What information that Low Income Taxpayer Clinics can provide? Tell me what are taxpayer rights?",
        key="input_text"
    )
    send_button = st.form_submit_button("Send")

# Handle submission
if send_button and user_input.strip():
    # Limit chat history to last 3 Q&A pairs
    recent_history = st.session_state.chat_history[:3]  # most recent first

    # Format history as list of {"question": ..., "answer": ...}
    formatted_history = [
        {"question": item["question"], "answer": item["answer"]}
        for item in recent_history
    ]
    # Prepare request
    try:
        with st.spinner("Thinking..."):
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
                    source_docs = []  # Don't care about source documents if no answer
                
            except requests.exceptions.JSONDecodeError:
                answer = "⚠️ Invalid response from backend."
                source_docs = []
        else:
            answer = f"⚠️ Error: {response.text}"
            source_docs = []
    except Exception as e:
        answer = f"⚠️ Connection error: {str(e)}"
        source_docs = []
        
    st.session_state.chat_history.insert(0, {
        "question": user_input,
        "answer": answer,
        "source_docs": source_docs
    })

    # Rerun to display the result
    st.rerun()

# Display chat history below input form
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        # Do not show source docs if no anwer
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

