import streamlit as st
import requests
from datetime import datetime
from typing import List, Dict

class RAGPipeline:
    def __init__(self, ragie_api_key: str, anthropic_api_key: str):
        self.ragie_api_key = ragie_api_key
        self.anthropic_api_key = anthropic_api_key
        self.RAGIE_UPLOAD_URL = "https://api.ragie.ai/documents/url"

    def upload_document(self, document_content: str, name: str) -> Dict:
        """Uploads a document to Ragie."""
        payload = {
            "mode": "fast",
            "name": name,
            "content": document_content
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.ragie_api_key}"
        }
        response = requests.post(self.RAGIE_UPLOAD_URL, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to upload document: {response.status_code} {response.text}")
        return response.json()

# Initialize session state for conversation management
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = {
            "id": f"conversation_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "messages": []
        }

# Save the conversation to Ragie
def save_conversation_to_ragie(rag_pipeline: RAGPipeline):
    conversation = st.session_state.current_conversation
    # Format conversation into a plain text document
    document_content = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation["messages"]]
    )
    # Upload to Ragie
    response = rag_pipeline.upload_document(
        document_content=document_content,
        name=f"{conversation['id']}.txt"
    )
    st.success(f"Conversation saved successfully! Document ID: {response['id']}")

# Chat Interface
def chat_interface(rag_pipeline: RAGPipeline):
    st.markdown("### 🕵️‍♂️ Chat with Enrique AI")
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for msg in st.session_state.current_conversation["messages"]:
            if msg["role"] == "user":
                st.markdown(f"**You**: {msg['content']}")
            else:
                st.markdown(f"**AI**: {msg['content']}")

    # Chat Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message:")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input.strip():
        # Append user message to chat history
        st.session_state.current_conversation["messages"].append({"role": "user", "content": user_input})
        
        # Generate AI response
        with st.spinner("Generating response..."):
            try:
                response_text = generate_ai_response(user_input, st.session_state.current_conversation["messages"])
                st.session_state.current_conversation["messages"].append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Save Conversation Button
    if st.button("Save Conversation"):
        save_conversation_to_ragie(rag_pipeline)

# Generate AI Response (Dummy Example)
def generate_ai_response(user_input: str, conversation_history: List[Dict]) -> str:
    # Replace this logic with Claude's API call
    return f"This is a dummy response to your query: {user_input}"

# Main App
def main():
    st.set_page_config(page_title="Enrique AI with Conversation Saving", layout="centered")
    initialize_session_state()

    # Initialize Ragie Pipeline
    rag_pipeline = RAGPipeline(
        ragie_api_key=st.secrets["RAGIE_API_KEY"],
        anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
    )

    chat_interface(rag_pipeline)

if __name__ == "__main__":
    main()
