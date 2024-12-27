import streamlit as st
import requests
from anthropic import Anthropic
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse


class RAGPipeline:
    def __init__(self, ragie_api_key: str, anthropic_api_key: str):
        self.ragie_api_key = ragie_api_key
        self.anthropic_api_key = anthropic_api_key
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)

        # API endpoints
        self.RAGIE_UPLOAD_URL = "https://api.ragie.ai/documents/url"
        self.RAGIE_RETRIEVAL_URL = "https://api.ragie.ai/retrievals"

    def upload_document(self, url: str, name: Optional[str] = None, mode: str = "fast") -> Dict:
        if not name:
            name = urlparse(url).path.split('/')[-1] or "document"

        payload = {
            "mode": mode,
            "name": name,
            "url": url
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.ragie_api_key}"
        }

        response = requests.post(self.RAGIE_UPLOAD_URL, json=payload, headers=headers)
        if not response.ok:
            raise Exception(f"Document upload failed: {response.status_code} {response.reason}")

        return response.json()

    def retrieve_chunks(self, query: str, scope: str = None) -> List[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ragie_api_key}"
        }

        payload = {
            "query": query,
            "filters": {
                "scope": scope  # Query across all uploaded documents
            }
        }

        response = requests.post(self.RAGIE_RETRIEVAL_URL, headers=headers, json=payload)
        if not response.ok:
            raise Exception(f"Retrieval failed: {response.status_code} {response.reason}")

        data = response.json()
        return [chunk["text"] for chunk in data["scored_chunks"]]

    def create_system_prompt(self, chunk_texts: List[str]) -> str:
        return f"""You are an assistant trained to help with document management and answer questions related to the uploaded documents. 
Here are the relevant pieces of information retrieved for your question: {chunk_texts}.
Provide an accurate and concise response using this information."""

    def generate_response(self, system_prompt: str, query: str, chat_history: List[Dict]) -> str:
        chat_history.append({"role": "user", "content": query})
        message = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_prompt,
            messages=chat_history
        )
        chat_history.append({"role": "assistant", "content": message.content[0].text})
        return message.content[0].text


def initialize_session_state():
    """Initialize all session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'document_uploaded' not in st.session_state:
        st.session_state.document_uploaded = False
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []  # Initialize as an empty list
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Initialize as an empty list


def main():
    st.set_page_config(page_title="Multi-Document Chat System", layout="centered")
    initialize_session_state()  # Ensure session state is initialized

    st.title("📄 Multi-Document Chat System")

    # API Keys Section
    with st.expander("🔑 Configure API Keys", expanded=not st.session_state.pipeline):
        ragie_key = st.text_input("Ragie API Key", type="password")
        anthropic_key = st.text_input("Anthropic API Key", type="password")

        if st.button("Submit API Keys"):
            if ragie_key and anthropic_key:
                st.session_state.pipeline = RAGPipeline(ragie_key, anthropic_key)
                st.success("API keys configured successfully!")
            else:
                st.error("Please provide both API keys.")

    # Document Upload Section
    if st.session_state.pipeline:
        st.markdown("### 📄 Upload Documents")
        doc_url = st.text_input("Enter Document URL")
        doc_name = st.text_input("Document Name (Optional)")
        upload_mode = st.selectbox("Upload Mode", ["fast", "accurate"])

        if st.button("Upload Document"):
            if doc_url:
                try:
                    with st.spinner("Uploading document..."):
                        response = st.session_state.pipeline.upload_document(
                            url=doc_url,
                            name=doc_name if doc_name else None,
                            mode=upload_mode
                        )
                        st.session_state.uploaded_documents.append({
                            "url": doc_url,
                            "name": doc_name or "Document",
                            "response": response
                        })
                        st.success(f"Document '{doc_name or doc_url}' uploaded successfully!")
                except Exception as e:
                    st.error(f"Error uploading document: {str(e)}")
            else:
                st.error("Please provide a document URL.")

    # Display Uploaded Documents
    if st.session_state.uploaded_documents:
        st.markdown("### Uploaded Documents")
        for doc in st.session_state.uploaded_documents:
            st.write(f"- **{doc['name']}** ({doc['url']})")

    # Chat Section
    if st.session_state.uploaded_documents:
        st.markdown("### 🔍 Chat with the Documents")

        # Display chat history
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(f"**You:** {chat['content']}")
                else:
                    st.markdown(f"**AI:** {chat['content']}")

        # Input for user query
        query = st.text_input("Enter your message", key="chat_query")

        if st.button("Send"):
            if query:
                try:
                    with st.spinner("Generating response..."):
                        # Process the query and update chat history
                        chunks = st.session_state.pipeline.retrieve_chunks(query)
                        if not chunks:
                            st.error("No relevant information found.")
                        else:
                            system_prompt = st.session_state.pipeline.create_system_prompt(chunks)
                            response = st.session_state.pipeline.generate_response(
                                system_prompt=system_prompt,
                                query=query,
                                chat_history=st.session_state.chat_history
                            )
                            # Update chat history
                            st.session_state.chat_history.append({"role": "user", "content": query})
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            else:
                st.error("Please enter a message.")


if __name__ == "__main__":
    main()
