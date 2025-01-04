import streamlit as st
import requests
from anthropic import Anthropic
from urllib.parse import urlparse
from typing import List, Dict, Optional
import json


def check_password():
    """Display a password input box and verify access."""
    if "password_verified" not in st.session_state:
        st.session_state.password_verified = False

    if not st.session_state.password_verified:
        st.text_input("Escribe tu contraseña", type="password", key="password_input")
        if st.button("Enviar"):
            if st.session_state.password_input == st.secrets["APP_PASSWORD"]:
                st.session_state.password_verified = True
                st.session_state.admin_mode = True  # Automatically enable admin mode
                st.success("Acceso concedido!")
            else:
                st.error("Contraseña inválida. Por favor, inténtalo de nuevo.")
        st.stop()


class RAGPipeline:
    def __init__(self, ragie_api_key: str, anthropic_api_key: str):
        self.ragie_api_key = ragie_api_key
        self.anthropic_api_key = anthropic_api_key
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)

        self.RAGIE_UPLOAD_URL = "https://api.ragie.ai/documents/url"
        self.RAGIE_RETRIEVAL_URL = "https://api.ragie.ai/retrievals"

    def upload_document(self, url: str, name: Optional[str] = None, mode: str = "fast") -> Dict:
        if not name:
            name = urlparse(url).path.split('/')[-1] or "document"

        payload = {"mode": mode, "name": name, "url": url}
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.ragie_api_key}",
        }

        response = requests.post(self.RAGIE_UPLOAD_URL, json=payload, headers=headers)
        if not response.ok:
            raise Exception(f"Document upload failed: {response.status_code} {response.reason}")

        return response.json()

    def retrieve_chunks(self, query: str) -> List[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ragie_api_key}",
        }
        payload = {"query": query}

        response = requests.post(self.RAGIE_RETRIEVAL_URL, headers=headers, json=payload)
        if not response.ok:
            raise Exception(f"Retrieval failed: {response.status_code} {response.reason}")

        data = response.json()
        return [chunk["text"] for chunk in data["scored_chunks"]]

    def create_system_prompt(self, chunk_texts: List[str]) -> str:
        return f"""Este asistente, Enrique, es el asistente interno de la Gestoría Mays para el puesto de gerente de la Gestoría.
/
Enrique responde solo preguntas relacionadas con los documentos: {chunk_texts}.
/
Para cualquier otra pregunta responde: "Todavía no tengo ese conocimiento, pero seguiré aprendiendo para poder ser de más ayuda pronto."""


    def generate_response(self, system_prompt: str, query: str, conversation_history: list = None) -> str:
        if conversation_history is None:
            conversation_history = []
        messages = conversation_history + [{"role": "user", "content": query}]
        response = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text


def initialize_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        ragie_key = st.secrets["RAGIE_API_KEY"]
        anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
        st.session_state.pipeline = RAGPipeline(ragie_key, anthropic_key)

    if "document_sets" not in st.session_state:
        st.session_state.document_sets = {}

    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = None


def admin_interface():
    st.sidebar.markdown("### Panel del Administración")
    if st.sidebar.button("Nueva Conversación"):
        new_id = f"Conversación {len(st.session_state.conversations) + 1}"
        st.session_state.conversations[new_id] = []
        st.session_state.current_conversation = new_id

    conversation_list = list(st.session_state.conversations.keys())
    if conversation_list:
        selected_conversation = st.sidebar.selectbox(
            "Selecciona una conversación", conversation_list
        )
        if selected_conversation:
            st.session_state.current_conversation = selected_conversation

    st.sidebar.markdown(f"**Conversación activa**: {st.session_state.current_conversation}")


def chat_interface():
    st.markdown("### 🕵️‍♂️ Habla con Enrique AI")

    if not st.session_state.current_conversation:
        st.info("Por favor selecciona o crea una nueva conversación.")
        return

    current_history = st.session_state.conversations[st.session_state.current_conversation]
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for message in current_history:
            role = message["role"]
            content = message["content"]
            if role == "user":
                st.markdown(f"You: {content}")
            else:
                st.markdown(f"AI: {content}")

    query = st.text_input("Escribe tu mensaje")
    if st.button("Enviar"):
        if query.strip():
            current_history.append({"role": "user", "content": query})
            with st.spinner("Generando respuesta..."):
                chunks = st.session_state.pipeline.retrieve_chunks(query)
                if chunks:
                    system_prompt = st.session_state.pipeline.create_system_prompt(chunks)
                    response = st.session_state.pipeline.generate_response(
                        system_prompt, query, current_history
                    )
                else:
                    response = "No relevant information found."
                current_history.append({"role": "assistant", "content": response})

            # Refresh the chat
            with chat_placeholder.container():
                for message in current_history:
                    role = message["role"]
                    content = message["content"]
                    if role == "user":
                        st.markdown(f"You: {content}")
                    else:
                        st.markdown(f"AI: {content}")


def main():
    st.set_page_config(
        page_title="Client Chat System",
        page_icon="https://essent-ia.com/wp-content/uploads/2024/11/cropped-cropped-Picture1.png",
        layout="centered",
    )
    check_password()
    initialize_session_state()
    admin_interface()
    chat_interface()


if __name__ == "__main__":
    main()
