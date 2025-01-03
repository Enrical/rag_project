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
        if st.button("enviar"):
            if st.session_state.password_input == st.secrets["APP_PASSWORD"]:
                st.session_state.password_verified = True
                st.success("Access granted!")
            else:
                st.error("Invalid password. Please try again.")
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

    def retrieve_chunks(self, query: str) -> List[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ragie_api_key}"
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
        Personalidad:
        Estructurado, con capacidad para manejar sistemas y herramientas administrativas, y con una actitud proactiva hacia la mejora de procesos. A la vez, demuestra una cierta flexibilidad y empatía en la gestión del equipo, asegurándose de que haya consenso y evitando conflictos innecesarios.
        /
        Objetivo: Responder preguntas sobre los documentos a los que tengo acceso de manera precisa y explicando con cercanía y familiaridad.
        /
        Enrique responde solo preguntas relacionadas con los documentos: {chunk_texts}.
        /
        Para cualquier otra pregunta responde: "Todavía no tengo ese conocimiento, pero seguiré aprendiendo para poder ser de más ayuda pronto."""

    def generate_response(self, system_prompt: str, query: str) -> str:
        messages = [
            {"role": "user", "content": query}
        ]
        response = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text


def load_documents():
    """Load documents from a JSON file."""
    try:
        with open("documents.json", "r") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading documents.json: {str(e)}")
        return {}


def initialize_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        ragie_key = st.secrets["RAGIE_API_KEY"]
        anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
        st.session_state.pipeline = RAGPipeline(ragie_key, anthropic_key)

    if 'document_sets' not in st.session_state:
        st.session_state.document_sets = load_documents()

    if 'current_client' not in st.session_state:
        st.session_state.current_client = None

    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = False

    if 'admin_mode' not in st.session_state:
        st.session_state.admin_mode = True


def admin_interface():
    st.sidebar.markdown("### Panel del Admin")

    if st.session_state.admin_mode:
        client = st.sidebar.selectbox(
            "Selecciona tu asistente",
            options= list(st.session_state.document_sets.keys())
        )

        if client != "Selecciona tu asistente":
            st.session_state.current_client = client
            st.session_state.uploaded_documents = st.session_state.document_sets.get(client, [])

            st.sidebar.markdown("### Documentos seleccionados")
            for doc in st.session_state.uploaded_documents:
                st.sidebar.markdown(f"- [**{doc['name']}**]({doc['url']})", unsafe_allow_html=True)

    toggle_button_label = "Switch to Chat Mode" if st.session_state.admin_mode else "Switch to Admin Mode"
    if st.sidebar.button(toggle_button_label):
        st.session_state.admin_mode = not st.session_state.admin_mode
        st.session_state.chat_mode = not st.session_state.chat_mode


def chat_interface():
    st.markdown(
        """
        <style>
        .user-message {
            color: black;
            font-weight: normal;
            margin-bottom: 10px;
        }
        .ai-message {
            color: black;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .st-key-chat_query{
            display: flex;
            flex-direction: column-reverse;
            overflow-y: auto;
            max-height: 60vh;
        }        
        /* Ensure child <p> elements inside .ai-message inherit the styles */
        .ai-message p {
        color: inherit; /* Use the color of the parent */
        font-weight: inherit; /* Use the font weight of the parent */
        }
        .stButton {
            display: flex;
            flex-direction: column-reverse;
            overflow-y: auto;
            max-height: 60vh;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🕵️‍♂️ Habla con Enrique AI")
    if not st.session_state.pipeline:
        st.error("The system is not configured yet. Please contact the administrator.")
        return

    # Display chat history dynamically
    chat_placeholder = st.empty()  # Placeholder to dynamically update chat history
    with chat_placeholder.container():
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            if role == "user":
                st.markdown(f'<div class="user-message">You: {content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message">🕵️‍♂️ Enrique AI: {content}</div>', unsafe_allow_html=True)

    # Input and form for handling Enter or button click
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("Escribe tu mensaje", value="", key="chat_query")
        submit_button = st.form_submit_button("Enviar")

    if submit_button:
        if query.strip():
            try:
            # Append the user's query to chat history
                st.session_state.chat_history.append({"role": "user", "content": query})

                with st.spinner("Generando respuesta..."):
                    # Retrieve relevant chunks and generate a response
                    chunks = st.session_state.pipeline.retrieve_chunks(query)
                    if not chunks:
                        response = "No relevant information found."
                    else:
                        system_prompt = st.session_state.pipeline.create_system_prompt(chunks)
                        response = st.session_state.pipeline.generate_response(system_prompt, query)

                    # Append the assistant's response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})


                # Refresh the chat history dynamically
                with chat_placeholder.container():
                    for message in st.session_state.chat_history:
                        role = message["role"]
                        content = message["content"]
                        if role == "user":
                            st.markdown(f'<div class="user-message">You: {content}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="ai-message">🕵️‍♂️ Enrique AI: {content}</div>', unsafe_allow_html=True)

                    # Clear the input field
                    #st.session_state.chat_query = ""

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
        else:
            st.error("Please enter a message.")

def main():
    st.set_page_config(page_title="Client Chat System",
                       page_icon="https://essent-ia.com/wp-content/uploads/2024/11/cropped-cropped-Picture1.png",
                       layout="centered")

    check_password()
    initialize_session_state()
    admin_interface()

    if st.session_state.chat_mode:
        chat_interface()


if __name__ == "__main__":
    main()
