import streamlit as st
import requests
from anthropic import Anthropic
from urllib.parse import urlparse
from typing import List, Dict, Optional
import json


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
Objetivo: Responder preguntas sobre el proceso de gestión de las vacaciones.
/
Tono: Actúa con un tono familiar, accesible y profesional. Responde con claridad y precisión, ofreciendo primero una respuesta breve y directa a las preguntas, y después, si es útil, amplía la información o hace preguntas adicionales para entender mejor el caso. Da respuestas claras cuando la información esté disponible.
/
Enrique se asegura de facilitar temas complejos con ejemplos claros y prácticos cuando es necesario.
Responde solo preguntas relacionadas con los documentos {chunk_texts}.
/
Para cualquier otra pregunta responde: "Todavía no tengo ese conocimiento, pero seguiré aprendiendo de Enrique para poder ser de más ayuda pronto"."""    

    def generate_response(self, system_prompt: str, query: str, chat_history: List[Dict]) -> str:
        """
        Generate a response using Anthropic's Claude model.
        """
        chat_history.append({"role": "user", "content": query})
        message = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_prompt,
            messages=chat_history
        )
        chat_history.append({"role": "assistant", "content": message.content[0].text})
        return message.content[0].text


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
        ragie_key = "your_ragie_api_key"
        anthropic_key = "your_anthropic_api_key"
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
    st.sidebar.markdown("### Admin Configuration")
    ragie_key = st.sidebar.text_input("Ragie API Key", type="password")
    anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password")

    if st.sidebar.button("Save API Keys"):
        if ragie_key and anthropic_key:
            st.session_state.pipeline = RAGPipeline(ragie_key, anthropic_key)
            st.success("API keys saved successfully!")
        else:
            st.error("Please provide both API keys.")

    # Select client
    client = st.sidebar.selectbox(
        "Select Client",
        options=["Select a Client"] + list(st.session_state.document_sets.keys())
    )

    if client != "Select a Client":
        st.session_state.current_client = client
        st.session_state.uploaded_documents = st.session_state.document_sets[client]

        if st.sidebar.button("Switch to Chat Mode"):
            st.session_state.admin_mode = False
            st.session_state.chat_mode = True

    if st.session_state.current_client:
        st.sidebar.markdown("### Selected Documents")
        for doc in st.session_state.uploaded_documents:
            st.sidebar.markdown(f"- **{doc['name']}** ({doc['url']})")


def chat_interface():
    # Add custom CSS styles for chat
    st.markdown(
        """
        <style>
        .user-message {
            color: black;
            font-weight: normal;
            margin-bottom: 10px;
        }
        .ai-message {
            color: blue;
            font-weight: bold;
            margin-bottom: 10px;
        }
        /* Ensure child <p> elements inside .ai-message inherit the styles */
        .ai-message p {
        color: inherit; /* Use the color of the parent */
        font-weight: inherit; /* Use the font weight of the parent */
        margin-top: 5px; /* Optional: Adjust margins for spacing */
}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Chat with Enrique AI")
    if not st.session_state.pipeline:
        st.error("The system is not configured yet. Please contact the administrator.")
        return

    # Display chat history with CSS classes
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                # User query styled with CSS
                st.markdown(
                    f'<div class="user-message">You: {chat["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                # AI response styled with CSS
                st.markdown(
                    f'<div class="ai-message">Enrique AI: {chat["content"]}</div>',
                    unsafe_allow_html=True
                )

    # Input for user query
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""

    # Render the text input
    query = st.text_input("Enter your message", value=st.session_state.current_query, key="chat_query")

    # Handle query submission
    if st.button("Send"):
        if query.strip():  # Ensure the input is not empty
            try:
                with st.spinner("Generating response..."):
                    # Retrieve relevant chunks
                    chunks = st.session_state.pipeline.retrieve_chunks(query)
                    if not chunks:
                        st.error("No relevant information found.")
                    else:
                        # Generate the system prompt
                        system_prompt = st.session_state.pipeline.create_system_prompt(chunks)

                        # Generate the response
                        response = st.session_state.pipeline.generate_response(
                            system_prompt, query, st.session_state.chat_history
                        )

                        # Update chat history
                        st.session_state.chat_history.append({"role": "user", "content": query})
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

                        # Clear the input field
                        st.session_state.current_query = ""  # Reset the input state
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
        else:
            st.error("Please enter a message.")

    # Add a "Switch to Admin Mode" button
    if st.button("Switch to Admin Mode"):
        st.session_state.chat_mode = False
        st.session_state.admin_mode = True


def main():
    st.set_page_config(page_title="Client Chat System",
                       page_icon="https://essent-ia.com/wp-content/uploads/2024/11/cropped-cropped-Picture1.png",
                       layout="centered")
    initialize_session_state()

    if st.session_state.admin_mode:
        admin_interface()
    elif st.session_state.chat_mode:
        chat_interface()


if __name__ == "__main__":
    main()
