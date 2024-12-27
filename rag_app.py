import streamlit as st
import requests
from anthropic import Anthropic
from urllib.parse import urlparse
from typing import List, Dict, Optional


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
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'admin_mode' not in st.session_state:
        st.session_state.admin_mode = False
    if 'chat_mode' not in st.session_state:  # New state for client chat mode
        st.session_state.chat_mode = False


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

    st.sidebar.markdown("### Document Upload")
    doc_url = st.sidebar.text_input("Enter Document URL")
    doc_name = st.sidebar.text_input("Document Name (Optional)")
    upload_mode = st.sidebar.selectbox("Upload Mode", ["fast", "accurate"])

    if st.sidebar.button("Upload Document"):
        if st.session_state.pipeline and doc_url:
            try:
                response = st.session_state.pipeline.upload_document(doc_url, doc_name, upload_mode)
                st.session_state.uploaded_documents.append({"url": doc_url, "name": doc_name or "Document"})
                st.success(f"Document '{doc_name or doc_url}' uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading document: {str(e)}")
        else:
            st.error("Please configure API keys first.")

    st.sidebar.write("Uploaded Documents:")
    for doc in st.session_state.uploaded_documents:
        st.sidebar.write(f"- {doc['name']} ({doc['url']})")

    # Enable the "Switch to Chat" button once everything is ready
    if st.session_state.pipeline and st.session_state.uploaded_documents:
        if st.sidebar.button("Switch to Chat Mode"):
            st.session_state.admin_mode = False
            st.session_state.chat_mode = True

def chat_interface():
    st.markdown("### 🕵️‍♂️ Habla con Enrique tu asistente virtual")
    if not st.session_state.pipeline:
        st.error("The system is not configured yet. Please contact the administrator.")
        return

    # Display chat history
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['content']}")
            else:
                st.markdown(f"**Enrique AI:** {chat['content']}")

                # Use a temporary variable for the input
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""

    st.session_state.current_query = st.text_input(
        "Enter your message", 
        value=st.session_state.current_query, 
        key="chat_query"
    )

    # Input for user query
    #query = st.text_input("Escribe tu mensaje", key="chat_query")

    # Add a state variable to track query submission
    #if "query_submitted" not in st.session_state:
     #   st.session_state.query_submitted = False

    if st.button("Enviar"):
        if st.session_state.current_query.strip():  # Check if input is not empty
            try:
                with st.spinner("Generando respuesta..."):
                    chunks = st.session_state.pipeline.retrieve_chunks(st.session_state.current_query)
                    if not chunks:
                        st.error("No relevant information found.")
                    else:
                        system_prompt = st.session_state.pipeline.create_system_prompt(chunks)
                        response = st.session_state.pipeline.generate_response(
                            system_prompt, st.session_state.current_query, st.session_state.chat_history
                        )
                        # Update chat history
                        st.session_state.chat_history.append({"role": "user", "content": st.session_state.current_query})
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

                        # Clear the input field
                        st.session_state.current_query = ""  # Clear after processing
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
    else:
        # Default interface: Allow admin to log in or switch modes
        password = st.sidebar.text_input("Enter Admin Password", type="password")
        if st.sidebar.button("Switch to Admin Mode"):
            if password == "admin_password":  # Replace with your secure password
                st.session_state.admin_mode = True
            else:
                st.error("Incorrect password.")



if __name__ == "__main__":
    main()
