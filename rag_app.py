import streamlit as st
import requests
from anthropic import Anthropic
from urllib.parse import urlparse
from typing import List, Dict, Optional
import json
import os
import bcrypt
import logging
logging.basicConfig(level=logging.DEBUG)

def save_user_data(user_data):
    """Save user data to a JSON file."""
    logging.debug(f"Saving user data: {user_data}")
    with open("user_data.json", "w") as file:
        json.dump(user_data, file, indent=4)

#with open("user_data.json", "w") as test_file:
#    test_file.write("test")


def ensure_user_data_file():
    """Ensure the user data file exists and is valid."""
    if not os.path.exists("user_data.json"):
        with open("user_data.json", "w") as file:
            json.dump({}, file)

def load_user_data():
    """Load user data from a JSON file."""
    if os.path.exists("user_data.json"):
        with open("user_data.json", "r") as file:
            return json.load(file)
    return {}

def check_login():
    """Handle user login and conversation persistence."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if not st.session_state.logged_in:
        st.text_input("Username", key="login_username")
        st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_button"):
            username = st.session_state.login_username
            password = st.session_state.login_password.encode('utf-8')
            user_data = load_user_data()

            if username in user_data and bcrypt.checkpw(password, user_data[username]["password"].encode('utf-8')):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.conversations = user_data[username]["conversations"]
                st.success("Login successful!")
            else:
                st.error("Invalid username or password.")

        st.stop()

def register_user():
    """Handle user registration."""
    if "registration_success" not in st.session_state:
        st.session_state.registration_success = False

    # Registration form
    with st.form(key="register_form", clear_on_submit=False):
        username = st.text_input("New Username", key="register_username")
        password = st.text_input("New Password", type="password", key="register_password")
        register_submit = st.form_submit_button("Register")

    # Handle registration logic
    if register_submit:
        if not username.strip() or not password.strip():
            st.error("Username and password cannot be empty.")
            return

        user_data = load_user_data()

        if username in user_data:
            st.error("Username already exists.")
        else:
            try:
                # Securely hash the password
                hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                # Add the new user to the user data
                user_data[username] = {"password": hashed_password, "conversations": {}}
                save_user_data(user_data)

                # Indicate successful registration
                st.session_state.registration_success = True
                st.success("User registered successfully. Please log in.")
            except Exception as e:
                st.error(f"An error occurred while registering the user: {str(e)}")
                return

    # Reset success message if page re-renders
    if st.session_state.registration_success:
        st.success("User registered successfully. Please log in.")


       # st.stop()

def save_conversation(username, conversations):
    """Save the user's conversations to the data file."""
    user_data = load_user_data()
    if username not in user_data:
        user_data[username] = {"password": "", "conversations": {}}
    user_data[username]["conversations"] = conversations
    save_user_data(user_data)

class RAGPipeline:
    def __init__(self, ragie_api_key: str, anthropic_api_key: str):
        self.ragie_api_key = ragie_api_key
        self.anthropic_api_key = anthropic_api_key
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)

        self.RAGIE_UPLOAD_URL = "https://api.ragie.ai/documents/url"
        self.RAGIE_RETRIEVAL_URL = "https://api.ragie.ai/retrievals"

    def upload_document(self, content: str, name: Optional[str] = None, mode: str = "fast") -> Dict:
        payload = {
            "mode": mode,
            "name": name or "Generated Document",
            "content": content
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

    def generate_response(self, system_prompt: str, query: str, conversation_history: list = None) -> str:
        if conversation_history is None:
            conversation_history = []
        
        messages = conversation_history + [{"role": "user", "content": query}]
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            )
            return response.get("completion", "No response generated").strip()
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")

def initialize_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        ragie_key = st.secrets["RAGIE_API_KEY"]
        anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
        st.session_state.pipeline = RAGPipeline(ragie_key, anthropic_key)

    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}

    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None

def chat_interface():
    st.markdown("### 🕵️‍♂️ Habla con Enrique AI")

    if not st.session_state.current_conversation:
        st.info("Por favor selecciona o crea una nueva conversación.")
        new_convo_name = st.text_input("Nombre de la nueva conversación")
        if st.button("Crear conversación", key="create_convo_button"):
            if new_convo_name.strip():
                st.session_state.conversations[new_convo_name] = []
                st.session_state.current_conversation = new_convo_name
                save_conversation(st.session_state.username, st.session_state.conversations)
            else:
                st.error("Por favor introduce un nombre válido para la conversación.")
        return

    current_history = st.session_state.conversations[st.session_state.current_conversation]

    for message in current_history:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**AI:** {content}")

    query = st.text_input("Escribe tu mensaje")
    if st.button("Enviar", key="send_message_button"):
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
                save_conversation(st.session_state.username, st.session_state.conversations)

def main():
    st.set_page_config(page_title="Client Chat System",
                       page_icon="https://essent-ia.com/wp-content/uploads/2024/11/cropped-cropped-Picture1.png",
                       layout="centered")
    
    ensure_user_data_file()  # Ensure the file exists

    if st.sidebar.button("Register", key="sidebar_register_button"):
        register_user()

    check_login()
    initialize_session_state()

    st.sidebar.markdown("## Conversaciones")
    if st.session_state.conversations:
        for convo in st.session_state.conversations.keys():
            if st.sidebar.button(convo, key=f"select_convo_{convo}"):
                st.session_state.current_conversation = convo

    chat_interface()

if __name__ == "__main__":
    main()
