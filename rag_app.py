
import streamlit as st
import requests
from anthropic import Anthropic
from urllib.parse import urlparse
from typing import List, Dict, Optional
import json
import os
import bcrypt
import logging

#logging.basicConfig(level=logging.DEBUG)


def ensure_user_data_file():
    """Ensure the user data file exists and is valid JSON."""
    if not os.path.exists("user_data.json"):
        with open("user_data.json", "w") as file:
            json.dump({}, file)


def load_user_data():
    """Load user data from a JSON file."""
    try:
        with open("user_data.json", "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        # If the file is invalid, reset it
        with open("user_data.json", "w") as file:
            json.dump({}, file)  # Reset to an empty JSON object
        return {}

def save_user_data(user_data):
    """Save user data to a JSON file."""
    try:
        with open("user_data.json", "w") as file:
            json.dump(user_data, file, indent=4)
        #logging.debug(f"User data saved successfully: {user_data}")
    except Exception as e:
        logging.error(f"Error saving user data: {str(e)}")
        raise Exception(f"Failed to save user data: {str(e)}")

def check_login():
    """Handle user login and conversation persistence."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if not st.session_state.logged_in:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_button"):
            user_data = load_user_data()

            if username in user_data and bcrypt.checkpw(password.encode('utf-8'), user_data[username]["password"].encode('utf-8')):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.conversations = user_data[username].get("conversations", {})
                st.success("Login successful!")
            else:
                st.error("Invalid username or password.")
        st.stop()


def register_user():
    """Handle user registration."""
    ensure_user_data_file()
    user_data = load_user_data()

    with st.form(key="register_form"):
        username = st.text_input("New Username", key="register_username")
        password = st.text_input("New Password", type="password", key="register_password")
        submit = st.form_submit_button("Register")

    if submit:
        if not username.strip() or not password.strip():
            st.error("Username and password cannot be empty.")
            return

        if username in user_data:
            st.error("Username already exists.")
            return

        try:
            hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            user_data[username] = {"password": hashed_password, "conversations": {}}
            save_user_data(user_data)
            st.success(f"User '{username}' registered successfully. Please log in.")
        except Exception as e:
            st.error(f"An error occurred during registration: {str(e)}")


def save_conversation(username, conversations):
    """Save the user's conversations to the data file."""
    user_data = load_user_data()
    if username in user_data:
        user_data[username]["conversations"] = conversations
        save_user_data(user_data)



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

    def generate_response(self, system_prompt: str, query: str, conversation_history: list = None) -> str:
        if conversation_history is None:
            conversation_history = []

        messages = conversation_history + [{"role": "user", "content": query}]

        try:
            # Make the API request
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            )

            # Log the complete response object for debugging
            #logging.debug(f"Complete API response object: {vars(response)}")

            # Check if response has content attribute
            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'completion'):
                return response.completion
            else:
                # Log the response structure
                logging.error(f"Unexpected response structure. Available attributes: {dir(response)}")
                raise ValueError(f"Unexpected response structure. Response type: {type(response)}")

        except Exception as e:
            logging.error(f"Error in generate_response: {type(e).__name__}: {str(e)}")
            if hasattr(e, '__dict__'):
                logging.error(f"Error details: {vars(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")
        
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
        try:
            ragie_key = st.secrets["RAGIE_API_KEY"]
            anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
            st.session_state.pipeline = RAGPipeline(ragie_key, anthropic_key)
        except KeyError as e:
            raise Exception(f"Missing API key in secrets: {str(e)}")

    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}

    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None

def chat_interface():
    st.markdown("### Chat with Enrique AI")

    # Get Current Conversation History
    current_history = st.session_state.conversations.get(st.session_state.current_conversation, [])

    # Display Conversation History
    for message in current_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Enrique AI:** {message['content']}")

    # Input and Send Message
    query = st.text_input("Your message", key="message_input")
    if st.button("Send", key="send_message_button"):
        if query.strip():
            # Add User Message to Conversation
            current_history.append({"role": "user", "content": query})

            with st.spinner("Generating response..."):
                try:
                    # Retrieve Chunks (Mocked or Actual)
                    chunks = ["example chunk"]  # Replace with actual retrieval logic
                    system_prompt = st.session_state.pipeline.create_system_prompt(chunks)

                    # Generate AI Response
                    response = st.session_state.pipeline.generate_response(system_prompt, query, current_history)

                    # Add AI Response to Conversation
                    current_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

            # Save Updated Conversation
            st.session_state.conversations[st.session_state.current_conversation] = current_history
            save_conversation(st.session_state.username, st.session_state.conversations)

def main():
    st.set_page_config(page_title="Client Chat System", layout="wide")

    ensure_user_data_file()

    # Login and Register Options
    option = st.sidebar.selectbox("Choose an option", ["Login", "Register"])

    if option == "Register":
        register_user()
    elif option == "Login":
        check_login()

    # Show chat interface if logged in
    if st.session_state.get("logged_in", False):
        st.sidebar.write(f"Welcome, {st.session_state.username}")
        initialize_session_state()

        # List and Select Conversations
        st.sidebar.markdown("## Conversations")
        if st.session_state.conversations:
            for convo in st.session_state.conversations.keys():
                if st.sidebar.button(convo, key=f"select_convo_{convo}"):
                    st.session_state.current_conversation = convo

        # Option to Create a New Conversation
        with st.sidebar.expander("Create a New Conversation"):
            new_convo_name = st.text_input("Conversation Name", key="new_convo_name")
            if st.button("Create Conversation", key="create_new_convo"):
                if new_convo_name.strip():
                    st.session_state.conversations[new_convo_name] = []
                    st.session_state.current_conversation = new_convo_name
                    save_conversation(st.session_state.username, st.session_state.conversations)
                    st.success(f"Conversation '{new_convo_name}' created!")
                else:
                    st.error("Conversation name cannot be empty.")

        # Display Chat Interface
        if st.session_state.current_conversation:
            chat_interface()
        else:
            st.info("Please select or create a conversation to start chatting.")

if __name__ == "__main__":
    main()