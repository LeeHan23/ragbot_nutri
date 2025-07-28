import streamlit as st
import asyncio
import os
import shutil
from dotenv import load_dotenv
import streamlit_authenticator as stauth
import yaml # We still need this for the authenticator, but won't load the file

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from rag import get_contextual_response
from knowledge_manager import build_user_database, get_prompts

# --- Constants ---
# Path for the persistent disk on Render
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Personalized AI Chatbot", page_icon="🤖", layout="wide")

# --- HARD-CODED USER CONFIGURATION ---
# This bypasses the config.yaml file to solve the parsing error.
# Use your generate_keys.py script to get the hashed passwords.
config = {
    'credentials': {
        'usernames': {
            'jsmith': {
                'email': 'jsmith@gmail.com',
                'name': 'John Smith',
                'password': '$2b$12$.rGI8rG9..3EaNye8VXIIuJS4txbPDF9eQzKgz23KVsQjzjddv72.' # REPLACE with your first hashed password
            },
            'rdoe': {
                'email': 'rdoe@gmail.com',
                'name': 'Rebecca Doe',
                'password': '$2b$12$4.fcJJWQGIZYFHKLksBtTOSYB4PnneZmCEpLhzyICWjYREv4jIheK' # REPLACE with your second hashed password
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'a_very_secret_random_key_12345', # Replace with your own random string
        'name': 'nutrition_bot_cookie'
    }
}

# --- Initialize the Authenticator ---
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- Render the Login Form ---
name, authentication_status, username = authenticator.login(location='main')

# --- Main Application Logic ---
if authentication_status:
    # --- Successful Login ---
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f"Welcome, {name}!")

    # GoatCounter Tracking Script
    st.components.v1.html("""
        <script data-goatcounter="https://han233.goatcounter.com/count"
                async src="//gc.zgo.at/count.js"></script>
    """, height=0)

    # --- Sidebar for Knowledge Management ---
    with st.sidebar:
        st.divider()
        st.header("Train Your Bot")
        uploaded_files = st.file_uploader(
            "Upload .docx files to create a custom knowledge base:",
            accept_multiple_files=True,
            type=['docx']
        )

        if st.button("Build Custom Knowledge Base"):
            if not uploaded_files:
                st.warning("Please upload at least one .docx document.")
            else:
                with st.spinner("Building new knowledge base... This will replace any existing custom knowledge and may take a moment."):
                    build_user_database(username, uploaded_files, status_callback=st.write)
                st.success("Training complete! Your bot is now using your custom knowledge.")
        
        if st.button("Reset to Foundational Knowledge"):
            with st.spinner("Resetting knowledge base..."):
                user_db_path = os.path.join(USER_DB_PATH, username)
                if os.path.exists(user_db_path):
                    shutil.rmtree(user_db_path)
            st.success(f"Custom knowledge for user '{username}' has been cleared. The bot will now use the foundational knowledge.")

    # --- Main Chat Interface ---
    st.title("🤖 Personalized AI Chatbot")
    st.caption(f"You are chatting as: {username}")
    
    # Initialize chat history for the logged-in user
    if 'messages' not in st.session_state:
        st.session_state['messages'] = {}
    if username not in st.session_state['messages']:
        st.session_state['messages'][username] = []
    
    for message in st.session_state.messages[username]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages[username].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Eva is thinking..."):
                chat_history = st.session_state.messages[username]
                response = asyncio.run(get_contextual_response(prompt, chat_history, username))
                st.write(response)

        st.session_state.messages[username].append({"role": "assistant", "content": response})

elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.info('Please enter your username and password')
