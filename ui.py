import streamlit as st
import asyncio
import os
import shutil
import sqlite3
import bcrypt
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from rag import get_contextual_response
from knowledge_manager import build_user_database

# --- Constants ---
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")
DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "users.db")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Personalized AI Chatbot", page_icon="🤖", layout="wide")

# --- Database Authentication Functions ---
def check_login(username, password):
    """Checks user credentials against the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password, name FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        hashed_password = user[0].encode('utf-8')
        password_bytes = password.encode('utf-8')
        if bcrypt.checkpw(password_bytes, hashed_password):
            return True, user[1] # Return success and the user's full name
    return False, None

# --- Main App Logic ---
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

if st.session_state['authentication_status']:
    # --- LOGGED IN STATE ---
    username = st.session_state['username']
    name = st.session_state['name']
    
    st.sidebar.success(f"Welcome, {name}!")
    if st.sidebar.button('Logout'):
        st.session_state['authentication_status'] = None
        st.session_state['username'] = None
        st.session_state['name'] = None
        st.rerun()

    # (The rest of your application UI goes here, same as before)
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
                with st.spinner("Building new knowledge base..."):
                    build_user_database(username, uploaded_files, status_callback=st.write)
                st.success("Training complete!")
        
        if st.button("Reset to Foundational Knowledge"):
            with st.spinner("Resetting knowledge base..."):
                user_db_path = os.path.join(USER_DB_PATH, username)
                if os.path.exists(user_db_path):
                    shutil.rmtree(user_db_path)
                # Also clear chat history for this user
                if 'messages' in st.session_state and username in st.session_state.messages:
                    st.session_state.messages[username] = []
            st.success("Custom knowledge reset.")

    # --- Main Chat Interface ---
    st.title("🤖 Personalized AI Chatbot")
    st.caption(f"You are chatting as: {username}")
    
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

else:
    # --- LOGIN FORM ---
    st.header("Login to Your Chatbot")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            is_correct, name = check_login(username, password)
            if is_correct:
                st.session_state['authentication_status'] = True
                st.session_state['username'] = username
                st.session_state['name'] = name
                st.rerun()
            else:
                st.error("Username/password is incorrect")
