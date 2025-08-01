import streamlit as st
import asyncio
import os
import shutil
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from rag import get_contextual_response
from knowledge_manager import build_user_database
# Import the new database functions
from database import add_user, check_login, verify_user, create_user_table

# --- INITIALIZE DATABASE ---
# This ensures the users.db file and the users table are created on startup.
create_user_table()

# --- Constants ---
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Personalized AI Chatbot", page_icon="🤖", layout="wide")

# --- Main App Logic ---
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'

# --- Page Navigation ---
if st.session_state['authentication_status']:
    # --- LOGGED IN STATE ---
    username = st.session_state['username']
    name = st.session_state['name']
    
    st.sidebar.success(f"Welcome, {name}!")
    if st.sidebar.button('Logout'):
        st.session_state['authentication_status'] = None
        st.session_state['username'] = None
        st.session_state['name'] = None
        st.session_state['page'] = 'login'
        st.rerun()

    # (The rest of your application UI goes here)
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
                if 'messages' in st.session_state and username in st.session_state.messages:
                    st.session_state.messages[username] = []
            st.success("Custom knowledge reset.")

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
    # --- LOGIN / SIGN UP / VERIFY FORMS ---
    if st.session_state['page'] == 'login':
        st.header("Login to Your Chatbot")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                is_correct, name, is_verified = check_login(username, password)
                if is_correct:
                    if not is_verified:
                        st.session_state['page'] = 'verify'
                        st.session_state['temp_username'] = username
                        st.rerun()
                    else:
                        st.session_state['authentication_status'] = True
                        st.session_state['username'] = username
                        st.session_state['name'] = name
                        st.rerun()
                else:
                    st.error("Username/password is incorrect")
        
        if st.button("Don't have an account? Sign Up"):
            st.session_state['page'] = 'signup'
            st.rerun()

    elif st.session_state['page'] == 'signup':
        st.header("Create a New Account")
        with st.form("signup_form"):
            name = st.text_input("Full Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if not name or not username or not password:
                    st.warning("Please fill out all fields.")
                else:
                    verification_key = add_user(username, name, password)
                    if verification_key:
                        st.success("Account created successfully!")
                        st.info(f"Your one-time verification key is: {verification_key}")
                        st.warning("Please copy this key. You will need it for your first login.")
                        st.session_state['page'] = 'login'
                        # No rerun here, let the user see the key
                    else:
                        st.error("That username is already taken. Please choose another one.")
        
        if st.button("Back to Login"):
            st.session_state['page'] = 'login'
            st.rerun()

    elif st.session_state['page'] == 'verify':
        st.header("Account Verification")
        st.info(f"Please enter the one-time verification key for user '{st.session_state['temp_username']}'.")
        with st.form("verify_form"):
            key = st.text_input("Verification Key")
            submitted = st.form_submit_button("Verify")

            if submitted:
                if verify_user(st.session_state['temp_username'], key):
                    st.success("Account verified successfully! Please log in.")
                    st.session_state['page'] = 'login'
                    st.session_state.pop('temp_username', None)
                    st.rerun()
                else:
                    st.error("The verification key is incorrect.")
