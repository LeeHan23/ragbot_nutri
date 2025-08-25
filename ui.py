import streamlit as st
import asyncio
import os
import shutil
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from rag import get_contextual_response
from knowledge_manager import build_user_database # Import from new consolidated manager
from database import add_user, check_login, verify_user
from langchain_core.messages import HumanMessage, AIMessage

# --- Constants ---
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Personalized AI Chatbot", page_icon="🤖", layout="wide")

# --- Initialize Session State ---
# (Session state initialization remains the same)
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "name" not in st.session_state:
    st.session_state["name"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "requires_verification" not in st.session_state:
    st.session_state["requires_verification"] = False
if "username_for_verification" not in st.session_state:
    st.session_state["username_for_verification"] = ""

# --- Authentication and Main App Logic ---
if not st.session_state["authentication_status"]:
    # (Login/Signup logic remains the same)
    st.title("Welcome to the Personalized AI Nutrition Chatbot")
    choice = st.sidebar.selectbox("Login or Sign Up", ["Login", "Sign Up"])
    if choice == "Sign Up":
        st.subheader("Create a New Account")
        st.session_state["requires_verification"] = False 
        with st.form("Sign Up Form"):
            # ... (signup form code)
            pass # Placeholder for brevity
    else: # Login
        st.subheader("Login to Your Account")
        if st.session_state.get("requires_verification"):
            # ... (verification code)
            pass # Placeholder for brevity
        else:
            with st.form("Login Form"):
                # ... (login form code)
                pass # Placeholder for brevity

# If user IS authenticated, run the main chatbot app
else:
    name = st.session_state["name"]
    username = st.session_state["username"]
    user_id = username

    def logout():
        st.session_state["authentication_status"] = None
        st.session_state["name"] = None
        st.session_state["username"] = None
        st.session_state["requires_verification"] = False
        st.session_state["username_for_verification"] = ""
        if user_id in st.session_state.messages:
             del st.session_state.messages[user_id]
        st.rerun()

    st.sidebar.title(f"Welcome *{name}*")
    st.sidebar.button('Logout', on_click=logout)
    
    # --- NEW: Add Clear Chat History Button ---
    if st.sidebar.button("Clear Chat History"):
        if user_id in st.session_state.messages:
            st.session_state.messages[user_id] = []
        st.rerun()
    
    if user_id not in st.session_state.messages:
        st.session_state.messages[user_id] = []
    
    with st.sidebar:
        st.divider()
        st.header("Train Your Bot")
        uploaded_files = st.file_uploader(
            "Upload your .docx persona files",
            accept_multiple_files=True,
            type=['docx']
        )

        if st.button("Build Custom Knowledge Base"):
            if not uploaded_files:
                st.warning("Please upload at least one .docx document.")
            else:
                with st.spinner("Building your custom knowledge base..."):
                    build_user_database(user_id, uploaded_files, status_callback=st.write)
                st.success("Training complete!")
        
        if st.button("Reset to Foundational Knowledge"):
            with st.spinner("Resetting knowledge base..."):
                user_db_path = os.path.join(USER_DB_PATH, user_id)
                if os.path.exists(user_db_path):
                    shutil.rmtree(user_db_path)
                st.session_state.messages[user_id] = []
            st.success("Knowledge base reset.")

    # --- Main Chat Interface ---
    # (The chat interface logic remains the same)
    st.title("🤖 Personalized AI Chatbot")
    st.caption(f"You are chatting as: {username}")
    # ... (rest of the chat UI code)
