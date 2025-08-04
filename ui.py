import streamlit as st
import asyncio
import os
import shutil
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from rag import get_contextual_response
from knowledge_manager import build_user_database
from database import add_user, check_login, verify_user # <-- Import custom DB functions
from langchain_core.messages import HumanMessage, AIMessage

# --- Constants ---
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Personalized AI Chatbot", page_icon="🤖", layout="wide")

# --- Initialize Session State ---
# This ensures that the keys exist before we access them.
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "name" not in st.session_state:
    st.session_state["name"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "messages" not in st.session_state:
    st.session_state.messages = {}

# --- Authentication and Main App Logic ---

# If user is not authenticated, show the login/signup page
if not st.session_state["authentication_status"]:
    st.title("Welcome to the Personalized AI Nutrition Chatbot")
    
    choice = st.sidebar.selectbox("Login or Sign Up", ["Login", "Sign Up"])
    
    if choice == "Sign Up":
        st.subheader("Create a New Account")
        with st.form("Sign Up Form"):
            new_name = st.text_input("Full Name")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if not all([new_name, new_username, new_password]):
                    st.error("Please fill out all fields.")
                else:
                    verification_key = add_user(new_username, new_name, new_password)
                    if verification_key:
                        st.success("Account created successfully!")
                        st.info(f"Please copy your one-time verification key and use it on your first login: **{verification_key}**")
                    else:
                        st.error("Username already exists. Please choose a different one.")

    else: # Login
        st.subheader("Login to Your Account")
        with st.form("Login Form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                success, user_name, is_verified = check_login(username, password)
                
                if success:
                    if is_verified:
                        # Standard login
                        st.session_state["authentication_status"] = True
                        st.session_state["name"] = user_name
                        st.session_state["username"] = username
                        st.rerun()
                    else:
                        # First-time login, requires verification
                        st.info("This is your first login. Please enter your verification key.")
                        verification_key = st.text_input("One-Time Verification Key")
                        if st.button("Verify and Login"):
                            if verify_user(username, verification_key):
                                st.session_state["authentication_status"] = True
                                st.session_state["name"] = user_name
                                st.session_state["username"] = username
                                st.rerun()
                            else:
                                st.error("Verification key is incorrect.")
                else:
                    st.error("Username or password is not correct.")

# If user IS authenticated, run the main chatbot app
else:
    # --- LOGGED IN ---
    name = st.session_state["name"]
    username = st.session_state["username"]
    user_id = username

    def logout():
        st.session_state["authentication_status"] = None
        st.session_state["name"] = None
        st.session_state["username"] = None
        if user_id in st.session_state.messages:
             del st.session_state.messages[user_id] # Optional: clear messages on logout
        st.rerun()

    st.sidebar.title(f"Welcome *{name}*")
    st.sidebar.button('Logout', on_click=logout)
    
    # Initialize session state for the logged-in user's messages
    if user_id not in st.session_state.messages:
        st.session_state.messages[user_id] = []
    
    # --- Sidebar for Knowledge Management ---
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
                with st.spinner("Building your custom knowledge base... This may take a moment."):
                    build_user_database(user_id, uploaded_files, status_callback=st.write)
                st.success("Training complete! Your custom knowledge base is ready.")
        
        if st.button("Reset to Foundational Knowledge"):
            with st.spinner("Resetting knowledge base..."):
                user_db_path = os.path.join(USER_DB_PATH, user_id)
                if os.path.exists(user_db_path):
                    shutil.rmtree(user_db_path)
                st.session_state.messages[user_id] = []
            st.success(f"Custom knowledge for user '{user_id}' has been cleared.")

    # --- Main Chat Interface ---
    st.title("🤖 Personalized AI Chatbot")
    st.caption(f"You are chatting as: {username}")

    for message in st.session_state.messages.get(user_id, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        source_name = os.path.basename(source.metadata.get('source', 'Unknown'))
                        st.info(f"Source: {source_name}, Page: {source.metadata.get('page', 'N/A')}")
                        st.caption(f"> {source.page_content[:250]}...")

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages[user_id].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Eva is thinking..."):
                chat_history = st.session_state.messages[user_id]
                response_data = asyncio.run(get_contextual_response(prompt, chat_history, user_id))
                
                response_text = response_data.get("answer", "I'm sorry, an error occurred.")
                sources = response_data.get("sources", [])
                
                st.write(response_text)

                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            source_name = os.path.basename(source.metadata.get('source', 'Unknown'))
                            st.info(f"Source: {source_name}, Page: {source.metadata.get('page', 'N/A')}")
                            st.caption(f"> {source.page_content[:250]}...")

        st.session_state.messages[user_id].append({
            "role": "assistant", 
            "content": response_text,
            "sources": sources
        })