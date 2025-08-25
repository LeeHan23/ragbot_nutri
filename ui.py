import streamlit as st
import asyncio
import os
import shutil
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from rag import get_contextual_response
from knowledge_manager import build_user_database
from database import add_user, check_login, verify_user
from langchain_core.messages import HumanMessage, AIMessage

# --- Constants ---
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Personalized AI Chatbot", page_icon="🤖", layout="wide")

# --- Initialize Session State ---
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
    st.title("Welcome to the Personalized AI Nutrition Chatbot")
    
    choice = st.sidebar.selectbox("Login or Sign Up", ["Login", "Sign Up"])
    
    if choice == "Sign Up":
        st.subheader("Create a New Account")
        st.session_state["requires_verification"] = False 
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
        
        if st.session_state.get("requires_verification"):
            st.info("This is your first login. Please enter your verification key to continue.")
            
            verification_key = st.text_input("One-Time Verification Key")
            
            if st.button("Verify and Login"):
                username_to_verify = st.session_state["username_for_verification"]
                success, user_name, is_verified = check_login(username_to_verify, "")
                
                if verify_user(username_to_verify, verification_key):
                    st.session_state["authentication_status"] = True
                    st.session_state["name"] = user_name
                    st.session_state["username"] = username_to_verify
                    st.session_state["requires_verification"] = False
                    st.session_state["username_for_verification"] = ""
                    st.rerun()
                else:
                    st.error("Verification key is incorrect.")
        
        else:
            with st.form("Login Form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")

                if submitted:
                    success, user_name, is_verified = check_login(username, password)
                    
                    if success:
                        if is_verified:
                            st.session_state["authentication_status"] = True
                            st.session_state["name"] = user_name
                            st.session_state["username"] = username
                            st.rerun()
                        else:
                            st.session_state["requires_verification"] = True
                            st.session_state["username_for_verification"] = username
                            st.rerun()
                    else:
                        st.error("Username or password is not correct.")

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

    st.title("🤖 Personalized AI Chatbot")
    st.caption(f"You are chatting as: {username}")

    for message in st.session_state.messages.get(user_id, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                knowledge_source = message.get("knowledge_source", "Unknown")
                with st.expander("View Sources"):
                    st.caption(f"Answer generated using: **{knowledge_source} Knowledge Base**")
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
                knowledge_source = response_data.get("knowledge_source", "Unknown")
                
                st.write(response_text)

                if sources:
                    with st.expander("View Sources"):
                        st.caption(f"Answer generated using: **{knowledge_source} Knowledge Base**")
                        for source in sources:
                            source_name = os.path.basename(source.metadata.get('source', 'Unknown'))
                            st.info(f"Source: {source_name}, Page: {source.metadata.get('page', 'N/A')}")
                            st.caption(f"> {source.page_content[:250]}...")

        st.session_state.messages[user_id].append({
            "role": "assistant", 
            "content": response_text,
            "sources": sources,
            "knowledge_source": knowledge_source
        })
