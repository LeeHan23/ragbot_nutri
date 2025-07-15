import streamlit as st
import asyncio
import os
import shutil
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from rag import get_contextual_response
from langchain_core.messages import HumanMessage, AIMessage
# Import the function from our refactored script
from create_database import build_user_database

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Personalized AI Chatbot", page_icon="🤖", layout="wide")

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None

# --- Sidebar for User Management and Training ---
with st.sidebar:
    st.header("User & Knowledge Management")
    
    user_id_input = st.text_input("Enter your User ID to start chatting:", key="user_id_input")
    if st.button("Start/Switch User Session"):
        if user_id_input:
            st.session_state.current_user_id = user_id_input
            if user_id_input not in st.session_state.messages:
                st.session_state.messages[user_id_input] = []
                st.session_state.user_data[user_id_input] = {
                    "chat_history": [], "visit_count": 1, "intent_summary": "User is starting a new conversation."
                }
            st.success(f"Session started for user: {user_id_input}")
        else:
            st.warning("Please enter a User ID.")

    st.divider()

    st.header("Train Your Bot")
    uploaded_files = st.file_uploader(
        "Upload your .docx files to create a new knowledge base",
        accept_multiple_files=True,
        type=['docx']
    )

    if st.button("Build New Knowledge Base"):
        user_id = st.session_state.current_user_id
        if not user_id:
            st.error("Please start a user session first.")
        elif not uploaded_files:
            st.warning("Please upload at least one .docx document.")
        else:
            with st.spinner("Building new knowledge base... This will replace any existing custom knowledge and may take several minutes."):
                # Call the updated function to build a new DB from base + uploaded docs
                build_user_database(user_id, uploaded_files, status_callback=st.write)
                
            st.success("Training complete! Your bot is ready with the new knowledge.")
    
    if st.button("Reset to Foundational Knowledge"):
        user_id = st.session_state.current_user_id
        if not user_id:
            st.error("Please start a user session first.")
        else:
            with st.spinner("Resetting knowledge base..."):
                user_db_path = os.path.join(USER_DB_PATH, user_id)
                if os.path.exists(user_db_path):
                    shutil.rmtree(user_db_path)
                
                st.session_state.messages[user_id] = []
                st.session_state.user_data[user_id]["chat_history"] = []

            st.success(f"Custom knowledge for user '{user_id}' has been cleared. The bot will now use the foundational knowledge.")

# --- Main Chat Interface ---
st.title("🤖 Personalized AI Chatbot")
st.caption("Your personal AI assistant. Upload documents in the sidebar to train it.")

if st.session_state.current_user_id:
    user_id = st.session_state.current_user_id
    
    # The logic in rag.py now handles the fallback to the base DB automatically
    for message in st.session_state.messages.get(user_id, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages[user_id].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Eva is thinking..."):
                current_user_data = st.session_state.user_data[user_id]
                response = asyncio.run(get_contextual_response(prompt, current_user_data, user_id))
                st.write(response)

        st.session_state.messages[user_id].append({"role": "assistant", "content": response})
        current_user_data["chat_history"].extend([HumanMessage(content=prompt), AIMessage(content=response)])
else:
    st.info("Please enter a User ID in the sidebar to begin.")
