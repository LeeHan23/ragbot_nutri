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
from knowledge_manager import add_documents_to_user_db

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "users")
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
    # --- MODIFIED: Restrict uploads to .docx files only ---
    uploaded_files = st.file_uploader(
        "Upload your .docx files here to add to the bot's knowledge",
        accept_multiple_files=True,
        type=['docx']
    )

    # --- MODIFIED: Changed button text and logic ---
    if st.button("Add Documents to Knowledge Base"):
        user_id = st.session_state.current_user_id
        if not user_id:
            st.error("Please start a user session first.")
        elif not uploaded_files:
            st.warning("Please upload at least one .docx document.")
        else:
            with st.spinner("Processing and adding new knowledge..."):
                user_knowledge_path = os.path.join(USER_KNOWLEDGE_DIR, user_id, "knowledge")
                os.makedirs(user_knowledge_path, exist_ok=True)

                saved_file_paths = []
                for file in uploaded_files:
                    file_path = os.path.join(user_knowledge_path, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    saved_file_paths.append(file_path)
                    st.write(f"Saved file: {file.name}")
                
                # Call the updated function to add documents to the user's DB
                add_documents_to_user_db(user_id, saved_file_paths, status_callback=st.write)
                
            st.success("Training complete! Your bot has learned from the new documents.")
    
    if st.button("Clear Custom Knowledge"):
        user_id = st.session_state.current_user_id
        if not user_id:
            st.error("Please start a user session first.")
        else:
            with st.spinner("Clearing custom knowledge base..."):
                user_db_path = os.path.join(USER_DB_PATH, user_id)
                if os.path.exists(user_db_path):
                    shutil.rmtree(user_db_path)
                
                # Also reset chat history for a fresh start with the base model
                st.session_state.messages[user_id] = []
                st.session_state.user_data[user_id]["chat_history"] = []

            st.success(f"Custom knowledge for user '{user_id}' has been cleared. The bot will now use the foundational knowledge.")

# --- Main Chat Interface ---
st.title("🤖 Personalized AI Chatbot")
st.caption("Your personal AI assistant. Add your own documents in the sidebar to customize its knowledge.")

if st.session_state.current_user_id:
    user_id = st.session_state.current_user_id
    
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
