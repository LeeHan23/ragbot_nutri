import streamlit as st
import asyncio
import os
import shutil
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from dotenv import load_dotenv
import streamlit.components.v1 as components

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from rag import get_contextual_response
from knowledge_manager import build_user_database
from langchain_core.messages import HumanMessage, AIMessage

# --- Constants ---
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Personalized AI Chatbot", page_icon="🤖", layout="wide")

# --- User Authentication ---
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    name, authentication_status, username = authenticator.login(location='main')

except FileNotFoundError:
    st.error("Authentication configuration file (`config.yaml`) not found.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during authentication setup: {e}")
    st.stop()


# --- Main Application Logic ---
if authentication_status:
    # --- LOGGED IN ---
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f"Welcome *{name}*")
    
    user_id = username

    # Initialize session state for the logged-in user
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if user_id not in st.session_state.messages:
        st.session_state.messages[user_id] = []
    
    # --- Sidebar for Knowledge Management ---
    with st.sidebar:
        st.divider()
        st.header("Train Your Bot")
        uploaded_files = st.file_uploader(
            "Upload your .docx files to create a custom knowledge base",
            accept_multiple_files=True,
            type=['docx']
        )

        if st.button("Build Custom Knowledge Base"):
            if not uploaded_files:
                st.warning("Please upload at least one .docx document.")
            else:
                with st.spinner("Building your custom knowledge base..."):
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
                
                # --- CORRECTED: Properly extract the answer from the dictionary ---
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
else:
    st.error('Username/password is incorrect')
