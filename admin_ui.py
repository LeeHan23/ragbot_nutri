import streamlit as st
import requests
import os
import asyncio
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# This needs to be imported to use the chat functionality
from rag import get_contextual_response

# --- Configuration ---
# The URL of your running FastAPI backend server
API_BASE_URL = "http://localhost:8000/admin"

# --- Helper Functions ---
def upload_file(endpoint: str, file, file_type: str):
    """Sends a file to a specified API endpoint."""
    files = {'file': (file.name, file, file.type)}
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", files=files)
        response.raise_for_status()  # Raise an exception for bad status codes
        st.success(f"{file_type} file '{file.name}' uploaded successfully!")
        st.json(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading file: {e}")
        if e.response:
            st.error(f"Server responded with: {e.response.text}")

# --- Streamlit App ---
st.set_page_config(page_title="Chatbot Admin Panel", layout="wide")
st.title("🤖 Chatbot Admin Control Panel")

st.info("Ensure your main application server is running (`python app.py`) before using this panel.")

# --- File Management Section ---
st.header("Manage Bot Knowledge & Persona")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("📚 Foundational Knowledge")
        st.markdown("Upload `.docx` or `.pdf` documents to the bot's main knowledge base.")
        knowledge_file = st.file_uploader("Upload Document", type=["docx", "pdf"], key="knowledge")
        if st.button("Add to Knowledge Base") and knowledge_file:
            # The endpoint for knowledge base already handles both types
            upload_file("add-to-knowledge-base", knowledge_file, "Knowledge")

with col2:
    with st.container(border=True):
        st.subheader("👤 Global Instructions")
        st.markdown("Upload a `.docx` or `.pdf` file to define the bot's core persona.")
        instructions_file = st.file_uploader("Upload Instructions", type=["docx", "pdf"], key="instructions")
        if st.button("Update Global Instructions") and instructions_file:
            upload_file("upload/instructions/global", instructions_file, "Instructions")

with col3:
    with st.container(border=True):
        st.subheader("💰 Global Promotions")
        st.markdown("Upload a `.docx` or `.pdf` with the latest sales or promotional info.")
        promo_file = st.file_uploader("Upload Promotions", type=["docx", "pdf"], key="promo")
        if st.button("Update Promotions") and promo_file:
            upload_file("upload/promo", promo_file, "Promotions")

st.divider()

# --- Chat Testing Section ---
st.header("Test the Bot (Global Persona)")
st.markdown("Chat with the bot here to test its responses using the foundational knowledge and global instructions.")

# Use a generic user_id for testing global settings
TEST_USER_ID = "admin_test_user"

if "admin_messages" not in st.session_state:
    st.session_state.admin_messages = []

for message in st.session_state.admin_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask the bot a question..."):
    st.session_state.admin_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Eva is thinking..."):
            chat_history = st.session_state.admin_messages
            # Call the RAG function directly
            response_data = asyncio.run(get_contextual_response(prompt, chat_history, TEST_USER_ID))
            
            response_text = response_data.get("answer", "I'm sorry, an error occurred.")
            sources = response_data.get("sources", [])
            
            st.write(response_text)

            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        source_name = os.path.basename(source.metadata.get('source', 'Unknown'))
                        st.info(f"Source: {source_name}, Page: {source.metadata.get('page', 'N/A')}")
                        st.caption(f"> {source.page_content[:250]}...")

    st.session_state.admin_messages.append({
        "role": "assistant", 
        "content": response_text,
    })

