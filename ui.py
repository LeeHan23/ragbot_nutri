import streamlit as st
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

# --- Constants ---
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")
BASE_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "vectorstore_base")

# --- GoatCounter Tracking Script ---
goat_counter_script = """
<script data-goatcounter="https://han233.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
"""

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
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    # Render the login form
    name, authentication_status, username = authenticator.login('Login', 'main')

except FileNotFoundError:
    st.error("Authentication configuration file (`config.yaml`) not found.")
    st.stop()


# --- Main Application Logic ---
if authentication_status:
    # --- LOGGED IN ---
    
    # Inject GoatCounter script invisibly
    components.html(goat_counter_script, height=0)

    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f"Welcome *{name}*")
    
    user_id = username # Use the authenticated username as the user_id

    # Initialize session state for the logged-in user
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if user_id not in st.session_state.messages:
        st.session_state.messages[user_id] = []

    # --- Sidebar for Knowledge Management ---
    with st.sidebar:
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
                with st.spinner("Building your custom knowledge base... This will create a separate DB with only your documents."):
                    build_user_database(user_id, uploaded_files, status_callback=st.write)
                st.success("Training complete! Your custom knowledge base is ready.")
        
        if st.button("Reset to Foundational Knowledge"):
            with st.spinner("Resetting knowledge base..."):
                user_db_path = os.path.join(USER_DB_PATH, user_id)
                if os.path.exists(user_db_path):
                    shutil.rmtree(user_db_path)
                st.session_state.messages[user_id] = [] # Clear chat history on reset
            st.success(f"Custom knowledge for user '{user_id}' has been cleared.")

    # --- Main Chat Interface ---
    st.title("🤖 Personalized AI Chatbot")
    st.caption(f"You are logged in as: {username}")

    # Display chat messages
    for message in st.session_state.messages.get(user_id, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages[user_id].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Eva is thinking..."):
                # Pass the simple list of dictionaries directly
                # CORRECTED: The response is now a simple string
                response_text = get_contextual_response(prompt, st.session_state.messages[user_id], user_id)
                st.write(response_text)
        
        st.session_state.messages[user_id].append({"role": "assistant", "content": response_text})

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
