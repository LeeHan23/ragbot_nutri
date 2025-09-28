import streamlit as st
import requests
import os
import asyncio
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from rag import get_contextual_response
from agent_tools import get_all_customer_reports
from database import add_user, check_login, verify_user, get_access_logs

API_BASE_URL = "http://localhost:8000/admin"

# --- UPDATED: Helper function now sends metadata ---
def upload_file(endpoint: str, file, file_type: str, metadata: dict):
    files = {'file': (file.name, file, file.type)}
    try:
        # Send metadata as part of the request payload
        response = requests.post(f"{API_BASE_URL}/{endpoint}", files=files, data=metadata)
        response.raise_for_status()
        st.success(f"{file_type} file '{file.name}' uploaded successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading file: {e}")
        if e.response: st.error(f"Server responded with: {e.response.text}")

st.set_page_config(page_title="Chatbot Admin Panel", layout="wide")

# (Login and Sign-up logic remains the same)
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.show_verification = False
    st.session_state.verification_key = ""

st.sidebar.title("Admin Access")
if not st.session_state.logged_in:
    login_mode = st.sidebar.selectbox("Choose Action", ["Login", "Sign Up"])
    if login_mode == "Login":
        with st.sidebar.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                success, name, is_verified = check_login(username, password)
                if success:
                    if is_verified:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.show_verification = False
                        st.rerun()
                    else:
                        st.session_state.show_verification = True
                        st.session_state.username = username
                        st.warning("Account not verified.")
                else:
                    st.error("Invalid username or password.")
        if st.session_state.show_verification:
            with st.sidebar.form("verification_form"):
                st.write(f"Verifying account for: **{st.session_state.username}**")
                key = st.text_input("Verification Key")
                verify_submitted = st.form_submit_button("Verify")
                if verify_submitted:
                    if verify_user(st.session_state.username, key):
                        st.success("Verification successful! You are now logged in.")
                        st.session_state.logged_in = True
                        st.session_state.show_verification = False
                        st.rerun()
                    else:
                        st.error("Invalid verification key.")
    elif login_mode == "Sign Up":
        with st.sidebar.form("signup_form"):
            username = st.text_input("Choose a Username")
            name = st.text_input("Your Name")
            password = st.text_input("Choose a Password", type="password")
            signup_submitted = st.form_submit_button("Sign Up")
            if signup_submitted:
                if username and name and password:
                    key = add_user(username, name, password)
                    if key:
                        st.success("Sign-up successful!")
                        st.info(f"Your verification key is: {key}")
                    else:
                        st.error("Username already exists.")
                else:
                    st.warning("Please fill out all fields.")

if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    st.title("🤖 Chatbot Admin Control Panel")
    st.info("Ensure your main application server is running (`python app.py`) before using this panel.")

    tab1, tab2, tab3, tab4 = st.tabs(["Customer Progress", "Knowledge Management", "Bot Testing", "Security Logs"])

    with tab1:
        st.header("📈 Customer Progress Reports")
        # (Content remains the same)
        admin_user_id = st.session_state.username
        if st.button("Refresh Customer Reports"):
            st.rerun()
        reports = get_all_customer_reports(user_id=admin_user_id)
        if not reports:
            st.info("No customer progress has been logged yet.")
        else:
            for customer, logs in reports.items():
                with st.expander(f"**Customer:** {customer}"):
                    for entry in logs:
                        st.markdown(entry)

    with tab2:
        st.header("Manage Bot Knowledge & Persona")
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.subheader("📚 Foundational Knowledge")
                st.markdown("Upload documents to the bot's main knowledge base.")
                knowledge_file = st.file_uploader("Upload Document", type=["pdf", "docx"], key="knowledge")
                # --- NEW: Metadata input field ---
                knowledge_tags = st.text_input("Add tags (comma-separated)", placeholder="e.g., low-sodium, diabetic-friendly", key="knowledge_tags")
                if st.button("Add to Knowledge Base") and knowledge_file:
                    metadata = {"tags": knowledge_tags}
                    upload_file("add-to-knowledge-base", knowledge_file, "Knowledge", metadata)
        with col2:
            with st.container(border=True):
                st.subheader("👤 Global Instructions")
                st.markdown("Upload a file to define the bot's core persona.")
                instructions_file = st.file_uploader("Upload Instructions", type=["docx", "pdf"], key="instructions")
                if st.button("Update Global Instructions") and instructions_file:
                    # Instructions don't need metadata, so we send an empty dict
                    upload_file("upload/instructions/global", instructions_file, "Instructions", {})
        with col3:
            with st.container(border=True):
                st.subheader("💰 Global Promotions")
                st.markdown("Upload a file with the latest sales or promotional info.")
                promo_file = st.file_uploader("Upload Promotions", type=["docx", "pdf"], key="promo")
                if st.button("Update Promotions") and promo_file:
                     # Promotions don't need metadata, so we send an empty dict
                    upload_file("upload/promo", promo_file, "Promotions", {})

    with tab3:
        st.header("Test the Bot (Global Persona)")
        # (Content remains the same)
        test_user_id = st.session_state.username
        TEST_CUSTOMER_ID = "test_customer_123"
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
                    chat_history = [msg for msg in st.session_state.admin_messages if isinstance(msg, dict)]
                    response_data = asyncio.run(get_contextual_response(prompt, chat_history, test_user_id, TEST_CUSTOMER_ID))
                    response_text = response_data.get("answer", "I'm sorry, an error occurred.")
                    st.write(response_text)
                    sources = response_data.get("sources", [])
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                source_name = os.path.basename(source.metadata.get('source', 'Unknown'))
                                st.info(f"Source: {source_name}, Page: {source.metadata.get('page', 'N/A')}")
                                st.caption(f"> {source.page_content[:250]}...")
            st.session_state.admin_messages.append({"role": "assistant", "content": response_text})

    with tab4:
        st.header("🔑 Security & Access Logs")
        # (Content remains the same)
        if st.button("Refresh Logs"):
            st.rerun()
        logs = get_access_logs()
        if logs:
            df = pd.DataFrame(logs, columns=["Username", "Timestamp", "Status"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No access attempts have been logged yet.")

else:
    st.title("Welcome to the Chatbot Admin Panel")
    st.header("Please log in or sign up using the sidebar to continue.")

