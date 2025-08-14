import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from rag import get_contextual_response
from instructions_manager import save_instruction_file
from database import add_user, check_login, verify_user
from langchain_core.messages import HumanMessage, AIMessage

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
# Add state to manage the verification step
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
        # Ensure verification state is reset when switching to Sign Up
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
        
        # --- FIX: The verification UI is now outside the form ---
        if st.session_state.get("requires_verification"):
            st.info("This is your first login. Please enter your verification key to continue.")
            
            verification_key = st.text_input("One-Time Verification Key")
            
            if st.button("Verify and Login"):
                username_to_verify = st.session_state["username_for_verification"]
                # We need to re-check the user's name from the DB to log them in
                success, user_name, is_verified = check_login(username_to_verify, "") # We can pass an empty password as we are verifying with a key
                
                if verify_user(username_to_verify, verification_key):
                    st.session_state["authentication_status"] = True
                    st.session_state["name"] = user_name
                    st.session_state["username"] = username_to_verify
                    # Reset verification state
                    st.session_state["requires_verification"] = False
                    st.session_state["username_for_verification"] = ""
                    st.rerun()
                else:
                    st.error("Verification key is incorrect.")
        
        # Show the login form ONLY if verification is not required
        else:
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
                            # --- FIX: Set state to trigger verification UI ---
                            st.session_state["requires_verification"] = True
                            st.session_state["username_for_verification"] = username
                            st.rerun() # Rerun to show the verification UI
                    else:
                        st.error("Username or password is not correct.")

# If user IS authenticated, run the main chatbot app
else:
    # (The main application logic remains the same as before)
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
    
    if user_id not in st.session_state.messages:
        st.session_state.messages[user_id] = []
    
    with st.sidebar:
        st.divider()
        st.header("Customize Your Bot's Persona")
        st.info("Upload a .docx file with instructions on how your bot should behave.")

        uploaded_file = st.file_uploader(
            "Upload a .docx instruction file",
            type=['docx'],
            accept_multiple_files=False # Only one instruction file at a time
        )

        if st.button("Update Persona"):
            if uploaded_file:
                with st.spinner("Updating persona..."):
                    save_instruction_file(user_id, uploaded_file)
                st.success("✅ Persona updated successfully!")
            else:
                st.warning("Please upload a file first.")

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
