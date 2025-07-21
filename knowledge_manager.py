import os
import shutil
import argparse
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

# --- Constants ---
# Correctly point to the persistent disk path provided by the environment
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")
COLLECTION_NAME = "user_knowledge"
# Define a local temporary directory for file processing
TEMP_UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")


def build_user_database(user_id: str, uploaded_docx_files: list, status_callback=None):
    """
    Builds a user-specific knowledge base containing ONLY the user's uploaded documents.
    This is a much faster and more memory-efficient process that avoids file-locking errors.
    
    Args:
        user_id (str): The unique identifier for the user.
        uploaded_docx_files (list): A list of uploaded file objects from Streamlit.
        status_callback (function, optional): A function to call with status updates.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    user_db_path = os.path.join(USER_DB_PATH, user_id)
    
    if status_callback: status_callback(f"--- Starting new custom build for user: {user_id} ---")

    # 1. Clear any existing custom database for this user to ensure a fresh start
    if os.path.exists(user_db_path):
        if status_callback: status_callback("Clearing old custom knowledge base...")
        shutil.rmtree(user_db_path)

    # 2. Load the user's newly uploaded documents
    if status_callback: status_callback("Loading user documents...")
    all_docs = []
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True) # Ensure temp directory exists
    for file_obj in uploaded_docx_files:
        try:
            # Create a temporary path to load the document
            temp_path = os.path.join(TEMP_UPLOADS_DIR, file_obj.name)
            with open(temp_path, "wb") as f:
                f.write(file_obj.getvalue())
            
            loader = Docx2txtLoader(temp_path)
            all_docs.extend(loader.load())
            
            # Clean up the temporary file
            os.remove(temp_path)
        except Exception as e:
            if status_callback: status_callback(f"Error loading uploaded file {file_obj.name}: {e}")
            continue
    
    # Clean up temp directory if it's empty
    if os.path.exists(TEMP_UPLOADS_DIR) and not os.listdir(TEMP_UPLOADS_DIR):
        os.rmdir(TEMP_UPLOADS_DIR)

    if not all_docs:
        if status_callback: status_callback("No documents found to process.")
        return

    # 3. Split all documents together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    if status_callback: status_callback(f"Split documents into {len(chunks)} chunks.")

    # 4. Create the new user-specific database in a single, robust operation
    if chunks:
        if status_callback: status_callback("Embedding documents...")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=6, chunk_size=500)
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=user_db_path,
            collection_name=COLLECTION_NAME
        )
        
        if status_callback: status_callback("✅ Training complete! Your custom knowledge base is ready.")
    else:
        if status_callback: status_callback("No content found in documents to train on.")

if __name__ == "__main__":
    print("This script is primarily intended to be called from the UI.")
