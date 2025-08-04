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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_UPLOADS_DIR = os.path.join(BASE_DIR, "temp_uploads")

# Define paths for prompts, consistent with admin.py
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")


# --- Prompt Loading Functions ---

def _get_latest_file_content(directory: str) -> str:
    """
    Finds the most recently modified .txt file in a directory and returns its content.
    """
    try:
        # Ensure the directory exists before trying to list its contents
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return ""
            
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
        if not files:
            return ""  # Return empty string if no .txt files are found
        
        latest_file = max(files, key=os.path.getmtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
            
    except Exception as e:
        print(f"Error reading from {directory}: {e}")
        return ""

def get_prompts() -> tuple[str, str]:
    """
    Loads the content of the latest persona instructions and promotions files.
    This function is imported by rag.py to build the AI's prompt.
    
    Returns:
        A tuple containing (instructions, promotions).
    """
    print("Loading persona instructions and latest promotions...")
    
    instructions = _get_latest_file_content(INSTRUCTIONS_PATH)
    promotions = _get_latest_file_content(PROMOS_PATH)
    
    # Provide default text if files are empty or not found
    if not instructions:
        instructions = "You are a helpful general assistant. Since no specific instructions were provided, answer questions concisely."
    if not promotions:
        promotions = "There are no special promotions at this time."
        
    # Combine promotions into the main instructions for a complete context
    full_instructions = f"{instructions}\n\n[LATEST PROMOTIONS & OFFERS]\n{promotions}"
    
    # The function signature in rag.py expects two values. 
    # We return the combined instructions and an empty string for the second value
    # to maintain compatibility without needing to change rag.py.
    return full_instructions, ""


# --- Knowledge Base Building Functions ---

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

