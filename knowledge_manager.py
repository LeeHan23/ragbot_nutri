import os
import shutil
import argparse
import chromadb
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

# --- Constants ---
# This section defines key paths and names used throughout the module.
# It ensures that the application knows where to find and store data.
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")
COLLECTION_NAME = "user_knowledge"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_UPLOADS_DIR = os.path.join(BASE_DIR, "temp_uploads")
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")


# --- Prompt Loading Functions ---
# These functions are responsible for dynamically loading the AI's persona
# and any promotional information from text files.

def _get_latest_file_content(directory: str) -> str:
    """
    Finds the most recently modified .txt file in a directory, reads it,
    and returns its content as a string. This ensures the bot always
    uses the most up-to-date instructions or promotions.
    """
    try:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return ""
            
        # Get a list of all .txt files in the specified directory
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
        if not files:
            return ""
        
        # Find the file with the most recent modification time
        latest_file = max(files, key=os.path.getmtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
            
    except Exception as e:
        print(f"Error reading from {directory}: {e}")
        return ""

def get_prompts() -> tuple[str, str]:
    """
    Loads the content of the latest persona instructions and promotions files.
    This is called by the RAG chain every time a user sends a message.
    """
    print("Loading persona instructions and latest promotions...")
    
    instructions = _get_latest_file_content(INSTRUCTIONS_PATH)
    promotions = _get_latest_file_content(PROMOS_PATH)
    
    # Provide default text if the instruction or promotion files are missing or empty
    if not instructions:
        instructions = "You are a helpful general assistant. Since no specific instructions were provided, answer questions concisely."
    if not promotions:
        promotions = "There are no special promotions at this time."
        
    # Combine the instructions and promotions into a single block for the AI
    full_instructions = f"{instructions}\n\n[LATEST PROMOTIONS & OFFERS]\n{promotions}"
    
    # Return the combined instructions. The second empty string is for compatibility
    # with the function signature expected in rag.py.
    return full_instructions, ""


# --- Knowledge Base Building Functions ---

def build_user_database(user_id: str, uploaded_docx_files: list, status_callback=None):
    """
    Builds a user-specific knowledge base from uploaded .docx files.
    This function is triggered by the user in the Streamlit UI.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    user_db_path = os.path.join(USER_DB_PATH, user_id)
    
    if status_callback: status_callback(f"--- Starting new custom build for user: {user_id} ---")

    # FIX: To prevent file lock errors, we must reset the ChromaDB client.
    # This forces any lingering read connections to close before we attempt to write.
    if status_callback: status_callback("Resetting database connection...")
    chromadb.Client().reset()

    # 1. Clear any old custom database for this user to ensure a fresh start.
    if os.path.exists(user_db_path):
        if status_callback: status_callback("Clearing old custom knowledge base...")
        shutil.rmtree(user_db_path)

    # 2. Load the user's newly uploaded documents into memory.
    if status_callback: status_callback("Loading user documents...")
    all_docs = []
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
    for file_obj in uploaded_docx_files:
        try:
            # Save the file temporarily to disk to be loaded by the document loader
            temp_path = os.path.join(TEMP_UPLOADS_DIR, file_obj.name)
            with open(temp_path, "wb") as f:
                f.write(file_obj.getvalue())
            
            loader = Docx2txtLoader(temp_path)
            all_docs.extend(loader.load())
            
            os.remove(temp_path) # Clean up the temporary file
        except Exception as e:
            if status_callback: status_callback(f"Error loading uploaded file {file_obj.name}: {e}")
            continue
    
    # Clean up the temporary directory if it's empty
    if os.path.exists(TEMP_UPLOADS_DIR) and not os.listdir(TEMP_UPLOADS_DIR):
        os.rmdir(TEMP_UPLOADS_DIR)

    if not all_docs:
        if status_callback: status_callback("No documents found to process.")
        return

    # 3. Split the loaded documents into smaller, manageable chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    if status_callback: status_callback(f"Split documents into {len(chunks)} chunks.")

    # 4. Create the new vector database from the document chunks.
    if chunks:
        if status_callback: status_callback("Embedding documents... This may take a moment.")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=6, chunk_size=500)
        
        # This is the main operation that creates the persistent vector store on disk.
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
    print("This script is not intended to be run directly. It is imported by other modules.")
