import os
import shutil
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from fastapi import UploadFile
from uploader import save_uploaded_file_as_text
import chromadb # Import the base chromadb library
from langchain_chroma import Chroma # <-- UPDATED IMPORT

# --- UNIFIED PATH CONFIGURATION ---
# This new section ensures paths are consistent across local dev and production.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Default local data path is a 'data' folder in your project root
LOCAL_DATA_PATH = os.path.join(APP_DIR, "data")
# Use Render's persistent disk path if available, otherwise use the local path
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", LOCAL_DATA_PATH)

# --- Constants ---
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db") 
BASE_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "vectorstore_base")
USER_COLLECTION_NAME = "user_knowledge"
BASE_COLLECTION_NAME = "base_knowledge"
PROMOS_PATH = os.path.join(APP_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(APP_DIR, "data", "instructions")


# --- KNOWLEDGE BASE MANAGEMENT ---

def add_pdf_to_base_db(pdf_path: str):
    """
    Processes a single PDF file and adds its content to the existing
    foundational knowledge base. This is an incremental update.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return False

    if not os.path.exists(BASE_DB_PATH):
        print(f"Error: Base vector store at '{BASE_DB_PATH}' does not exist. Please build it first.")
        return False

    print(f"--- Starting incremental update for: {os.path.basename(pdf_path)} ---")
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            print("Warning: The PDF document appears to be empty or could not be loaded.")
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split document into {len(chunks)} chunks.")

        if not chunks:
            print("No text chunks to add to the database.")
            return False

        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Load the existing vector store
        vector_store = Chroma(
            persist_directory=BASE_DB_PATH,
            embedding_function=embedding_function,
            collection_name=BASE_COLLECTION_NAME
        )
        
        # Add the new document chunks
        vector_store.add_documents(chunks)
        
        print("✅ Incremental update complete.")
        return True
    except Exception as e:
        print(f"An error occurred during the incremental update: {e}")
        return False

def build_user_database(user_id: str, uploaded_docx_files: list, status_callback=None):
    """
    Builds a user-specific knowledge base from uploaded .docx files.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    user_db_path = os.path.join(USER_DB_PATH, user_id)
    if status_callback: status_callback(f"Preparing to build knowledge base for user '{user_id}'...")

    # Safely remove old database before creating a new one
    if os.path.exists(user_db_path):
        if status_callback: status_callback("Clearing old custom knowledge base...")
        shutil.rmtree(user_db_path)
    
    os.makedirs(user_db_path, exist_ok=True)
    
    all_docs = []
    # Create a temporary directory inside the user's db path to avoid conflicts
    temp_dir = os.path.join(user_db_path, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    for file_obj in uploaded_docx_files:
        try:
            temp_file_path = os.path.join(temp_dir, file_obj.name)
            
            # uploaded_docx_files from streamlit are BytesIO objects
            with open(temp_file_path, "wb") as f:
                f.write(file_obj.getbuffer())

            loader = Docx2txtLoader(temp_file_path)
            all_docs.extend(loader.load())

        except Exception as e:
            if status_callback: status_callback(f"Error loading file {file_obj.name}: {e}")
            continue
    
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)

    if not all_docs:
        if status_callback: status_callback("No documents found to process.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    if status_callback: status_callback(f"Split documents into {len(chunks)} chunks.")

    if chunks:
        if status_callback: status_callback("Embedding documents...")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=user_db_path,
            collection_name=USER_COLLECTION_NAME
        )
        if status_callback: status_callback("✅ Training complete!")
    else:
        if status_callback: status_callback("No content found in documents.")


# --- INSTRUCTION & PROMPT MANAGEMENT ---

def save_instruction_file(user_id: str, uploaded_file: UploadFile):
    """Saves a user-uploaded .docx file as their specific instruction text."""
    if not user_id or not uploaded_file:
        return None

    user_instructions_dir = os.path.join(INSTRUCTIONS_PATH, str(user_id))
    os.makedirs(user_instructions_dir, exist_ok=True)
    
    try:
        # Use the uploader utility to handle file processing
        # This assumes uploader.py's save_uploaded_file_as_text is available and works with FastAPI's UploadFile
        saved_path = save_uploaded_file_as_text(uploaded_file, user_instructions_dir)
        return saved_path
        
    except Exception as e:
        print(f"Error saving instruction file for user '{user_id}': {e}")
        return None

def _get_latest_file_content(directory: str) -> str:
    """Finds the most recently modified .txt file in a directory and returns its content."""
    try:
        if not os.path.exists(directory): return ""
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
        if not files: return ""
        latest_file = max(files, key=os.path.getmtime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading from {directory}: {e}")
        return ""

def get_prompts(user_id: str = None) -> tuple[str, str]:
    """
    Loads persona instructions, prioritizing user-specific ones.
    """
    instructions = ""
    if user_id:
        user_instructions_dir = os.path.join(INSTRUCTIONS_PATH, str(user_id))
        if os.path.exists(user_instructions_dir):
            instructions = _get_latest_file_content(user_instructions_dir)

    if not instructions:
        instructions = _get_latest_file_content(INSTRUCTIONS_PATH)

    promotions = _get_latest_file_content(PROMOS_PATH)
    
    if not instructions:
        instructions = "You are a helpful general assistant."
    if not promotions:
        promotions = "There are no special promotions at this time."
        
    full_instructions = f"{instructions}\n\n[LATEST PROMOTIONS & OFFERS]\n{promotions}"
    
    return full_instructions, ""

