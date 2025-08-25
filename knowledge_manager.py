import os
import shutil
import chromadb
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from fastapi import UploadFile
from uploader import save_uploaded_file_as_text

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")

# Paths for Foundational Knowledge Base
BASE_DOCS_DIR = os.path.join(BASE_DIR, "data", "base_documents")
BASE_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "vectorstore_base")
BASE_COLLECTION_NAME = "base_knowledge"

# Paths for User-Specific Knowledge & Instructions
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")
USER_COLLECTION_NAME = "user_knowledge"
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
TEMP_UPLOADS_DIR = os.path.join(BASE_DIR, "temp_uploads")


# --- Foundational Knowledge Base Management ---

def build_base_database():
    """
    Builds the foundational vector store from all .pdf files in the base_documents directory.
    This is typically run once during initial setup.
    """
    print("--- Building Foundational Knowledge Base ---")
    if os.path.exists(BASE_DB_PATH):
        print(f"Clearing existing base vector store at: {BASE_DB_PATH}")
        shutil.rmtree(BASE_DB_PATH)

    all_docs = []
    if not os.path.exists(BASE_DOCS_DIR):
        print(f"Error: The directory '{BASE_DOCS_DIR}' was not found.")
        return

    for filename in os.listdir(BASE_DOCS_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(BASE_DOCS_DIR, filename)
            try:
                loader = PyMuPDFLoader(path)
                all_docs.extend(loader.load())
                print(f"Successfully loaded {filename}.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not all_docs:
        print("No PDF documents found. Aborting database creation.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split base documents into {len(chunks)} chunks.")

    if chunks:
        print("\nCreating and persisting the base vector store...")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=BASE_DB_PATH,
            collection_name=BASE_COLLECTION_NAME
        )
        print("\n✅ Foundational knowledge base created successfully!")

def add_pdf_to_base_db(pdf_path: str):
    """
    Incrementally adds a single PDF to the existing foundational knowledge base.
    """
    if not os.path.exists(BASE_DB_PATH):
        print(f"Error: The base vector store at '{BASE_DB_PATH}' does not exist.")
        return False
    
    print(f"--- Starting incremental update for: {os.path.basename(pdf_path)} ---")
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            print("No text chunks to add.")
            return False

        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = Chroma(
            persist_directory=BASE_DB_PATH,
            embedding_function=embedding_function,
            collection_name=BASE_COLLECTION_NAME
        )
        vector_store.add_documents(chunks)
        print("✅ Incremental update complete.")
        return True
    except Exception as e:
        print(f"An error occurred during the incremental update: {e}")
        return False

# --- User-Specific Knowledge and Instructions Management ---

def build_user_database(user_id: str, uploaded_docx_files: list, status_callback=None):
    """
    Builds a user-specific knowledge base from uploaded .docx files.
    """
    user_db_path = os.path.join(USER_DB_PATH, user_id)
    if status_callback: status_callback(f"Starting custom build for user: {user_id}")

    if status_callback: status_callback("Resetting database connection...")
    chromadb.Client().reset()

    if os.path.exists(user_db_path):
        if status_callback: status_callback("Clearing old custom knowledge base...")
        shutil.rmtree(user_db_path)

    all_docs = []
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
    for file_obj in uploaded_docx_files:
        temp_path = os.path.join(TEMP_UPLOADS_DIR, file_obj.name)
        with open(temp_path, "wb") as f:
            f.write(file_obj.getvalue())
        loader = Docx2txtLoader(temp_path)
        all_docs.extend(loader.load())
        os.remove(temp_path)

    if not all_docs:
        if status_callback: status_callback("No documents found to process.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    if chunks:
        if status_callback: status_callback(f"Embedding {len(chunks)} document chunks...")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=user_db_path,
            collection_name=USER_COLLECTION_NAME
        )
        if status_callback: status_callback("✅ Training complete!")
    else:
        if status_callback: status_callback("No content found to train on.")

def save_instruction_file(user_id: str, uploaded_file: UploadFile):
    """
    Saves a user-uploaded .docx file as their specific instruction text.
    """
    user_instructions_dir = os.path.join(INSTRUCTIONS_PATH, str(user_id))
    os.makedirs(user_instructions_dir, exist_ok=True)
    try:
        saved_path = save_uploaded_file_as_text(uploaded_file, user_instructions_dir)
        print(f"Saved new instructions for user '{user_id}' at: {saved_path}")
        return saved_path
    except Exception as e:
        print(f"Error saving instruction file for user '{user_id}': {e}")
        return None

# --- Prompt Loading ---

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
    Loads persona instructions, prioritizing user-specific ones, and global promotions.
    """
    print("Loading persona instructions and promotions...")
    instructions = ""
    if user_id:
        user_instructions_dir = os.path.join(INSTRUCTIONS_PATH, str(user_id))
        if os.path.exists(user_instructions_dir):
            instructions = _get_latest_file_content(user_instructions_dir)

    if not instructions:
        instructions = _get_latest_file_content(INSTRUCTIONS_PATH)

    promotions = _get_latest_file_content(PROMOS_PATH)
    
    if not instructions: instructions = "You are a helpful general assistant."
    if not promotions: promotions = "There are no special promotions at this time."
        
    full_instructions = f"{instructions}\n\n[LATEST PROMOTIONS & OFFERS]\n{promotions}"
    return full_instructions, ""
