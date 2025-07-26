import os
import shutil
import time
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BASE_DOCS_PATH = os.path.join(DATA_DIR, "base_documents")

PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")
COLLECTION_NAME = "user_knowledge"

# --- NEW: Function to load persona and promo instructions ---
def get_prompts():
    """Loads the latest instruction and promo files."""
    instructions_path = os.path.join(DATA_DIR, "instructions")
    promos_path = os.path.join(DATA_DIR, "promos")

    # Load instructions
    instruction_docs = []
    if os.path.exists(instructions_path):
        for filename in os.listdir(instructions_path):
            if filename.endswith(".docx"):
                loader = Docx2txtLoader(os.path.join(instructions_path, filename))
                instruction_docs.extend(loader.load())
    
    # Load promos
    promo_docs = []
    if os.path.exists(promos_path):
        for filename in os.listdir(promos_path):
            if filename.endswith(".docx"):
                loader = Docx2txtLoader(os.path.join(promos_path, filename))
                promo_docs.extend(loader.load())

    instructions_text = "\n\n".join([doc.page_content for doc in instruction_docs]) if instruction_docs else "No special instructions provided."
    promos_text = "\n\n".join([doc.page_content for doc in promo_docs]) if promo_docs else "No active promotions."
    
    return instructions_text, promos_text

def build_user_database(user_id: str, uploaded_docx_files: list, status_callback=None):
    """
    Builds a user-specific knowledge base by combining the foundational PDF knowledge
    with the user's newly uploaded DOCX files.
    This creates the database in a single, robust operation.
    
    Args:
        user_id (str): The unique identifier for the user.
        uploaded_docx_files (list): A list of uploaded file objects from Streamlit.
        status_callback (function, optional): A function to call with status updates.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    user_db_path = os.path.join(USER_DB_PATH, user_id)
    
    if status_callback: status_callback(f"--- Starting new build for user: {user_id} ---")

    # 1. Clear any existing database for this user to ensure a fresh start
    if os.path.exists(user_db_path):
        if status_callback: status_callback("Clearing old knowledge base...")
        shutil.rmtree(user_db_path)
        time.sleep(1) # Add a small delay to ensure the directory is fully removed

    # 2. Load all documents (base PDFs + uploaded DOCX) into memory
    if status_callback: status_callback("Loading documents...")
    all_docs = []
    
    # Load base documents
    if os.path.exists(BASE_DOCS_PATH):
        for filename in os.listdir(BASE_DOCS_PATH):
            if filename.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(os.path.join(BASE_DOCS_PATH, filename))
                    all_docs.extend(loader.load())
                except Exception as e:
                    if status_callback: status_callback(f"Warning: Could not load base file {filename}. Error: {e}")

    # Load user-uploaded documents
    for file_obj in uploaded_docx_files:
        try:
            # Create a temporary path to load the document
            temp_dir = os.path.join(BASE_DIR, "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, file_obj.name)
            with open(temp_path, "wb") as f:
                f.write(file_obj.getvalue())
            
            loader = Docx2txtLoader(temp_path)
            all_docs.extend(loader.load())
            
            # Clean up the temporary file
            os.remove(temp_path)
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            if status_callback: status_callback(f"Error loading uploaded file {file_obj.name}: {e}")
            continue

    if not all_docs:
        if status_callback: status_callback("No documents found to process.")
        return

    # 3. Split all documents together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    if status_callback: status_callback(f"Split documents into {len(chunks)} chunks.")

    # 4. Create the new database in a single, robust operation
    if chunks:
        if status_callback: status_callback(f"Embedding documents... This may take a while depending on the number of documents.")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=6, chunk_size=500)
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=user_db_path,
            collection_name=COLLECTION_NAME
        )
        
        if status_callback: status_callback("✅ Training complete! Your knowledge base is ready.")
    else:
        if status_callback: status_callback("No content to train on after processing files.")
