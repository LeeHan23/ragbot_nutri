import os
import shutil
import argparse
import time # <-- IMPORT THE TIME MODULE
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory where the foundational PDFs are stored
BASE_DOCS_DIR = os.path.join(BASE_DIR, "data", "base_documents")
# Directory where the final user-specific databases are saved
USER_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "user_knowledge"

# --- List of foundational PDF documents ---
BASE_PDF_FILES = [
    "FA-Buku-RNI.pdf",
    "latest-01.Buku-MDG-2020_12Mac2024.pdf"
]

def build_user_database(user_id: str, uploaded_docx_files: list, status_callback=None):
    """
    Builds a complete, user-specific knowledge base by combining foundational
    PDFs with newly uploaded user documents in a single, robust operation.
    
    Args:
        user_id (str): The unique identifier for the user.
        uploaded_docx_files (list): A list of uploaded file objects from Streamlit.
        status_callback (function, optional): A function to call with status updates.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    user_db_path = os.path.join(USER_DB_PATH, user_id)
    
    if status_callback: status_callback(f"--- Starting fresh build for user: {user_id} ---")

    # 1. Clear any existing database for this user to ensure a fresh start
    if os.path.exists(user_db_path):
        if status_callback: status_callback("Clearing old custom knowledge base...")
        shutil.rmtree(user_db_path)
        # --- ADDED: A short delay to allow the file system to process the deletion ---
        time.sleep(1) 
        if status_callback: status_callback("Old knowledge base cleared.")

    all_docs = []

    # 2. Load the foundational knowledge from the base PDFs
    if status_callback: status_callback("Loading foundational knowledge...")
    for filename in BASE_PDF_FILES:
        path = os.path.join(BASE_DOCS_DIR, filename)
        if os.path.exists(path):
            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
            except Exception as e:
                if status_callback: status_callback(f"Error loading base file {filename}: {e}")
        else:
            if status_callback: status_callback(f"Warning: Base document not found: {filename}")
    
    # 3. Load the user's newly uploaded documents
    if status_callback: status_callback("Loading new user documents...")
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

    # 4. Split all documents together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    if status_callback: status_callback(f"Split all documents into {len(chunks)} chunks.")

    # 5. Create the new database in a single, robust operation
    if chunks:
        if status_callback: status_callback("Embedding documents... This may take several minutes.")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=6, chunk_size=500)
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=user_db_path,
            collection_name=COLLECTION_NAME
        )
        
        if status_callback: status_callback("✅ Training complete! New knowledge base is ready.")
    else:
        if status_callback: status_callback("No content found in documents to train on.")

if __name__ == "__main__":
    print("This script is primarily intended to be called from the UI.")
