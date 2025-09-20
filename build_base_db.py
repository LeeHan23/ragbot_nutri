import os
import shutil
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_chroma import Chroma # <-- UPDATED IMPORT
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# --- UNIFIED PATH CONFIGURATION ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_PATH = os.path.join(APP_DIR, "data")
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", LOCAL_DATA_PATH)

# ========= CONFIGURATION =========
BASE_DOCS_DIR = os.path.join(APP_DIR, "data", "base_documents")
BASE_INDEX_DIR = os.path.join(PERSISTENT_DISK_PATH, "vectorstore_base") # Use the consistent path
COLLECTION_NAME = "base_knowledge"
# =================================

def build_base_database():
    """
    Builds the foundational vector store by processing all .pdf files
    in the 'data/base_documents' directory.
    """
    print("--- Building Foundational Knowledge Base ---")
    
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        max_retries=10
    )

    if os.path.exists(BASE_INDEX_DIR):
        print(f"Clearing existing base vector store at: {BASE_INDEX_DIR}")
        shutil.rmtree(BASE_INDEX_DIR)
    
    # Ensure the parent directory exists
    os.makedirs(PERSISTENT_DISK_PATH, exist_ok=True)
    print("Base vector store directory prepared.")

    all_docs = []
    print(f"Scanning for PDF documents in '{BASE_DOCS_DIR}'...")
    
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
        print("No PDF documents were found or loaded. Aborting database creation.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split base documents into {len(chunks)} chunks.")

    if chunks:
        print(f"\nCreating and persisting the base vector store at {BASE_INDEX_DIR}...")
        print("This may take several minutes...")
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=BASE_INDEX_DIR,
            collection_name=COLLECTION_NAME
        )
        
        print("\n✅ Foundational knowledge base created successfully!")
    else:
        print("No content to process after splitting.")

if __name__ == "__main__":
    build_base_database()

