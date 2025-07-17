import os
import shutil
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# ========= CONFIGURATION =========
BASE_PDF_FILES = [
    "FA-Buku-RNI.pdf",
    "latest-01.Buku-MDG-2020_12Mac2024.pdf"
]
# The code's location of the source documents
BASE_DOCS_DIR = os.path.join("data", "base_documents")
# --- CORRECTED: Point to the persistent disk mount path ---
PERSISTENT_DISK_PATH = "/data"
BASE_INDEX_DIR = os.path.join(PERSISTENT_DISK_PATH, "vectorstore_base")
COLLECTION_NAME = "base_knowledge"
# =================================

def build_base_database():
    """
    This script builds the foundational vector store from the core PDF documents
    and saves it to the persistent disk at /data/vectorstore_base.
    """
    print("--- Building Foundational Knowledge Base ---")
    
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        max_retries=10,
        retry_min_seconds=20,
        retry_max_seconds=60,
        chunk_size=500
    )

    if os.path.exists(BASE_INDEX_DIR):
        print(f"Clearing existing base vector store at: {BASE_INDEX_DIR}")
        shutil.rmtree(BASE_INDEX_DIR)
    print("Base vector store cleared.")

    all_docs = []
    print("Loading core PDF documents...")
    for filename in BASE_PDF_FILES:
        path = os.path.join(BASE_DOCS_DIR, filename)
        if not os.path.exists(path):
            print(f"Warning: Could not find file {path}. Skipping.")
            continue
        try:
            loader = PyMuPDFLoader(path)
            all_docs.extend(loader.load())
            print(f"Successfully loaded {filename}.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not all_docs:
        print("No base documents were loaded. Aborting database creation.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split base documents into {len(chunks)} chunks.")

    if chunks:
        print("\nCreating and persisting the base vector store...")
        print("This is a one-time setup and may take several minutes...")
        
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
