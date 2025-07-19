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
# Define the directory where your core PDFs are stored
BASE_DOCS_DIR = os.path.join("data", "base_documents")
# Define the directory where the foundational database will be saved
BASE_INDEX_DIR = "vectorstore_base"
COLLECTION_NAME = "base_knowledge"
# =================================

def build_base_database():
    """
    This script builds the foundational vector store by automatically finding
    and processing ALL .pdf files in the 'data/base_documents' directory.
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

    # --- MODIFIED: Automatically find all PDF files in the directory ---
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

    # 3. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split base documents into {len(chunks)} chunks.")

    # 4. Create the database in a single, robust operation
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
