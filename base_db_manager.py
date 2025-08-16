import os
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# --- Constants ---
# Define the path to the foundational database
BASE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore_base")
BASE_COLLECTION_NAME = "base_knowledge"

def add_pdf_to_base_db(pdf_path: str):
    """
    Processes a single PDF file and adds its content to the existing
    foundational knowledge base.

    This function performs an incremental update, meaning it does not
    rebuild the entire database. It loads the new document, splits it
    into chunks, and adds those chunks to the existing ChromaDB store.

    Args:
        pdf_path (str): The full path to the PDF file to be added.
    
    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return False

    if not os.path.exists(BASE_DB_PATH):
        print(f"Error: The base vector store at '{BASE_DB_PATH}' does not exist. Please build it first using `build_base_db.py`.")
        return False

    print(f"--- Starting incremental update for: {os.path.basename(pdf_path)} ---")

    try:
        # 1. Load the new document
        print(f"Loading document: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            print("Warning: The PDF document appears to be empty or could not be loaded.")
            return False

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split document into {len(chunks)} chunks.")

        if not chunks:
            print("No text chunks to add to the database.")
            return False

        # 3. Initialize the embedding function
        embedding_function = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            max_retries=6,
            chunk_size=500
        )

        # 4. Load the existing vector store
        print(f"Loading existing vector store from: {BASE_DB_PATH}")
        vector_store = Chroma(
            persist_directory=BASE_DB_PATH,
            embedding_function=embedding_function,
            collection_name=BASE_COLLECTION_NAME
        )

        # 5. Add the new document chunks to the vector store
        print(f"Adding {len(chunks)} new chunks to the vector store...")
        vector_store.add_documents(chunks)
        
        # Note: ChromaDB with a persistent client auto-saves changes.
        # There is no explicit .save() method needed.

        print("✅ Incremental update complete. The new document has been added to the knowledge base.")
        return True

    except Exception as e:
        print(f"An error occurred during the incremental update: {e}")
        return False
