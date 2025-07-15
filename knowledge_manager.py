import os
import shutil
import argparse
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

# LangChain imports for loading/splitting documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

# ChromaDB native client for writing and its embedding function helper
import chromadb
from chromadb.utils import embedding_functions as chroma_embedding_functions

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "users")
BASE_DB_PATH = "vectorstore_base"
USER_DB_PATH = "chroma_db"
COLLECTION_NAME = "user_knowledge"

def add_documents_to_user_db(user_id: str, list_of_doc_paths: list, status_callback=None):
    """
    Adds new documents to a user's specific knowledge base using the native ChromaDB client.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    user_db_path = os.path.join(USER_DB_PATH, user_id)

    # Step 1: Ensure the user has a starting database by copying the base
    if not os.path.exists(user_db_path):
        if status_callback: status_callback("Creating personal knowledge base for new user...")
        if not os.path.exists(BASE_DB_PATH):
            if status_callback: status_callback("Error: Foundational knowledge base not found.")
            return
        shutil.copytree(BASE_DB_PATH, user_db_path)
        if status_callback: status_callback("Personal knowledge base created.")

    # Step 2: Load and process the new .docx documents
    if status_callback: status_callback("Loading new documents...")
    all_new_docs = []
    for doc_path in list_of_doc_paths:
        try:
            loader = Docx2txtLoader(doc_path)
            all_new_docs.extend(loader.load())
        except Exception as e:
            if status_callback: status_callback(f"Error loading {os.path.basename(doc_path)}: {e}")
            continue

    if not all_new_docs:
        if status_callback: status_callback("No new documents to add.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_new_docs)
    if status_callback: status_callback(f"Split new documents into {len(chunks)} chunks.")

    # Step 3: Add new chunks using the native ChromaDB client to avoid file locks
    if chunks:
        try:
            if status_callback: status_callback("Connecting to user database...")
            
            # Initialize the native ChromaDB client
            client = chromadb.PersistentClient(path=user_db_path)
            
            # Get the OpenAI API key for the embedding function
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            
            # Create a ChromaDB-compatible embedding function
            chroma_ef = chroma_embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-ada-002"
            )

            # --- CORRECTED: Use get_or_create_collection to prevent errors for new users ---
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME, # This should be the user collection name
                embedding_function=chroma_ef
            )
            
            if status_callback: status_callback("Adding new knowledge to the database...")
            
            # Prepare data for the native client, ensuring unique IDs
            existing_ids_count = collection.count()
            ids = [f"user_chunk_{existing_ids_count + i}" for i in range(len(chunks))]
            documents_content = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]

            # Add the new documents to the collection
            collection.add(
                documents=documents_content,
                metadatas=metadatas,
                ids=ids
            )
            
            if status_callback: status_callback("✅ Knowledge base successfully updated!")

        except Exception as e:
            if status_callback: status_callback(f"An error occurred during database update: {e}")
            raise e

# This part of the script is for manual command-line testing and is not used by the UI.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add documents to a user's knowledge base.")
    parser.add_argument("--user_id", type=str, required=True, help="The unique identifier for the user.")
    parser.add_argument("files", nargs='+', help="The path(s) to the .docx file(s) to add.")
    args = parser.parse_args()
    
    add_documents_to_user_db(args.user_id, args.files, status_callback=print)
