import os
import shutil
import argparse
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory for user-uploaded DOCX files (converted to TXT)
USER_KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "users") 
# Directory for the foundational PDF knowledge base
BASE_DB_PATH = "vectorstore_base" 
# Directory for user-specific, augmented knowledge bases
USER_DB_PATH = "chroma_db"
COLLECTION_NAME = "user_knowledge"

def add_documents_to_user_db(user_id: str, list_of_doc_paths: list, status_callback=None):
    """
    Adds new documents to a user's specific knowledge base.
    If the user's knowledge base doesn't exist, it creates it by copying the base DB first.
    
    Args:
        user_id (str): The unique identifier for the user.
        list_of_doc_paths (list): A list of file paths to the new .docx documents to add.
        status_callback (function, optional): A function to call with status updates.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    user_db_path = os.path.join(USER_DB_PATH, user_id)
    
    # Step 1: Ensure the user has a starting database
    if not os.path.exists(user_db_path):
        if status_callback: status_callback("Creating personal knowledge base for new user...")
        if not os.path.exists(BASE_DB_PATH):
            if status_callback: status_callback("Error: Foundational knowledge base not found. Please run 'build_base_db.py' first.")
            return
        # Copy the base database to the user's specific path
        shutil.copytree(BASE_DB_PATH, user_db_path)
        if status_callback: status_callback("Personal knowledge base created.")

    # Step 2: Load and process the new documents
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

    # Step 3: Add the new chunks to the user's existing database
    if chunks:
        if status_callback: status_callback("Adding new knowledge to the database...")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Load the user's database
        db = Chroma(
            persist_directory=user_db_path,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME # Use a consistent collection name
        )
        
        # Add the new document chunks
        db.add_documents(chunks)
        db.persist() # Save the changes
        
        if status_callback: status_callback("✅ Knowledge base successfully updated!")
    else:
        if status_callback: status_callback("No new content to add.")

if __name__ == "__main__":
    # This part allows running the script from the command line for testing, if needed.
    parser = argparse.ArgumentParser(description="Add documents to a user's knowledge base.")
    parser.add_argument("--user_id", type=str, required=True, help="The unique identifier for the user.")
    parser.add_argument("files", nargs='+', help="The path(s) to the .docx file(s) to add.")
    args = parser.parse_args()
    
    add_documents_to_user_db(args.user_id, args.files, status_callback=print)
