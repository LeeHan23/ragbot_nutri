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
# Directory where the foundational PDF knowledge base is stored
BASE_DB_PATH = os.path.join(BASE_DIR, "vectorstore_base")
# Directory where the final user-specific databases are saved
USER_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
# The collection name is consistent across base and user DBs after copying
COLLECTION_NAME = "base_knowledge" 

def build_user_database(user_id: str, uploaded_docx_files: list, status_callback=None):
    """
    Builds a user-specific knowledge base by copying the foundational DB
    and augmenting it with the user's uploaded documents.
    
    Args:
        user_id (str): The unique identifier for the user.
        uploaded_docx_files (list): A list of uploaded file objects from Streamlit.
        status_callback (function, optional): A function to call with status updates.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    user_db_path = os.path.join(USER_DB_PATH, user_id)
    
    if status_callback: status_callback(f"--- Starting knowledge build for user: {user_id} ---")

    # 1. If user has an existing custom DB, clear it to start fresh.
    if os.path.exists(user_db_path):
        if status_callback: status_callback("Clearing old custom knowledge base...")
        shutil.rmtree(user_db_path)

    # 2. Copy the foundational database to create the user's personal starting point
    if status_callback: status_callback("Copying foundational knowledge...")
    if not os.path.exists(BASE_DB_PATH):
        if status_callback: status_callback("FATAL ERROR: Foundational knowledge base not found. Please run 'build_base_db.py' first.")
        return
    shutil.copytree(BASE_DB_PATH, user_db_path)

    # 3. Load the user's newly uploaded documents
    if status_callback: status_callback("Loading new user documents...")
    all_new_docs = []
    for file_obj in uploaded_docx_files:
        try:
            # Create a temporary path to load the document
            temp_dir = os.path.join(BASE_DIR, "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, file_obj.name)
            with open(temp_path, "wb") as f:
                f.write(file_obj.getvalue())
            
            loader = Docx2txtLoader(temp_path)
            all_new_docs.extend(loader.load())
            
            os.remove(temp_path) # Clean up the temporary file
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            if status_callback: status_callback(f"Error loading uploaded file {file_obj.name}: {e}")
            continue

    if not all_new_docs:
        if status_callback: status_callback("No new documents to process. User DB is now a copy of the base knowledge.")
        return

    # 4. Split the new documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_new_docs)
    if status_callback: status_callback(f"Split new documents into {len(chunks)} chunks.")

    # 5. Add the new chunks to the user's copied database
    if chunks:
        if status_callback: status_callback("Embedding new documents and adding to knowledge base...")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Load the user's new database
        db = Chroma(
            persist_directory=user_db_path,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME 
        )
        
        # Add the new document chunks
        db.add_documents(chunks)
        db.persist() # Save the changes
        
        if status_callback: status_callback("✅ Training complete! Your bot is now augmented with the new knowledge.")
    else:
        if status_callback: status_callback("No new content to add from uploaded files.")

# This part is for manual command-line testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a user-specific knowledge base.")
    print("This script is primarily intended to be called from the UI.")
