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
BASE_DB_PATH = os.path.join(BASE_DIR, "vectorstore_base")
USER_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
# The collection name MUST be consistent because we are copying the base DB
COLLECTION_NAME = "base_knowledge" 

def build_user_database(user_id: str, uploaded_docx_files: list, status_callback=None):
    """
    Builds a user-specific knowledge base by copying the foundational DB
    and augmenting it with the user's uploaded documents. This is the most
    efficient method.
    
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

    # 1. Clear any existing custom database for this user
    if os.path.exists(user_db_path):
        if status_callback: status_callback("Clearing old custom knowledge base...")
        shutil.rmtree(user_db_path)

    # 2. Copy the foundational database to create the user's personal starting point
    if status_callback: status_callback("Copying foundational knowledge...")
    if not os.path.exists(BASE_DB_PATH):
        if status_callback: status_callback("FATAL ERROR: Foundational knowledge base not found.")
        return
    shutil.copytree(BASE_DB_PATH, user_db_path)

    # 3. Load the user's newly uploaded documents
    if not uploaded_docx_files:
        if status_callback: status_callback("No new documents uploaded. User DB is a copy of the base knowledge.")
        return
        
    if status_callback: status_callback("Loading new user documents...")
    all_new_docs = []
    for file_obj in uploaded_docx_files:
        try:
            temp_dir = os.path.join(BASE_DIR, "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, file_obj.name)
            with open(temp_path, "wb") as f:
                f.write(file_obj.getvalue())
            
            loader = Docx2txtLoader(temp_path)
            all_new_docs.extend(loader.load())
            
            os.remove(temp_path)
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            if status_callback: status_callback(f"Error loading uploaded file {file_obj.name}: {e}")
            continue

    if not all_new_docs:
        if status_callback: status_callback("No new content found in documents.")
        return

    # 4. Split the new documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_new_docs)
    if status_callback: status_callback(f"Split new documents into {len(chunks)} chunks.")

    # 5. Add the new chunks to the user's copied database using the native client
    if chunks:
        try:
            if status_callback: status_callback("Connecting to user database...")
            client = chromadb.PersistentClient(path=user_db_path)
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found.")
            
            chroma_ef = chroma_embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key, model_name="text-embedding-ada-002"
            )

            collection = client.get_collection(name=COLLECTION_NAME, embedding_function=chroma_ef)
            
            if status_callback: status_callback("Adding new knowledge to the database...")
            
            existing_ids_count = collection.count()
            ids = [f"user_chunk_{existing_ids_count + i}" for i in range(len(chunks))]
            documents_content = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]

            collection.add(documents=documents_content, metadatas=metadatas, ids=ids)
            
            if status_callback: status_callback("✅ Knowledge base successfully updated!")

        except Exception as e:
            if status_callback: status_callback(f"An error occurred during database update: {e}")
            raise e

if __name__ == "__main__":
    print("This script is primarily intended to be called from the UI.")
