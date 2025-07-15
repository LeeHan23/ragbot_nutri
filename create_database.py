import os
import shutil
import time
import argparse
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "data", "users") 
CHROMA_BASE_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "user_knowledge"

def build_database(user_id: str, status_callback=None, wipe_existing=False):
    """
    Builds or updates the ChromaDB vector store for a specific user.
    
    Args:
        user_id (str): The unique identifier for the user.
        status_callback (function, optional): A function to call with status updates.
        wipe_existing (bool): If True, deletes the old database before building.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    knowledge_path = os.path.join(KNOWLEDGE_BASE_DIR, user_id, "knowledge")
    chroma_path = os.path.join(CHROMA_BASE_PATH, user_id)
    
    if status_callback: status_callback(f"--- Starting database build for user: {user_id} ---")

    if wipe_existing and os.path.exists(chroma_path):
        if status_callback: status_callback("Clearing existing knowledge base...")
        shutil.rmtree(chroma_path)
    
    if status_callback: status_callback("Loading documents...")
    all_documents = []
    if not os.path.exists(knowledge_path) or not os.listdir(knowledge_path):
        if status_callback: status_callback("Knowledge folder is empty. Nothing to train on.")
        return

    for filename in os.listdir(knowledge_path):
        file_path = os.path.join(knowledge_path, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                all_documents.extend(loader.load())
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                all_documents.extend(loader.load())
        except Exception as e:
            if status_callback: status_callback(f"Error loading {filename}: {e}")
            continue

    if not all_documents:
        if status_callback: status_callback("No processable documents found.")
        return
    if status_callback: status_callback(f"Loaded content from documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)
    if status_callback: status_callback(f"Split documents into {len(chunks)} chunks.")

    if chunks:
        if status_callback: status_callback("Embedding documents and updating database... This may take several minutes.")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=6, chunk_size=500)
        
        # Initialize a persistent client
        db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=chroma_path
        )
        
        # Add the new documents to the existing database
        db.add_documents(chunks)
        db.persist()
        
        if status_callback: status_callback("\nTraining complete!")
    else:
        if status_callback: status_callback("No content to process.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a user-specific knowledge base.")
    parser.add_argument("--user_id", type=str, required=True, help="The unique identifier for the user.")
    parser.add_argument("--wipe", action='store_true', help="Wipe the existing database before building.")
    args = parser.parse_args()
    
    build_database(args.user_id, status_callback=print, wipe_existing=args.wipe)
