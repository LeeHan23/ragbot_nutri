import os
import shutil
import chromadb
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

# --- Constants ---
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db")
COLLECTION_NAME = "user_knowledge"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_UPLOADS_DIR = os.path.join(BASE_DIR, "temp_uploads")
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")


# --- Prompt Loading Functions ---
# (This section remains the same)
def _get_latest_file_content(directory: str) -> str:
    try:
        if not os.path.exists(directory):
            return ""
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
        if not files:
            return ""
        latest_file = max(files, key=os.path.getmtime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading from {directory}: {e}")
        return ""

def get_prompts(user_id: str = None) -> tuple[str, str]:
    print("Loading persona instructions and latest promotions...")
    instructions = ""
    if user_id:
        user_instructions_dir = os.path.join(INSTRUCTIONS_PATH, str(user_id))
        if os.path.exists(user_instructions_dir):
            instructions = _get_latest_file_content(user_instructions_dir)
    if not instructions:
        instructions = _get_latest_file_content(INSTRUCTIONS_PATH)
    promotions = _get_latest_file_content(PROMOS_PATH)
    if not instructions:
        instructions = "You are a helpful general assistant."
    if not promotions:
        promotions = "There are no special promotions at this time."
    full_instructions = f"{instructions}\n\n[LATEST PROMOTIONS & OFFERS]\n{promotions}"
    return full_instructions, ""


# --- Knowledge Base Building Functions ---

def build_user_database(user_id: str, uploaded_docx_files: list, status_callback=None):
    """
    Builds a user-specific knowledge base from uploaded .docx files.
    """
    if not user_id:
        if status_callback: status_callback("Error: A User ID must be provided.")
        return

    user_db_path = os.path.join(USER_DB_PATH, user_id)
    
    if status_callback: status_callback(f"--- Starting new custom build for user: {user_id} ---")

    # --- FIX: Remove the chromadb.Client().reset() call ---
    # The reset() function is disabled by default in newer versions and is not the
    # correct way to handle this. The file lock issue is transient and simply
    # deleting the old directory is the intended workflow.

    # 1. Clear any existing custom database for this user to ensure a fresh start.
    if os.path.exists(user_db_path):
        if status_callback: status_callback("Clearing old custom knowledge base...")
        shutil.rmtree(user_db_path)

    # 2. Load the user's newly uploaded documents
    if status_callback: status_callback("Loading user documents...")
    all_docs = []
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
    for file_obj in uploaded_docx_files:
        try:
            temp_path = os.path.join(TEMP_UPLOADS_DIR, file_obj.name)
            with open(temp_path, "wb") as f:
                f.write(file_obj.getvalue())
            loader = Docx2txtLoader(temp_path)
            all_docs.extend(loader.load())
            os.remove(temp_path)
        except Exception as e:
            if status_callback: status_callback(f"Error loading uploaded file {file_obj.name}: {e}")
            continue
    
    if os.path.exists(TEMP_UPLOADS_DIR) and not os.listdir(TEMP_UPLOADS_DIR):
        os.rmdir(TEMP_UPLOADS_DIR)

    if not all_docs:
        if status_callback: status_callback("No documents found to process.")
        return

    # 3. Split all documents together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    if status_callback: status_callback(f"Split documents into {len(chunks)} chunks.")

    # 4. Create the new user-specific database
    if chunks:
        if status_callback: status_callback("Embedding documents...")
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=6, chunk_size=500)
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=user_db_path,
            collection_name=COLLECTION_NAME
        )
        
        if status_callback: status_callback("✅ Training complete! Your custom knowledge base is ready.")
    else:
        if status_callback: status_callback("No content found in documents to train on.")

if __name__ == "__main__":
    print("This script is primarily intended to be called from the UI.")
