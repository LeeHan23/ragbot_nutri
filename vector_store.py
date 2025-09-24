import os
from functools import lru_cache # <-- 1. IMPORT THE CACHING UTILITY
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# --- UNIFIED PATH CONFIGURATION ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_PATH = os.path.join(APP_DIR, "data")
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", LOCAL_DATA_PATH)

# --- Constants ---
USER_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "chroma_db") 
BASE_DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "vectorstore_base")
USER_COLLECTION_NAME = "user_knowledge"
BASE_COLLECTION_NAME = "base_knowledge"

# --- Embedding Function ---
try:
    embedding_function = OpenAIEmbeddings()
except Exception as e:
    print(f"Error initializing LangChain OpenAI Embeddings: {e}")
    embedding_function = None

# --- Retriever Functions ---

@lru_cache(maxsize=1) # <-- 2. CACHE THE FOUNDATIONAL DATABASE
def get_base_retriever():
    """
    Loads and returns a retriever for the foundational knowledge base.
    This function is cached to prevent reloading the large database from disk on every call.
    """
    if not embedding_function:
        raise ValueError("Embedding function not initialized.")
    
    if not os.path.exists(BASE_DB_PATH):
        raise FileNotFoundError(f"Foundational database not found at {BASE_DB_PATH}. Please run build_base_db.py.")

    print("Loading foundational knowledge base into memory cache...")
    vector_store = Chroma(
        persist_directory=BASE_DB_PATH,
        embedding_function=embedding_function,
        collection_name=BASE_COLLECTION_NAME
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})

@lru_cache(maxsize=32) # <-- 3. CACHE USER-SPECIFIC DATABASES
def get_user_retriever(user_id: str):
    """
    Loads and returns a retriever for a user's specific knowledge base.
    Returns None if the user does not have a custom database.
    This function is cached to prevent reloading from disk for active users.
    """
    if not user_id: return None
    if not embedding_function:
        raise ValueError("Embedding function not initialized.")

    user_specific_db_path = os.path.join(USER_DB_PATH, user_id)
    
    if not os.path.exists(user_specific_db_path):
        return None # No custom database for this user

    print(f"Loading custom knowledge base for user '{user_id}' into memory cache...")
    vector_store = Chroma(
        persist_directory=user_specific_db_path,
        embedding_function=embedding_function,
        collection_name=USER_COLLECTION_NAME
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})

