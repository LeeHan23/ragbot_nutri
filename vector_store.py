import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# --- Constants ---
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
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

# --- Retriever Function ---
def get_retriever(user_id: str):
    """
    Initializes and returns a vector store retriever.
    It prioritizes the user-specific database if it exists, otherwise
    it falls back to the foundational base database.
    """
    if not embedding_function:
        raise ValueError("Embedding function is not initialized.")
    
    user_specific_db_path = os.path.join(USER_DB_PATH, user_id)
    knowledge_source = ""
    
    # --- IMPROVEMENT: Restore the logic to check for a user-specific DB ---
    if os.path.exists(user_specific_db_path):
        print(f"Loading custom knowledge base for user '{user_id}'.")
        persistent_directory = user_specific_db_path
        collection_name = USER_COLLECTION_NAME
        knowledge_source = "Custom"
    else:
        print(f"No custom knowledge for user '{user_id}'. Falling back to foundational knowledge base.")
        persistent_directory = BASE_DB_PATH
        collection_name = BASE_COLLECTION_NAME
        knowledge_source = "Foundational"

    if not os.path.exists(persistent_directory):
        raise FileNotFoundError(f"Required database not found at {persistent_directory}.")

    vector_store = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_function,
        collection_name=collection_name
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    print(f"Retriever initialized for user '{user_id}' from '{knowledge_source}' source.")
    return retriever, knowledge_source
