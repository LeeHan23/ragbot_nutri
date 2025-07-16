import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the directory for user-specific, augmented knowledge bases
USER_DB_PATH = os.path.join(BASE_DIR, "chroma_db") 
# Path to the directory for the foundational PDF knowledge base
BASE_DB_PATH = os.path.join(BASE_DIR, "vectorstore_base")
# The collection name is now consistent because we copy the base DB
COLLECTION_NAME = "base_knowledge" 

# --- Embedding Function ---
try:
    embedding_function = OpenAIEmbeddings()
except Exception as e:
    print(f"Error initializing LangChain OpenAI Embeddings: {e}")
    embedding_function = None

# --- Retriever Function ---
def get_retriever(user_id: str):
    """
    Initializes and returns a standard vector store retriever.
    It prioritizes the user-specific database if it exists, otherwise
    it falls back to the foundational base database.
    """
    if not embedding_function:
        raise ValueError("Embedding function is not initialized. Cannot create retriever.")
    
    user_specific_db_path = os.path.join(USER_DB_PATH, user_id)
    
    # Determine which database path to use
    if os.path.exists(user_specific_db_path):
        print(f"Loading custom knowledge base for user '{user_id}'.")
        persistent_directory = user_specific_db_path
    else:
        print(f"No custom knowledge for user '{user_id}'. Falling back to foundational knowledge base.")
        persistent_directory = BASE_DB_PATH

    if not os.path.exists(persistent_directory):
        raise FileNotFoundError(
            f"Required database not found at {persistent_directory}. "
            "Please ensure the base database is built by running 'build_base_db.py'."
        )

    vector_store = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME # Use the consistent collection name
    )
    
    # Use a standard, more reliable retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Increased k to retrieve more context
    
    print(f"Standard retriever initialized for user '{user_id}'.")
    return retriever
