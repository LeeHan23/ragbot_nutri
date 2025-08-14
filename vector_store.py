import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# --- Constants ---
# Correctly point to the persistent disk path provided by the environment
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
BASE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore_base")
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
    Initializes and returns a vector store retriever for the foundational
    knowledge base. All users share this single data source.
    
    Returns:
        A tuple containing (retriever, knowledge_source_name).
    """
    if not embedding_function:
        raise ValueError("Embedding function is not initialized. Cannot create retriever.")
    
    print(f"Loading foundational knowledge base for user '{user_id}'.")
    
    if not os.path.exists(BASE_DB_PATH):
        raise FileNotFoundError(f"The foundational database was not found at {BASE_DB_PATH}. Please run `build_base_db.py` first.")

    # Always load the base vector store
    vector_store = Chroma(
        persist_directory=BASE_DB_PATH,
        embedding_function=embedding_function,
        collection_name=BASE_COLLECTION_NAME
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    print(f"Retriever initialized for user '{user_id}' from 'Foundational' source.")
    return retriever, "Foundational"
