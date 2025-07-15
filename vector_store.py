import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_BASE_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "user_knowledge"

# --- Embedding Function ---
try:
    embedding_function = OpenAIEmbeddings()
except Exception as e:
    print(f"Error initializing LangChain OpenAI Embeddings: {e}")
    embedding_function = None

# --- Retriever Function ---
def get_retriever(user_id: str):
    """
    Initializes and returns a Chroma vector store retriever for a specific user.
    """
    if not embedding_function:
        raise ValueError("Embedding function is not initialized. Cannot create retriever.")
    
    # Define the path to the specific user's database
    persistent_directory = os.path.join(CHROMA_BASE_PATH, user_id)

    if not os.path.exists(persistent_directory):
        # Return a dummy retriever or handle this case gracefully if no DB exists
        print(f"Warning: No database found for user '{user_id}'. The bot will have no specific knowledge.")
        # You could create an empty retriever here if needed, but for now we'll raise an error.
        raise FileNotFoundError(f"Database for user '{user_id}' not found.")

    vector_store = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print(f"Retriever initialized for user '{user_id}'.")
    return retriever
