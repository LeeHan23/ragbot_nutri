import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "data", "knowledge")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "nutritionist_knowledge"

# --- Embedding Function ---
# Initialize the embedding model from OpenAI. This will be used to convert
# text documents into numerical vectors for similarity searching.
# Ensure your OPENAI_API_KEY is set in your environment.
try:
    embedding_function = OpenAIEmbeddings()
except Exception as e:
    print(f"Error initializing OpenAI Embeddings: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    embedding_function = None

# --- Retriever Function ---
def get_retriever():
    """
    Initializes and returns a Chroma vector store retriever.

    This function sets up a persistent ChromaDB client and configures it
    as a LangChain retriever to find the top 3 most relevant documents
    for a given query.

    Returns:
        A LangChain retriever object.
    """
    if not embedding_function:
        raise ValueError("Embedding function is not initialized. Cannot create retriever.")

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    # Configure the retriever to fetch the top_k most similar documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print(f"Retriever for collection '{COLLECTION_NAME}' initialized.")
    return retriever

# --- Re-indexing Function ---
def reindex_knowledge_base():
    """
    Clears the existing vector store and re-indexes all documents from the knowledge base directory.

    This function performs the following steps:
    1. Deletes the old ChromaDB database directory to ensure a fresh start.
    2. Loads all .txt files from the `data/knowledge` directory.
    3. Splits the loaded documents into smaller, manageable chunks.
    4. Creates new vector embeddings for these chunks.
    5. Persists the new embeddings to the ChromaDB database.

    Returns:
        tuple: A tuple containing the number of documents indexed and the number of chunks created.
    """
    if not embedding_function:
        raise ValueError("Embedding function is not initialized. Cannot re-index.")

    # 1. Clear existing database
    if os.path.exists(CHROMA_PATH):
        print(f"Clearing existing vector store at: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH)
    print("Vector store cleared.")

    # 2. Load documents from the knowledge directory
    if not os.path.exists(KNOWLEDGE_PATH) or not os.listdir(KNOWLEDGE_PATH):
        print("Knowledge base directory is empty or does not exist. No documents to index.")
        return 0, 0

    # Use DirectoryLoader to load all .txt files.
    loader = DirectoryLoader(KNOWLEDGE_PATH, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    if not documents:
        print("No .txt documents found to index.")
        return 0, 0
    print(f"Loaded {len(documents)} document(s) from {KNOWLEDGE_PATH}.")

    # 3. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # 4. Create and persist the new vector store
    print("Creating new vector store and embeddings. This may take a moment...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )
    print("Re-indexing complete. Vector store is up to date.")

    return len(documents), len(chunks)

