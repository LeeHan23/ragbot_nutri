import os
from langchain.prompts import PromptTemplate

# Import functions from other modules
from vector_store import get_retriever
from llm import get_llm_response

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")

# --- Prompt Template Definition ---
# This template structures the input for the language model, ensuring it has all
# the necessary context (behavior, promos, retrieved info) to generate a relevant response.
RAG_PROMPT_TEMPLATE = """
You are "Eva," a friendly and expert customer service assistant for a nutritionist company.
Your personality and response style are strictly defined by the detailed instructions below. You must follow them precisely.

**BEHAVIOR AND PERSONA INSTRUCTIONS:**
{behavior_instructions}

**ACTIVE PROMOTIONS/DISCOUNTS:**
{promo_text}

**CONTEXTUAL KNOWLEDGE BASE:**
Use the following retrieved context to answer the user's question. This is your only source of truth for facts.
{retrieved_docs}

**CUSTOMER INTERACTION:**
Customer Question: "{user_message}"

Based on all the above, provide a comprehensive, natural, and helpful response that strictly follows your persona instructions.
"""

# Create a LangChain PromptTemplate object
rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["behavior_instructions", "promo_text", "retrieved_docs", "user_message"],
)


# --- Helper Function to Load Dynamic Content ---
def _load_latest_text_file(directory: str, default_text: str = "Not available.") -> str:
    """
    Loads content from the most recently created/modified text file in a directory.

    Args:
        directory (str): The path to the directory containing text files.
        default_text (str): The text to return if the directory is empty or doesn't exist.

    Returns:
        str: The content of the latest file, or the default text.
    """
    if not os.path.exists(directory) or not os.listdir(directory):
        return default_text
    
    try:
        # Get all files and sort them by modification time (most recent first)
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
        files.sort(key=os.path.getmtime, reverse=True)
        
        latest_file = files[0]
        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading latest file from {directory}: {e}")
        return default_text


# --- Main RAG Pipeline Function ---
async def get_rag_response(user_message: str) -> str:
    """
    Generates a response using the full RAG pipeline.

    This function performs the following steps:
    1. Retrieves relevant documents from the vector store.
    2. Loads the latest behavioral instructions and promotional text.
    3. Formats the complete prompt.
    4. Calls the language model to get the final response.

    Args:
        user_message (str): The customer's question.

    Returns:
        str: The generated response from the chatbot.
    """
    print("--- Starting RAG Pipeline ---")

    # 1. Retrieve relevant documents from the vector store
    try:
        retriever = get_retriever()
        # The retriever finds documents in the vector store that are semantically
        # similar to the user's message.
        retrieved_docs = retriever.invoke(user_message)
        
        # Format the retrieved documents into a single string
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(f"Retrieved {len(retrieved_docs)} documents from vector store.")
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        retrieved_context = "Could not retrieve context information."

    # 2. Load dynamic content (instructions and promos)
    behavior_instructions = _load_latest_text_file(INSTRUCTIONS_PATH, "Be friendly and professional.")
    promo_text = _load_latest_text_file(PROMOS_PATH, "No active promotions at the moment.")
    
    print(f"Loaded Instructions: {' '.join(behavior_instructions.split()[:10])}...")
    print(f"Loaded Promotions: {' '.join(promo_text.split()[:10])}...")

    # 3. Format the final prompt
    final_prompt = rag_prompt.format(
        behavior_instructions=behavior_instructions,
        promo_text=promo_text,
        retrieved_docs=retrieved_context,
        user_message=user_message
    )
    
    # print(f"--- Final Prompt ---\n{final_prompt}\n--------------------")

    # 4. Get the response from the language model
    try:
        final_response = await get_llm_response(final_prompt)
        print("--- RAG Pipeline Finished ---")
        return final_response
    except Exception as e:
        print(f"Error getting response from LLM: {e}")
        # Provide a professional fallback response
        return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again in a moment."
