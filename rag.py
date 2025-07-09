import os
from typing import List
from langchain.prompts import PromptTemplate

# Import functions from other modules
from vector_store import get_retriever
from llm import get_llm_conversation # <-- IMPORT THE NEW FUNCTION

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")

# --- Prompt Template Definition ---
# This new template explicitly tells the AI to break its response into multiple,
# short messages to simulate a natural conversation.
RAG_PROMPT_TEMPLATE = """
You are "Eva," an expert wellness assistant. Your task is to have a natural, helpful conversation with the user.

**CRITICAL INSTRUCTIONS:**
1.  You MUST break your response down into a sequence of short, individual messages.
2.  Your final output MUST be a JSON object containing a list of these messages, as per the format instructions.
3.  Follow your detailed persona instructions precisely.

**PERSONA AND BEHAVIOR INSTRUCTIONS:**
{behavior_instructions}

**ACTIVE PROMOTIONS/DISCOUNTS:**
{promo_text}

**CONTEXTUAL KNOWLEDGE BASE (Your source of truth):**
{retrieved_docs}

**INTERACTION:**
Customer Question: "{user_message}"

Now, based on all the above, generate the sequence of messages you will send to the user.
"""

# Create a LangChain PromptTemplate object
rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["behavior_instructions", "promo_text", "retrieved_docs", "user_message"],
)


# --- Helper Function to Load Dynamic Content ---
def _load_latest_text_file(directory: str, default_text: str = "Not available.") -> str:
    """Loads content from the most recently created/modified text file in a directory."""
    if not os.path.exists(directory) or not os.listdir(directory):
        return default_text
    
    try:
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
        files.sort(key=os.path.getmtime, reverse=True)
        
        latest_file = files[0]
        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading latest file from {directory}: {e}")
        return default_text


# --- Main RAG Pipeline Function ---
async def get_rag_response(user_message: str) -> List[str]:
    """
    Generates a sequence of response messages using the RAG pipeline.

    Returns:
        List[str]: A list of messages to be sent to the user.
    """
    print("--- Starting RAG Pipeline ---")

    try:
        retriever = get_retriever()
        retrieved_docs = retriever.invoke(user_message)
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(f"Retrieved {len(retrieved_docs)} documents.")
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        retrieved_context = "Could not retrieve context information."

    behavior_instructions = _load_latest_text_file(INSTRUCTIONS_PATH, "Be friendly and professional.")
    promo_text = _load_latest_text_file(PROMOS_PATH, "No active promotions at the moment.")
    
    final_prompt = rag_prompt.format(
        behavior_instructions=behavior_instructions,
        promo_text=promo_text,
        retrieved_docs=retrieved_context,
        user_message=user_message
    )
    
    try:
        # Call the new function that returns a list of messages
        message_list = await get_llm_conversation(final_prompt)
        print("--- RAG Pipeline Finished ---")
        return message_list
    except Exception as e:
        print(f"Error getting response from LLM: {e}")
        return ["I'm sorry, I'm having a little trouble thinking of a response right now. Please try asking in a different way."]
