import os
from typing import List, Tuple
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Import functions from other modules
from vector_store import get_retriever
from llm import get_llm

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")

# --- Prompt Templates ---

# This template combines the persona, chat history, and retrieved context.
# It has multiple input variables that will be populated.
SYSTEM_PROMPT_TEMPLATE = """
You are "Eva," a friendly and expert wellness assistant from NutriLife Wellness.
Your personality and response style are strictly defined by the detailed instructions below.
You must have a natural, back-and-forth conversation, breaking your response into short, individual messages.
Use the chat history to understand the context of the conversation.

**PERSONA AND BEHAVIOR INSTRUCTIONS:**
{behavior_instructions}

**ACTIVE PROMOTIONS/DISCOUNTS:**
{promo_text}

**CONTEXTUAL KNOWLEDGE BASE (Your source of truth):**
{context}

Now, answer the user's question based on the chat history and the provided knowledge.
"""

# This is a separate, simpler prompt used by the chain internally to condense
# the chat history and the new question into a single, standalone question.
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
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
def get_rag_conversation_chain():
    """
    Initializes and returns the complete ConversationalRetrievalChain.
    """
    llm = get_llm()
    retriever = get_retriever()
    
    # Load dynamic content
    behavior_instructions = _load_latest_text_file(INSTRUCTIONS_PATH, "Be friendly and professional.")
    promo_text = _load_latest_text_file(PROMOS_PATH, "No active promotions at the moment.")

    # Define the final prompt template with all its expected input variables
    final_docs_prompt = PromptTemplate(
        template=SYSTEM_PROMPT_TEMPLATE,
        input_variables=["context", "behavior_instructions", "promo_text"]
    )

    # Use the .partial() method to safely pre-fill parts of the prompt.
    # This is the correct way to handle dynamic system prompts in a chain.
    # It correctly preserves the remaining 'context' variable for the chain to use.
    final_docs_prompt = final_docs_prompt.partial(
        behavior_instructions=behavior_instructions,
        promo_text=promo_text
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the final conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": final_docs_prompt}
    )
    
    print("ConversationalRetrievalChain initialized.")
    return conversation_chain

# --- We create a single, reusable chain instance ---
# This avoids re-initializing the chain on every single message.
rag_chain = get_rag_conversation_chain()

async def get_contextual_response(user_message: str, chat_history: List[Tuple[str, str]]) -> str:
    """
    Gets a contextual response using the pre-initialized chain.

    Args:
        user_message (str): The new message from the user.
        chat_history (List[Tuple[str, str]]): The past conversation history.

    Returns:
        str: The AI's generated response.
    """
    try:
        print("--- Invoking Conversational Chain ---")
        result = await rag_chain.ainvoke({"question": user_message, "chat_history": chat_history})
        return result["answer"]
    except Exception as e:
        print(f"Error invoking conversational chain: {e}")
        return "I'm sorry, I encountered an issue while trying to understand our conversation. Could you please rephrase that?"
