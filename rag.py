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
# The chain will feed the chat history and the user's new question into this.
SYSTEM_PROMPT_TEMPLATE = """
You are Eva, a warm and intelligent virtual wellness guide for NutriLife Wellness.
Your job is to guide users as if you're having a friendly and insightful conversation.
You specialize in health and nutrition services and always prioritize making users feel heard, safe, and respected.

Key Behaviors:
- Never sound robotic. Always respond like a caring human.
- Use emojis naturally to soften the tone (😊, 💡, 💬, 💰, ✨).
- Break responses into short, 1-3 sentence chunks.
- Ask follow-up questions instead of listing everything.
- Use information from the documents to respond accurately and confidently.
- If uncertain, respond with honesty and offer to connect the user with a dietitian.

ALWAYS end your messages with a friendly question to keep the conversation flowing.
If a promo is valid and matches the situation, mention it naturally and encourage action.
"""

USER_PROMPT_TEMPLATE = """
The user has asked: {query}

Context from documents:
{retrieved_docs}

Your reply should:
- Sound human, caring, and consultative
- Be broken into short, warm replies
- Include helpful follow-up questions
- Mention promotions only if relevant
- Use emojis and bold where needed

Your response:
"""

# This is a separate, simpler prompt used by the chain internally to condense
# the chat history and the new question into a single, standalone question.
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
)


# --- Helper Function to Load Dynamic Content ---
def _load_latest_text_file(directory: str, default_text: str = "Not available.") -> str:
    """Loads content from the most recently created/modified text file in a directory."""
    # (This function remains unchanged)
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
    
    # Load dynamic content once when creating the chain
    behavior_instructions = _load_latest_text_file(INSTRUCTIONS_PATH, "Be friendly and professional.")
    promo_text = _load_latest_text_file(PROMOS_PATH, "No active promotions at the moment.")

    # Inject the dynamic content into the system prompt
    custom_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        behavior_instructions=behavior_instructions,
        promo_text=promo_text,
        context="{context}" # The {context} placeholder must be kept for the chain
    )
    
    # The memory object that the chain will use to store the conversation
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
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(custom_system_prompt)}
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
        # The chain automatically uses its internal memory, but we can also
        # pass history if we were managing it externally. For simplicity,
        # we'll let the chain's internal memory handle it.
        # The 'question' key is what the chain expects.
        result = await rag_chain.ainvoke({"question": user_message, "chat_history": chat_history})
        return result["answer"]
    except Exception as e:
        print(f"Error invoking conversational chain: {e}")
        return "I'm sorry, I encountered an issue while trying to understand our conversation. Could you please rephrase that?"
