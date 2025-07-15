import os
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Import functions from other modules
from vector_store import get_retriever
from llm import get_llm

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")

# --- Helper Function (Unchanged) ---
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

# --- Main RAG Pipeline ---
async def get_contextual_response(user_message: str, user_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Gets a contextual response and the source documents used to generate it.
    """
    try:
        print(f"--- Invoking LCEL Chain for user: {user_id} ---")
        
        llm = get_llm()
        retriever = get_retriever(user_id)

        # 1. Prompt to rephrase a follow-up question
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # 2. Main prompt to answer the question
        system_prompt = """
        You are "Eva," an expert wellness assistant. Your goal is to have a personalized, stateful conversation.
        Your personality and response style are strictly defined by the detailed instructions below.
        You MUST have a natural, back-and-forth conversation, breaking your response into short, individual messages using newlines (\\n).
        Use the chat history to understand the context of the conversation and the user metadata to personalize your greeting and responses.

        **USER METADATA (Your long-term memory of the user):**
        - Visit Count: {visit_count}
        - Summary of Past Interests: {intent_summary}

        **PERSONA AND BEHAVIOR INSTRUCTIONS:**
        {behavior_instructions}

        **ACTIVE PROMOTIONS/DISCOUNTS:**
        {promo_text}

        **CONTEXTUAL KNOWLEDGE BASE (Your source of truth):**
        {context}

        Based on ALL of the above, provide a comprehensive, natural, and helpful response.
        """
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Invoke the chain with all necessary inputs
        result = await rag_chain.ainvoke({
            "input": user_message,
            "chat_history": user_data.get("chat_history", []),
            "visit_count": user_data.get("visit_count", 1),
            "intent_summary": user_data.get("intent_summary", "No interactions yet."),
            "behavior_instructions": _load_latest_text_file(INSTRUCTIONS_PATH, "Be friendly."),
            "promo_text": _load_latest_text_file(PROMOS_PATH, "No active promotions."),
        })
        
        return {
            "answer": result.get("answer", "I'm not sure how to respond to that."),
            "sources": result.get("context", [])
        }

    except FileNotFoundError:
        return {
            "answer": "It looks like I don't have a knowledge base for you yet. Please upload some documents to get started!",
            "sources": []
        }
    except Exception as e:
        print(f"Error invoking conversational chain: {e}")
        return {
            "answer": "I'm sorry, I encountered an issue. Could you please rephrase?",
            "sources": []
        }
