import os
from typing import List, Tuple, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Import functions from other modules
from vector_store import get_retriever
from llm import get_llm

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")

# --- Define the desired output structure ---
class BotResponse(BaseModel):
    answer: str = Field(description="The full, multi-line response to send to the user, with newlines separating each message bubble.")
    new_summary: str = Field(description="An updated, concise summary of the user's intents based on the current conversation.")

# --- Prompt Templates ---
SYSTEM_PROMPT_TEMPLATE = """
You are "Eva," an expert wellness assistant. Your goal is to have a personalized, stateful conversation.

**USER METADATA (Your long-term memory of the user):**
- Visit Count: {visit_count}
- Summary of Past Interests: {intent_summary}

**PERSONA AND BEHAVIOR INSTRUCTIONS:**
Use the User Metadata to personalize the conversation. For example, greet returning users differently.
{behavior_instructions}

**ACTIVE PROMOTIONS/DISCOUNTS:**
{promo_text}

**CONTEXTUAL KNOWLEDGE BASE (Your source of truth):**
{context}

**CURRENT CONVERSATION (Your short-term memory):**
{chat_history}

**USER'S LATEST MESSAGE:**
{question}

Based on ALL of the above, generate your response and an updated summary of the user's intents.
Your final output must be a JSON object matching the required format.
"""

# The condense question prompt remains the same
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
)

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
def get_rag_conversation_chain():
    """Initializes the ConversationalRetrievalChain."""
    llm = get_llm()
    retriever = get_retriever()
    
    # The memory object that the chain will use to store the conversation.
    # The output_key='answer' is important for the chain's internal logic.
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        # The final prompt is now set dynamically per request, so we don't set it here.
        return_source_documents=False,
        return_generated_question=False,
    )
    print("ConversationalRetrievalChain initialized.")
    return conversation_chain

# Create a single, reusable chain instance to avoid re-initializing on every message.
rag_chain = get_rag_conversation_chain()

async def get_contextual_response(user_message: str, user_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Gets an intent-aware response using the user's entire session data.
    """
    try:
        print("--- Invoking Intent-Aware Conversational Chain ---")
        
        # Load dynamic content for each call to ensure it's always fresh
        behavior_instructions = _load_latest_text_file(INSTRUCTIONS_PATH, "Be friendly.")
        promo_text = _load_latest_text_file(PROMOS_PATH, "No active promotions.")

        # Set up the JSON parser
        parser = JsonOutputParser(pydantic_object=BotResponse)

        # Create the full, dynamic prompt template for the final combination step
        final_prompt = PromptTemplate(
            template=SYSTEM_PROMPT_TEMPLATE,
            input_variables=["chat_history", "question", "context"],
            partial_variables={
                "visit_count": user_data.get("visit_count", 1),
                "intent_summary": user_data.get("intent_summary", "No interactions yet."),
                "behavior_instructions": behavior_instructions,
                "promo_text": promo_text,
            }
        )
        
        # Temporarily override the chain's prompt for this specific call.
        # This ensures every user gets the most up-to-date persona and their own metadata.
        rag_chain.combine_docs_chain.llm_chain.prompt = final_prompt

        # Invoke the chain with the user's question and their specific chat history
        result = await rag_chain.ainvoke({
            "question": user_message,
            "chat_history": user_data.get("chat_history", [])
        })

        # The raw output from the LLM is a JSON string inside the 'answer' key.
        # We need to parse this string to get our structured BotResponse object.
        parsed_result = parser.parse(result["answer"])

        return parsed_result

    except Exception as e:
        print(f"Error invoking conversational chain: {e}")
        # Provide a safe fallback response in case of parsing or other errors
        return {
            "answer": "I'm sorry, I encountered an issue. Could you please rephrase that?",
            "new_summary": user_data.get("intent_summary", "Error in conversation.")
        }
