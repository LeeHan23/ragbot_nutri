import os
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Import functions from other modules
from vector_store import get_retriever
from llm import get_llm
from knowledge_manager import get_prompts # Use the helper from knowledge_manager

# --- RAG Prompt Template ---
# This is the core instruction set for the AI model.
RAG_PROMPT_TEMPLATE = """
You are "Eva," an expert wellness assistant. Your goal is to have a personalized, stateful conversation.
Your personality and response style are strictly defined by the detailed instructions in the [PERSONA INSTRUCTIONS] section.
You MUST base your answer on the information provided in the [CONTEXTUAL KNOWLEDGE BASE]. If the context is empty, inform the user you don't have specific information on that topic and guide them to a professional.

[PERSONA INSTRUCTIONS]
{persona_instructions}

[CONTEXTUAL KNOWLEDGE BASE (Your source of truth)]
{context}

[CURRENT CONVERSATION (Your short-term memory)]
{chat_history}

[USER'S LATEST MESSAGE]
Question: {question}

[YOUR RESPONSE]
Answer:
"""

def format_chat_history(chat_history: list) -> str:
    """Formats the chat history into a readable string for the prompt."""
    if not chat_history:
        return "No conversation history yet."
    
    # The history from Streamlit is a list of dicts: {'role': 'user'/'assistant', 'content': '...'}
    langchain_messages = []
    for message in chat_history:
        if message['role'] == 'user':
            langchain_messages.append(HumanMessage(content=message['content']))
        elif message['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=message['content']))
    
    # Format for display in the prompt
    formatted_history = []
    for msg in langchain_messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted_history.append(f"{role}: {msg.content}")
        
    return "\n".join(formatted_history)

async def get_contextual_response(user_question: str, chat_history: list, user_id: str) -> Dict[str, Any]:
    """
    This is the main function that generates the bot's response.
    It uses a direct, manually constructed RAG approach for maximum reliability.
    """
    try:
        print(f"--- Invoking Direct RAG Chain for user: {user_id} ---")
        
        # 1. Get the appropriate retriever (custom or base)
        retriever = get_retriever(user_id)
        
        # 2. Retrieve relevant documents from the knowledge base
        retrieved_docs = retriever.invoke(user_question)
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 3. Load the persona and promotion instructions
        instructions, promos = get_prompts()
        
        # 4. Format the chat history
        formatted_history = format_chat_history(chat_history)
        
        # 5. Build the final prompt
        prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        final_prompt = prompt_template.format(
            persona_instructions=instructions,
            context=context,
            chat_history=formatted_history,
            question=user_question
        )
        
        # 6. Call the AI model
        llm = get_llm()
        response = await llm.ainvoke(final_prompt)
        
        # 7. Return the response and sources
        return {
            "answer": response.content,
            "sources": retrieved_docs
        }

    except FileNotFoundError:
        return {
            "answer": "It looks like I don't have a knowledge base for you yet. Please upload some documents to get started!",
            "sources": []
        }
    except Exception as e:
        print(f"ERROR in get_contextual_response: {e}")
        return {
            "answer": "I'm sorry, I've encountered a critical issue and can't respond right now. Please try again later.",
            "sources": []
        }
