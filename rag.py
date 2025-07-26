import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from vector_store import get_retriever
from knowledge_manager import get_prompts

# --- Constants ---
MODEL_NAME = "gpt-3.5-turbo"

# --- RAG Prompt Template ---
# This is the core instruction set for the AI model.
RAG_PROMPT_TEMPLATE = """
### INSTRUCTIONS TO BOT ###
You are a helpful and professional AI assistant.
Your persona and specific instructions are defined by the text in the [PERSONA INSTRUCTIONS] section below.
Your primary goal is to answer the user's question based on the information provided in the [CONTEXTUAL KNOWLEDGE BASE].
If the knowledge base does not contain the answer, rely on your general training but strictly adhere to your persona, especially the rules about not giving advice.

### PERSONA INSTRUCTIONS ###
{persona_instructions}

### CONTEXTUAL KNOWLEDGE BASE (Your source of truth) ###
{context}

### CURRENT CONVERSATION (Your short-term memory) ###
{chat_history}

### USER'S LATEST MESSAGE ###
Question: {question}

### YOUR RESPONSE ###
Answer:
"""

# --- Core RAG Logic ---

def format_chat_history(chat_history: list) -> str:
    """Formats the chat history into a readable string for the prompt."""
    if not chat_history:
        return "No conversation history yet."
    
    formatted_history = []
    for message in chat_history:
        role = "User" if message['role'] == 'user' else "Assistant"
        formatted_history.append(f"{role}: {message['content']}")
    return "\n".join(formatted_history)

def get_contextual_response(user_question: str, chat_history: list, user_id: str):
    """
    This is the main function that generates the bot's response.
    It retrieves context, builds a prompt, and calls the AI model.
    """
    try:
        # 1. Get the appropriate retriever (custom or base)
        retriever = get_retriever(user_id)
        
        # 2. Load the persona and promotion instructions
        instructions, _ = get_prompts() # We only need the persona instructions here

        # 3. Define the RAG chain using LangChain Expression Language (LCEL)
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.7)

        # This chain manually performs the steps for clarity and reliability
        rag_chain = (
            {
                "context": retriever, 
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
                "persona_instructions": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Format the chat history for the prompt
        formatted_history = format_chat_history(chat_history)

        # Invoke the chain with all necessary inputs
        response = rag_chain.invoke({
            "question": user_question,
            "chat_history": formatted_history,
            "persona_instructions": instructions
        })
        
        # Return the response in the dictionary format the UI expects
        return {"answer": response}

    except Exception as e:
        print(f"ERROR in get_contextual_response: {e}")
        # In case of any failure, return a user-friendly error message
        return {"answer": "I'm sorry, I've encountered an issue and can't respond right now. Please try again later."}

