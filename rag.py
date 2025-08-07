import os
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers.multi_query import MultiQueryRetriever

# Import functions from other modules
from vector_store import get_retriever
from llm import get_llm
from knowledge_manager import get_prompts

# --- ADVANCED RAG PROMPT TEMPLATE ---
# This version is updated to handle and require direct source citations.
RAG_PROMPT_TEMPLATE = """
**Role:** You are "Eva," a world-class AI nutrition and dietetics consultant. Your task is to provide a detailed, evidence-based consultation based on the patient's situation.

**Objective:** Analyze the user's question and the provided [CONTEXTUAL KNOWLEDGE BASE] to formulate a comprehensive and professional response. You must synthesize information from all relevant context documents, not just one.

**Persona Instructions:**
{persona_instructions}

**Process:**
1.  **Assess the Situation:** Carefully read the [USER'S LATEST MESSAGE] and the [CURRENT CONVERSATION] to fully understand the patient's condition, goals, and constraints.
2.  **Synthesize Knowledge:** Review the entire [CONTEXTUAL KNOWLEDGE BASE]. Each entry is tagged with its source file and page number.
3.  **Formulate Response:** Structure your answer like a professional consultation note with the following sections:
    * **Assessment:** Briefly summarize your understanding of the patient's situation based on their query.
    * **Key Considerations:** Point out the most important nutritional factors and principles from the knowledge base that apply to this case.
    * **Recommendations:** Provide clear, actionable, and evidence-based recommendations.
    * **Rationale:** For each recommendation, briefly explain *why* you are suggesting it.
4.  **Constraint & Citation Mandate:** You MUST base your answer *only* on the information provided in the [CONTEXTUAL KNOWLEDGE BASE]. Crucially, at the end of each key point or recommendation, you **MUST** cite the source using the exact format `[Source: filename.pdf, Page: X]`. If the context is insufficient, state that and advise consulting a human professional.

---
[CONTEXTUAL KNOWLEDGE BASE (Your source of truth with citations)]
{context}

---
[CURRENT CONVERSATION (Your short-term memory)]
{chat_history}

---
[USER'S LATEST MESSAGE]
Question: {question}

---
[YOUR PROFESSIONAL RESPONSE]
"""

def format_chat_history(chat_history: list) -> str:
    """Formats the chat history into a readable string for the prompt."""
    if not chat_history:
        return "No conversation history yet."
    
    langchain_messages = []
    for message in chat_history:
        if message['role'] == 'user':
            langchain_messages.append(HumanMessage(content=message['content']))
        elif message['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=message['content']))
    
    formatted_history = []
    for msg in langchain_messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted_history.append(f"{role}: {msg.content}")
        
    return "\n".join(formatted_history)

async def get_contextual_response(user_question: str, chat_history: list, user_id: str) -> Dict[str, Any]:
    """
    Generates a sophisticated, contextual response using an advanced RAG strategy.
    """
    try:
        print(f"--- Invoking Advanced RAG Chain for user: {user_id} ---")
        llm = get_llm()
        
        # 1. Get the base retriever
        base_retriever, knowledge_source = get_retriever(user_id)
        
        # 2. Use MultiQueryRetriever to generate multiple queries
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm
        )
        
        retrieved_docs = multi_query_retriever.invoke(user_question)
        
        # --- ENHANCEMENT: Format the context to include source and page number ---
        context_parts = []
        for doc in retrieved_docs:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            context_part = f"Source: {source}, Page: {page}\nContent: {doc.page_content}"
            context_parts.append(context_part)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # 3. Load the persona instructions
        instructions, _ = get_prompts()
        
        # 4. Format the chat history
        formatted_history = format_chat_history(chat_history)
        
        # 5. Build the final, more detailed prompt
        prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        final_prompt = prompt_template.format(
            persona_instructions=instructions,
            context=context,
            chat_history=formatted_history,
            question=user_question
        )
        
        # 6. Call the AI model with the enhanced prompt and context
        response = await llm.ainvoke(final_prompt)
        
        # 7. Return the response and sources
        return {
            "answer": response.content,
            "sources": retrieved_docs,
            "knowledge_source": knowledge_source
        }

    except FileNotFoundError:
        return {
            "answer": "It looks like I don't have a knowledge base for you yet. Please upload some documents to get started!",
            "sources": [],
            "knowledge_source": "None"
        }
    except Exception as e:
        print(f"ERROR in get_contextual_response: {e}")
        return {
            "answer": "I'm sorry, I've encountered a critical issue and can't respond right now. Please try again later.",
            "sources": [],
            "knowledge_source": "Error"
        }
