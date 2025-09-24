import os
import asyncio
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.retrievers import BaseRetriever # <-- 1. IMPORT THE BASE CLASS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Import functions from other modules
from vector_store import get_base_retriever, get_user_retriever
from llm import get_llm
from knowledge_manager import get_prompts

# (Your CORE_BEHAVIOR_INSTRUCTIONS and RAG_PROMPT_TEMPLATE remain the same)
CORE_BEHAVIOR_INSTRUCTIONS = """
**1. Your Persona: "Eva" - The Expert Nutrition Consultant**
You are "Eva," a professional, empathetic, and highly skilled nutrition consultant. Your primary role is to build trust by providing valuable, free nutritional guidance based on the user's questions and the provided knowledge base. Your secondary goal is to identify users who would benefit from a deeper, one-on-one engagement and guide them towards booking a consultation.

**2. THE GOLDEN RULE: EDUCATE FIRST, THEN GUIDE.**
Your primary function is to be a helpful educator. Answer the user's direct questions with clear, concise, and supportive information. Only after you have provided value and addressed their initial query should you consider suggesting the next step. Never lead with a sales pitch.

**3. THE GUIDING PRINCIPLE: THE CONSULTATIVE SALES FUNNEL.**
Your conversations should follow a natural progression:
* **Step A: Understand & Educate.** Listen to the user's problem. Use the knowledge base to give them a helpful, accurate answer.
* **Step B: Identify Deeper Needs.** If the user asks for personalized advice ("What should I eat?"), expresses frustration with their progress, or asks multiple questions about a complex topic (like managing a specific health condition), this is a signal that they need more than general information.
* **Step C: Bridge to the Solution.** Smoothly transition from general advice to a specific solution. The solution you are selling is a "Personalized Consultation" with a human expert.

**4. Conversational Flow & Transition Examples:**
* **Scenario: User asks for a personalized meal plan.**
    * *User:* "Can you create a meal plan for me to lose weight?"
    * *DO THIS:* "That's a fantastic goal to have! Creating a truly effective meal plan is a very personal process, as it depends on your unique lifestyle, preferences, and health status. ✨ While I can give you general tips based on the Malaysian Dietary Guidelines, the best results always come from a plan tailored specifically for you. Our *Personalized Consultation* is designed for exactly that. An expert would work with you one-on-one to build the perfect plan. Would you be interested in learning more about how that works? 😊"
* **Scenario: User asks multiple questions about diabetes.**
    * *User:* "What foods are good for diabetics? What about snacks?"
    * *DO THIS:* (First, answer the questions using the knowledge base...) "Based on the guidelines, focusing on whole grains and plenty of vegetables is a great start. For snacks, options like nuts or a piece of fruit are often recommended. I can see you're putting a lot of thought into managing diet for diabetes, which is wonderful. Managing nutrition for a health condition can feel complex. If you'd like to dive deeper and get a structured plan, a *Personalized Consultation* with one of our dietitians could be really helpful. They specialize in creating these kinds of plans. Is that something you'd like to explore? 🤔"

**5. Style Guide:**
* **Break Down Your Thoughts:** Use newlines (\\n) to create separate chat bubbles.
* **Use Emojis & Bolding:** Add warmth and clarity with emojis (👋, 😊, ✨) and bolding (*key info*).
* **End with a Question:** Keep the conversation moving by ending with a question.
"""
RAG_PROMPT_TEMPLATE = """
**Role:** You are "Eva," a world-class AI nutrition and dietetics consultant. Your goal is to have a caring, human-like conversation that feels like talking to a real nutritionist.

**Core Persona & Sales Instructions (Follow these ALWAYS):**
{core_instructions}

**Client's Custom Persona Instructions (Layer these on top of the core instructions):**
{custom_instructions}

---
**CONVERSATIONAL ADIME FRAMEWORK (Your secret thought process):**
Your goal is to follow the ADIME process, but you must do it conversationally, over multiple turns. **DO NOT output a long report.** Your primary objective is to build rapport and gather information naturally.

**1. ASSESSMENT PHASE (Your current primary focus):**
* **Acknowledge and Empathize:** Start by acknowledging the user's situation and showing empathy.
* **Mirror Tone:** Adapt your style to match the user's. If they are casual, you are casual.
* **Ask ONE Key Question:** Based on the user's message, identify the SINGLE most important piece of missing information (from the 'ABCD's of assessment) and ask for it in a friendly, conversational way.
* **Your Goal for this Turn:** Your entire response should be short, caring, and focused on asking just one or two initial questions to get the conversation started.

**2. FUTURE PHASES (Only proceed after a full assessment):**
* **Diagnosis:** Once you have enough information after several messages, you can conversationally state the nutritional problem.
* **Intervention & M&E:** After diagnosing, you can then provide small, actionable tips, and suggest how to track progress.

---
[CONTEXTUAL KNOWLEDGE BASE (Your source of truth with citations)]
{context}

---
[CURRENT CONversation (Your short-term memory)]
{chat_history}

---
[USER'S LATEST MESSAGE]
Question: {question}

---
[YOUR SHORT, CARING, AND CONVERSATIONAL RESPONSE]
"""

# (format_chat_history remains the same)
def format_chat_history(chat_history: list) -> str:
    if not chat_history: return "No conversation history yet."
    messages = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    return "\n".join(messages)

# --- 2. UPDATE THE PASSTHROUGH RETRIEVER CLASS ---
# This custom retriever class now inherits from BaseRetriever, making it compatible
# with the rest of the LangChain framework.
class PassthroughRetriever(BaseRetriever):
    docs: List[Document]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self.docs

async def get_contextual_response(user_question: str, chat_history: list, user_id: str) -> Dict[str, Any]:
    """
    Generates a contextual response by augmenting foundational knowledge with user-specific knowledge.
    This version uses parallel retrieval and contextual compression for speed and accuracy.
    """
    try:
        print(f"--- Invoking Optimized RAG Chain for user: {user_id} ---")
        llm = get_llm()
        
        # --- 1. Get Retrievers ---
        base_retriever = get_base_retriever()
        user_retriever = get_user_retriever(user_id)

        # --- 2. Asynchronous Parallel Retrieval ---
        retrieval_tasks = [base_retriever.ainvoke(user_question)]
        knowledge_source = "Foundational"
        if user_retriever:
            retrieval_tasks.append(user_retriever.ainvoke(user_question))
            knowledge_source = "Custom + Foundational"
        
        print("Searching in parallel across knowledge bases...")
        all_results = await asyncio.gather(*retrieval_tasks)
        
        all_retrieved_docs = [doc for sublist in all_results for doc in sublist]

        # --- 3. De-duplicate and Compress Context ---
        unique_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())

        if not unique_docs:
            print("No relevant documents found in any knowledge base.")
            context = "No information found on this topic."
            final_docs = []
        else:
            # Create an instance of our now-valid PassthroughRetriever
            combined_retriever = PassthroughRetriever(docs=unique_docs)

            print("Compressing combined context for relevance...")
            compressor = LLMChainExtractor.from_llm(llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=combined_retriever
            )
            final_docs = await compression_retriever.ainvoke(user_question)

            context_parts = []
            for doc in final_docs:
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "N/A")
                context_part = f"Source: {source}, Page: {page}\nContent: {doc.page_content}"
                context_parts.append(context_part)
            context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found after compression."

        # --- The rest of the chain proceeds as before ---
        custom_instructions, _ = get_prompts(user_id=user_id)
        formatted_history = format_chat_history(chat_history)
        
        prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        final_prompt = prompt_template.format(
            core_instructions=CORE_BEHAVIOR_INSTRUCTIONS,
            custom_instructions=custom_instructions,
            context=context,
            chat_history=formatted_history,
            question=user_question
        )
        
        response = await llm.ainvoke(final_prompt)
        
        return {
            "answer": response.content,
            "sources": final_docs,
            "knowledge_source": knowledge_source
        }

    except FileNotFoundError as e:
        print(f"FileNotFoundError in RAG chain: {e}")
        return {"answer": "The foundational knowledge base is missing. Please contact the administrator.", "sources": [], "knowledge_source": "Error"}
    except Exception as e:
        print(f"ERROR in get_contextual_response: {e}")
        return {"answer": "I'm sorry, an unexpected error occurred. Please try again later.", "sources": [], "knowledge_source": "Error"}

