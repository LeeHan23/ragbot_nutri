import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from vector_store import get_base_retriever, get_user_retriever
from llm import get_llm
from knowledge_manager import get_prompts
# --- NEW: Import the agent tools ---
from agent_tools import log_customer_data, generate_progress_report

# (Your CORE_BEHAVIOR_INSTRUCTIONS remains the same)
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
# --- REVISED RAG PROMPT FOR AGENTIC BEHAVIOR ---
RAG_PROMPT_TEMPLATE = """
**Role:** You are "Eva," a world-class AI nutrition consultant and health coach. You can chat, provide advice, and also help customers track their health progress.

**Core Persona & Sales Instructions (Follow these ALWAYS):**
{core_instructions}

**Client's Custom Persona Instructions (Layer these on top of the core instructions):**
{custom_instructions}

---
**PRIMARY OBJECTIVE: TOOL USE FOR PROGRESS TRACKING**
Your most important new task is to identify when a customer is providing a health update. If their message contains a specific metric (like weight, blood sugar, etc.), you MUST use your tools to log it.

**Available Tools:**
1.  `log_customer_data(customer_contact: str, user_id: str, metric_name: str, metric_value: str, notes: str)`: Use this to record a data point. `customer_contact` is the customer's unique identifier. `user_id` is the business account this customer belongs to.
2.  `generate_progress_report(customer_contact: str, user_id: str)`: Use this if the customer asks for a summary of their progress.

**How to Use Tools:**
When you need to use a tool, you MUST respond with ONLY a JSON object in the following format:
`{{"tool_name": "function_name", "parameters": {{"arg1": "value1", ...}}}}`

**Tool-Use Examples:**
* User says: "Hi Eva, just wanted to let you know I weighed myself this morning and I'm 82.5kg!"
    * *Your thought process:* "The user is providing a weight update. I must log this."
    * *Your response:* `{{"tool_name": "log_customer_data", "parameters": {{"customer_contact": "{customer_contact}", "user_id": "{user_id}", "metric_name": "Weight", "metric_value": "82.5kg", "notes": "Morning weigh-in"}}}}`
* User asks: "How have I been doing over the last month?"
    * *Your thought process:* "The user wants a summary. I need to generate a report."
    * *Your response:* `{{"tool_name": "generate_progress_report", "parameters": {{"customer_contact": "{customer_contact}", "user_id": "{user_id}"}}}}`

If the user is NOT providing a trackable update or asking for a report, then proceed with the Conversational ADIME Framework below.
---
**CONVERSATIONAL ADIME FRAMEWORK (Your secret thought process):**
(This section remains the same for conversational chat)
**1. ASSESSMENT PHASE (Your current primary focus):**
* **Acknowledge and Empathize:** Start by acknowledging the user's situation and showing empathy. The user may have sent several rapid messages; treat their combined input as a single, complete thought.
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
[CURRENT CONVERSATION (Your short-term memory)]
{chat_history}

---
[USER'S LATEST MESSAGE]
Question: {question}

---
[YOUR RESPONSE (Either a JSON tool call or a conversational message)]
"""

def format_chat_history(chat_history: list) -> str:
    if not chat_history: return "No conversation history yet."
    messages = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    return "\n".join(messages)

class PassthroughRetriever(BaseRetriever):
    docs: List[Document]
    def _get_relevant_documents(self, query: str) -> List[Document]: return self.docs
    async def _aget_relevant_documents(self, query: str) -> List[Document]: return self.docs

async def get_contextual_response(user_question: str, chat_history: list, user_id: str, customer_contact: str) -> Dict[str, Any]:
    """
    This function now acts as an agent executor. It can chat, use tools, or retrieve information.
    """
    print(f"--- Invoking Agent Chain for user: {user_id}, customer: {customer_contact} ---")
    llm = get_llm()
    
    # --- 1. First LLM call to decide if a tool is needed ---
    custom_instructions, _ = get_prompts(user_id=user_id)
    formatted_history = format_chat_history(chat_history)
    
    agent_prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    agent_prompt = agent_prompt_template.format(
        core_instructions=CORE_BEHAVIOR_INSTRUCTIONS,
        custom_instructions=custom_instructions,
        context="No context needed for this step.",
        chat_history=formatted_history,
        question=user_question,
        customer_contact=customer_contact, # Pass IDs to the prompt
        user_id=user_id
    )
    
    initial_response = await llm.ainvoke(agent_prompt)
    response_text = initial_response.content.strip()

    # --- 2. Check if the response is a tool call ---
    if response_text.startswith('{') and response_text.endswith('}'):
        try:
            tool_call = json.loads(response_text)
            tool_name = tool_call.get("tool_name")
            parameters = tool_call.get("parameters", {})
            
            print(f"Executing tool: {tool_name} with params: {parameters}")

            if tool_name == "log_customer_data":
                result = log_customer_data(**parameters)
            elif tool_name == "generate_progress_report":
                result = generate_progress_report(**parameters)
            else:
                result = "Unknown tool."
            
            # --- 3. Second LLM call to convert tool result into a natural response ---
            second_prompt = f"The user's last message was: '{user_question}'. You used the tool '{tool_name}' and got this result: '{result}'. Now, provide a short, friendly, and encouraging response to the user based on this result. For example, if data was logged successfully, say something like 'That's great, I've logged that for you! Keep up the amazing work!'"
            final_response = await llm.ainvoke(second_prompt)
            
            return { "answer": final_response.content, "sources": [], "knowledge_source": "Agent Tool" }

        except json.JSONDecodeError:
            pass # Not a valid tool call, proceed to RAG

    # --- If not a tool call, proceed with the original RAG process ---
    base_retriever = get_base_retriever()
    user_retriever = get_user_retriever(user_id)
    retrieval_tasks = [base_retriever.ainvoke(user_question)]
    knowledge_source = "Foundational"
    if user_retriever:
        retrieval_tasks.append(user_retriever.ainvoke(user_question))
        knowledge_source = "Custom + Foundational"
    
    all_results = await asyncio.gather(*retrieval_tasks)
    all_retrieved_docs = [doc for sublist in all_results for doc in sublist]
    unique_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())
    
    if not unique_docs:
        context = "No information found on this topic."
        final_docs = []
    else:
        combined_retriever = PassthroughRetriever(docs=unique_docs)
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=combined_retriever)
        final_docs = await compression_retriever.ainvoke(user_question)
        context_parts = [f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}" for doc in final_docs]
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found after compression."

    final_prompt_rag = agent_prompt_template.format(
        core_instructions=CORE_BEHAVIOR_INSTRUCTIONS,
        custom_instructions=custom_instructions,
        context=context,
        chat_history=formatted_history,
        question=user_question,
        customer_contact=customer_contact,
        user_id=user_id
    )
    
    final_response_rag = await llm.ainvoke(final_prompt_rag)
    
    return {
        "answer": final_response_rag.content,
        "sources": final_docs,
        "knowledge_source": knowledge_source
    }

