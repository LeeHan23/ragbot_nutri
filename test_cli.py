import asyncio
import os
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

# Ensure we can import from the other files in the project
from rag import get_contextual_response

# --- Define a user_id for this test session ---
# This allows you to test both the foundational knowledge (by using a new user_id)
# and a custom knowledge base if you've already created one for a specific user.
TEST_USER_ID = "cli_test_user"

# --- Mock User Session Store ---
# This dictionary simulates the user_sessions store
user_data: Dict[str, Any] = {
    "chat_history": [],
}

async def main():
    """
    The main function to run the command-line chat interface for testing.
    """
    print("--- Nutrition Chatbot CLI Tester ---")
    print(f"--- Using User ID: {TEST_USER_ID} ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            # Get input from the user
            user_message = input("\nYou: ")

            if user_message.lower() in ["exit", "quit"]:
                print("\nGoodbye! 👋")
                break

            # --- Get contextual response from the RAG chain ---
            # This now correctly passes the user_id
            response_data = await get_contextual_response(user_message, user_data["chat_history"], TEST_USER_ID)
            
            # --- UPDATE: Handle the new dictionary response ---
            bot_response_text = response_data.get("answer", "Error: No answer found.")
            sources = response_data.get("sources", [])
            knowledge_source = response_data.get("knowledge_source", "Unknown")

            # --- Print the bot's response beautifully ---
            print("\nEva:")
            print(f"  (Using Knowledge: {knowledge_source})")
            print(f"  {bot_response_text}")

            # --- Print the sources used ---
            if sources:
                print("\n  --- Sources ---")
                for i, source in enumerate(sources):
                    source_name = os.path.basename(source.metadata.get('source', 'Unknown'))
                    page_num = source.metadata.get('page', 'N/A')
                    print(f"  [{i+1}] File: {source_name}, Page: {page_num}")
                print("  ---------------")


            # --- Update the chat history ---
            # The history now uses a dictionary format consistent with the UI
            user_data["chat_history"].append({"role": "user", "content": user_message})
            user_data["chat_history"].append({"role": "assistant", "content": bot_response_text})


        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")

