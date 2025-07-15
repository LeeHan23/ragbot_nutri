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
# This should match the user_id you used when running 'create_database.py'
TEST_USER_ID = "test_user_01"

# --- Mock User Session Store ---
# This dictionary simulates the user_sessions store from whatsapp_adapter.py
user_data: Dict[str, Any] = {
    "chat_history": [],
    "visit_count": 0,
    "last_visit_timestamp": None,
    "intent_summary": "No interactions yet."
}

async def main():
    """
    The main function to run the command-line chat interface.
    """
    print("--- Nutrition Chatbot CLI Tester ---")
    print(f"--- Using User ID: {TEST_USER_ID} ---")
    print("Type 'exit' or 'quit' to end the session.")
    print("This is a new session. Simulating a first-time user.")
    
    # Simulate a new visit
    user_data["visit_count"] += 1

    while True:
        try:
            # Get input from the user
            user_message = input("\nYou: ")

            if user_message.lower() in ["exit", "quit"]:
                print("\nGoodbye! 👋")
                break

            # --- Get contextual response from the RAG chain ---
            # This now correctly passes the user_id
            bot_response_text = await get_contextual_response(user_message, user_data, TEST_USER_ID)
            
            # --- Print the bot's response beautifully ---
            print("\nEva:")
            # Split the response into multiple lines for readability
            bot_messages = [msg.strip() for msg in bot_response_text.split('\n') if msg.strip()]
            for message in bot_messages:
                print(f"  {message}")
                await asyncio.sleep(0.5) # Add a small delay for realism

            # --- Update the chat history ---
            # The history now uses the correct LangChain message objects
            user_data["chat_history"].extend([
                HumanMessage(content=user_message),
                AIMessage(content=bot_response_text),
            ])

            # Optional: Limit history size
            max_history_length = 10 # 5 pairs of interactions
            if len(user_data["chat_history"]) > max_history_length:
                user_data["chat_history"] = user_data["chat_history"][-max_history_length:]

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    # To run an async main function, we use asyncio.run()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
