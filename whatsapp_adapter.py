import os
import re
import json
import asyncio
import aiofiles
import requests # Use requests library to send messages
from fastapi import APIRouter, Request, Response, HTTPException
from dotenv import load_dotenv
from collections import defaultdict
from typing import Dict, Any
from datetime import datetime, timedelta

from rag import get_contextual_response

# --- Load Environment Variables ---
load_dotenv()

# --- Meta WhatsApp Cloud API Configuration ---
# These are the new variables you'll get from your Meta for Developers App Dashboard
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN") # Your secret verify token

if not all([WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID, VERIFY_TOKEN]):
    raise EnvironmentError("Missing required Meta WhatsApp environment variables.")

# --- Persistent User State Store ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

session_locks = defaultdict(asyncio.Lock)

async def load_user_session(sender_id: str) -> Dict[str, Any]:
    """Loads a user's session from a JSON file."""
    session_path = os.path.join(SESSIONS_DIR, f"{sender_id}.json")
    if not os.path.exists(session_path):
        return { "chat_history": [], "visit_count": 0, "last_visit_timestamp": None, "intent_summary": "No interactions yet."}
    try:
        async with aiofiles.open(session_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        return { "chat_history": [], "visit_count": 0, "last_visit_timestamp": None, "intent_summary": "No interactions yet."}

async def save_user_session(sender_id: str, user_data: Dict[str, Any]):
    """Saves a user's session to a JSON file."""
    session_path = os.path.join(SESSIONS_DIR, f"{sender_id}.json")
    async with aiofiles.open(session_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(user_data, indent=4))

# --- FastAPI Router ---
whatsapp_router = APIRouter()

def send_whatsapp_message(to_number: str, message_body: str):
    """Sends a reply message to the user's WhatsApp number using Meta's Cloud API."""
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {"body": message_body},
    }
    
    try:
        print(f"Sending message to {to_number}: '{message_body}'")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Raises an exception for bad status codes
        print(f"Message sent successfully. Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending WhatsApp message via Meta API: {e}")
        # In a real app, you might want more robust error handling here
        raise HTTPException(status_code=500, detail=f"Failed to send WhatsApp message: {e}")


@whatsapp_router.get("/webhook")
async def whatsapp_verify_webhook(request: Request):
    """
    Handles the webhook verification challenge from Meta.
    """
    if request.query_params.get("hub.mode") == "subscribe" and request.query_params.get("hub.challenge"):
        if request.query_params.get("hub.verify_token") == VERIFY_TOKEN:
            return Response(content=request.query_params["hub.challenge"], status_code=200)
        return Response(content="Verification token mismatch", status_code=403)
    return Response(content="Invalid request", status_code=400)


@whatsapp_router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Handles incoming WhatsApp messages from Meta's webhook.
    """
    data = await request.json()
    print("--- Received WhatsApp Webhook ---")
    print(json.dumps(data, indent=2))

    # Ensure the webhook is for a message and has the necessary data
    if "entry" in data and data["entry"] and "changes" in data["entry"][0] and data["entry"][0]["changes"]:
        change = data["entry"][0]["changes"][0]
        if "value" in change and "messages" in change["value"] and change["value"]["messages"]:
            message_data = change["value"]["messages"][0]
            
            # Extract user message and sender ID
            user_message = message_data["text"]["body"]
            sender_id = message_data["from"] # This is the user's phone number
            
            print(f"Received message from {sender_id}: '{user_message}'")

            lock = session_locks[sender_id]
            async with lock:
                try:
                    user_data = await load_user_session(sender_id)
                    
                    # --- Get contextual response from the RAG chain ---
                    # Note: The RAG function needs to be adapted if it expects a different chat history format
                    response_data = await get_contextual_response(user_message, user_data.get("chat_history", []), sender_id)
                    bot_response_text = response_data.get("answer", "I'm not sure how to respond to that.")
                    
                    # Send the reply
                    send_whatsapp_message(to_number=sender_id, message_body=bot_response_text)

                    # --- Update User State ---
                    if "chat_history" not in user_data:
                        user_data["chat_history"] = []
                    user_data["chat_history"].append({"role": "user", "content": user_message})
                    user_data["chat_history"].append({"role": "assistant", "content": bot_response_text})
                    
                    await save_user_session(sender_id, user_data)

                except Exception as e:
                    print(f"An unexpected error occurred in the webhook for user {sender_id}: {e}")
                    send_whatsapp_message(to_number=sender_id, message_body="I'm sorry, an unexpected error occurred. Please try again in a moment.")
                    raise HTTPException(status_code=500, detail="An internal server error occurred.")

    return Response(status_code=200)
