import os
import json
import asyncio
import requests
import redis.asyncio as redis # Import the async redis library
from fastapi import APIRouter, Request, Response, HTTPException
from dotenv import load_dotenv
from typing import Dict, Any

from rag import get_contextual_response

# --- Load Environment Variables ---
load_dotenv()

# --- Meta WhatsApp Cloud API Configuration ---
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379") # Add REDIS_URL to your .env

if not all([WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID, VERIFY_TOKEN]):
    raise EnvironmentError("Missing required Meta WhatsApp environment variables.")

# --- IMPROVEMENT: Redis Client for Scalable Session Management ---
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    print(f"Could not connect to Redis: {e}")
    redis_client = None

async def load_user_session(sender_id: str) -> Dict[str, Any]:
    """Loads a user's session from Redis."""
    if not redis_client: return {"chat_history": []}
    session_data = await redis_client.get(f"session:{sender_id}")
    if session_data:
        return json.loads(session_data)
    return {"chat_history": []}

async def save_user_session(sender_id: str, user_data: Dict[str, Any]):
    """Saves a user's session to Redis with an expiration time."""
    if not redis_client: return
    # Expire sessions after 24 hours of inactivity
    await redis_client.set(f"session:{sender_id}", json.dumps(user_data), ex=86400)

# --- FastAPI Router ---
whatsapp_router = APIRouter()

def send_whatsapp_message(to_number: str, message_body: str):
    """Sends a reply message to the user's WhatsApp number using Meta's Cloud API."""
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
    data = {"messaging_product": "whatsapp", "to": to_number, "text": {"body": message_body}}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Message sent successfully to {to_number}.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending WhatsApp message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send WhatsApp message.")

@whatsapp_router.get("/webhook")
async def whatsapp_verify_webhook(request: Request):
    # (Webhook verification logic remains the same)
    if request.query_params.get("hub.mode") == "subscribe" and request.query_e_params.get("hub.challenge"):
        if request.query_params.get("hub.verify_token") == VERIFY_TOKEN:
            return Response(content=request.query_params["hub.challenge"], status_code=200)
        return Response(content="Verification token mismatch", status_code=403)
    return Response(content="Invalid request", status_code=400)

@whatsapp_router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """Handles incoming WhatsApp messages."""
    data = await request.json()
    try:
        message_data = data["entry"][0]["changes"][0]["value"]["messages"][0]
        user_message = message_data["text"]["body"]
        sender_id = message_data["from"]
        
        print(f"Received message from {sender_id}: '{user_message}'")

        user_data = await load_user_session(sender_id)
        response_data = await get_contextual_response(user_message, user_data.get("chat_history", []), sender_id)
        bot_response_text = response_data.get("answer", "I'm not sure how to respond.")
        
        send_whatsapp_message(to_number=sender_id, message_body=bot_response_text)

        user_data["chat_history"].append({"role": "user", "content": user_message})
        user_data["chat_history"].append({"role": "assistant", "content": bot_response_text})
        
        await save_user_session(sender_id, user_data)

    except (KeyError, IndexError):
        print("Webhook received a non-message event or malformed data.")
    except Exception as e:
        print(f"An unexpected error occurred in the webhook: {e}")
        # Avoid sending an error message back to the user unless necessary
    
    return Response(status_code=200)
