import os
import json
import asyncio
import requests
import redis.asyncio as redis
from fastapi import APIRouter, Request, Response, HTTPException
from dotenv import load_dotenv
from typing import Dict, Any

from rag import get_contextual_response

# --- Load Environment Variables ---
load_dotenv()

# --- Meta WhatsApp & Redis Configuration ---
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# --- NEW: Configuration for message batching ---
# Time to wait in seconds after the last message before processing the batch.
MESSAGE_BATCH_DELAY = 3 

# A dictionary to keep track of background processing tasks for each user
processing_tasks = {}

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    print(f"Could not connect to Redis: {e}")
    redis_client = None

# (Session management functions remain the same)
async def load_user_session(sender_id: str) -> Dict[str, Any]:
    if not redis_client: return {"chat_history": []}
    session_data = await redis_client.get(f"session:{sender_id}")
    if session_data:
        return json.loads(session_data)
    return {"chat_history": []}

async def save_user_session(sender_id: str, user_data: Dict[str, Any]):
    if not redis_client: return
    await redis_client.set(f"session:{sender_id}", json.dumps(user_data), ex=86400)

# --- FastAPI Router ---
whatsapp_router = APIRouter()

# --- NEW: Background processing function ---
async def process_message_batch(sender_id: str, user_id: str):
    """
    Waits for a delay, then processes all buffered messages for a user.
    """
    await asyncio.sleep(MESSAGE_BATCH_DELAY)
    
    buffer_key = f"buffer:{sender_id}"
    
    # Retrieve all buffered messages and combine them
    buffered_messages = await redis_client.lrange(buffer_key, 0, -1)
    if not buffered_messages:
        return # No messages to process
        
    combined_message = " ".join(buffered_messages)
    
    print(f"Processing batched message from {sender_id}: '{combined_message}'")

    # Clear the buffer now that we're processing it
    await redis_client.delete(buffer_key)
    
    # Get user session and RAG response
    user_data = await load_user_session(sender_id)
    response_data = await get_contextual_response(combined_message, user_data.get("chat_history", []), user_id)
    bot_response_text = response_data.get("answer", "I'm not sure how to respond to that.")
    
    send_whatsapp_message(to_number=sender_id, message_body=bot_response_text)

    # Update chat history with the combined interaction
    user_data["chat_history"].append({"role": "user", "content": combined_message})
    user_data["chat_history"].append({"role": "assistant", "content": bot_response_text})
    await save_user_session(sender_id, user_data)
    
    # Remove the task from the tracking dictionary
    processing_tasks.pop(sender_id, None)


@whatsapp_router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Handles incoming WhatsApp messages using a debounced batching strategy.
    """
    data = await request.json()
    try:
        message_data = data["entry"][0]["changes"][0]["value"]["messages"][0]
        user_message = message_data["text"]["body"]
        sender_id = message_data["from"]

        # For this example, we'll use the sender_id as the user_id for RAG.
        # In a multi-tenant system, you'd look up the user_id based on the business number.
        user_id_for_rag = sender_id 

        # --- NEW BATCHING LOGIC ---
        # Add the new message to this user's buffer list in Redis
        buffer_key = f"buffer:{sender_id}"
        await redis_client.rpush(buffer_key, user_message)

        # If a processing task for this user already exists, cancel it.
        if sender_id in processing_tasks:
            processing_tasks[sender_id].cancel()
        
        # Schedule a new processing task to run after the delay.
        task = asyncio.create_task(process_message_batch(sender_id, user_id_for_rag))
        processing_tasks[sender_id] = task

    except (KeyError, IndexError):
        print("Webhook received a non-message event or malformed data.")
    except Exception as e:
        print(f"An unexpected error occurred in the webhook: {e}")
    
    # Always return 200 OK immediately to acknowledge receipt of the message
    return Response(status_code=200)

# (send_whatsapp_message and webhook verification remain the same)
def send_whatsapp_message(to_number: str, message_body: str):
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
    data = {"messaging_product": "whatsapp", "to": to_number, "text": {"body": message_body}}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending WhatsApp message: {e}")

@whatsapp_router.get("/webhook")
async def whatsapp_verify_webhook(request: Request):
    if request.query_params.get("hub.mode") == "subscribe" and request.query_params.get("hub.challenge"):
        if request.query_params.get("hub.verify_token") == VERIFY_TOKEN:
            return Response(content=request.query_params["hub.challenge"], status_code=200)
        return Response(content="Verification token mismatch", status_code=403)
    return Response(content="Invalid request", status_code=400)

