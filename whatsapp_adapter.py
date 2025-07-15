import os
import re
import json
import asyncio
import aiofiles
from fastapi import APIRouter, Request, Response, HTTPException
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta

from rag import get_contextual_response

# --- Load Environment Variables ---
load_dotenv()

# --- Twilio Client Initialization ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER]):
    raise EnvironmentError("Missing required Twilio environment variables.")

try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
except Exception as e:
    print(f"Error initializing Twilio client: {e}")
    twilio_client = None

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
            session_data = json.loads(content)
            if session_data.get("last_visit_timestamp"):
                session_data["last_visit_timestamp"] = datetime.fromisoformat(session_data["last_visit_timestamp"])
            return session_data
    except (json.JSONDecodeError, FileNotFoundError):
        return { "chat_history": [], "visit_count": 0, "last_visit_timestamp": None, "intent_summary": "No interactions yet."}

async def save_user_session(sender_id: str, user_data: Dict[str, Any]):
    """Saves a user's session to a JSON file."""
    session_path = os.path.join(SESSIONS_DIR, f"{sender_id}.json")
    data_to_save = user_data.copy()
    if data_to_save.get("last_visit_timestamp"):
        data_to_save["last_visit_timestamp"] = data_to_save["last_visit_timestamp"].isoformat()
    
    async with aiofiles.open(session_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data_to_save, indent=4))

# --- FastAPI Router ---
whatsapp_router = APIRouter()

def send_whatsapp_message(to_number: str, message_body: str):
    """Sends a reply message to the user's WhatsApp number using Twilio."""
    if not twilio_client:
        print("Twilio client not initialized. Cannot send message.")
        raise HTTPException(status_code=500, detail="Twilio service is not configured.")
    try:
        print(f"Sending message to {to_number}: '{message_body}'")
        twilio_client.messages.create(from_=TWILIO_WHATSAPP_NUMBER, body=message_body, to=to_number)
    except TwilioRestException as e:
        print(f"Error sending WhatsApp message via Twilio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send WhatsApp message: {e}")

@whatsapp_router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Handles incoming WhatsApp messages, maintains persistent user state,
    and sends intent-aware contextual replies.
    """
    form_data = await request.form()
    user_message = form_data.get("Body")
    sender_id_raw = form_data.get("From")

    if not user_message or not sender_id_raw:
        raise HTTPException(status_code=400, detail="Missing 'Body' or 'From' in webhook payload.")
    
    sender_id = re.sub(r'\W+', '', sender_id_raw)
    print(f"Received message from {sender_id}: '{user_message}'")

    lock = session_locks[sender_id]
    async with lock:
        try:
            user_data = await load_user_session(sender_id)
            now = datetime.utcnow()
            
            is_new_visit = True
            if user_data.get("last_visit_timestamp"):
                if now - user_data["last_visit_timestamp"] < timedelta(minutes=30):
                    is_new_visit = False
            
            if is_new_visit:
                user_data["visit_count"] = user_data.get("visit_count", 0) + 1
                user_data["chat_history"] = [] 

            # --- Get contextual response from the RAG chain ---
            bot_response_text = await get_contextual_response(user_message, user_data)
            
            # --- We no longer need to parse JSON, simplifying this section ---
            bot_messages = [msg.strip() for msg in bot_response_text.split('\n') if msg.strip()]
            if not bot_messages:
                 bot_messages = ["I'm not sure how to respond to that, could you ask in another way?"]

            print(f"Generated {len(bot_messages)} sequential messages for visit #{user_data['visit_count']}.")

            # --- Send Replies Sequentially ---
            for i, message in enumerate(bot_messages):
                send_whatsapp_message(to_number=sender_id_raw, message_body=message)
                if i < len(bot_messages) - 1:
                    await asyncio.sleep(1.5)

            # --- Update User State ---
            # For now, we just update the chat history. The intent summary logic can be added back later.
            user_data["chat_history"].append((user_message, bot_response_text))
            user_data["last_visit_timestamp"] = now
            
            max_history_length = 10
            if len(user_data["chat_history"]) > max_history_length:
                user_data["chat_history"] = user_data["chat_history"][-max_history_length:]

            await save_user_session(sender_id, user_data)

        except Exception as e:
            print(f"An unexpected error occurred in the webhook for user {sender_id}: {e}")
            send_whatsapp_message(to_number=sender_id_raw, message_body="I'm sorry, an unexpected error occurred. Please try again in a moment.")
            raise HTTPException(status_code=500, detail="An internal server error occurred.")

    return Response(status_code=200)
