import os
import re
import asyncio
from fastapi import APIRouter, Request, Response, HTTPException
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv
from collections import defaultdict
from typing import List, Tuple

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

# --- In-Memory Chat History Store ---
# NOTE: This is a simple in-memory store. For production, you would replace
# this with a persistent database like Redis or a simple file-based store
# to prevent history from being lost on server restart.
conversation_history = defaultdict(list)

# --- FastAPI Router ---
whatsapp_router = APIRouter()

# --- Helper Function to Send WhatsApp Message ---
def send_whatsapp_message(to_number: str, message_body: str):
    """Sends a reply message to the user's WhatsApp number using Twilio."""
    if not twilio_client:
        print("Twilio client not initialized. Cannot send message.")
        raise HTTPException(status_code=500, detail="Twilio service is not configured.")

    try:
        print(f"Sending message to {to_number}: '{message_body}'")
        twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=to_number
        )
    except TwilioRestException as e:
        print(f"Error sending WhatsApp message via Twilio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send WhatsApp message: {e}")

# --- Webhook Endpoint ---
@whatsapp_router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Handles incoming WhatsApp messages, maintains conversation history,
    and sends contextual replies.
    """
    try:
        form_data = await request.form()
        user_message = form_data.get("Body")
        sender_id = form_data.get("From")

        if not user_message or not sender_id:
            raise HTTPException(status_code=400, detail="Missing 'Body' or 'From' in webhook payload.")

        print(f"Received message from {sender_id}: '{user_message}'")

        # --- Retrieve user's chat history ---
        history: List[Tuple[str, str]] = conversation_history[sender_id]
        
        # --- Get contextual response from the RAG chain ---
        bot_response_text = await get_contextual_response(user_message, history)

        # --- Split the response into multiple messages ---
        # The AI is instructed to create short messages. We can split by newline.
        bot_messages = [msg.strip() for msg in bot_response_text.split('\n') if msg.strip()]
        if not bot_messages:
             bot_messages = ["I'm not sure how to respond to that, could you ask in another way?"]


        print(f"Generated {len(bot_messages)} sequential messages.")

        # --- Send Replies Sequentially ---
        for i, message in enumerate(bot_messages):
            send_whatsapp_message(to_number=sender_id, message_body=message)
            if i < len(bot_messages) - 1:
                await asyncio.sleep(1.5)

        # --- Update the chat history ---
        # Add the user's message and the full bot response to the history
        conversation_history[sender_id].append((user_message, bot_response_text))
        
        # Optional: Limit history size to prevent it from growing too large
        max_history_length = 10 # Keep last 5 pairs of interactions
        if len(conversation_history[sender_id]) > max_history_length:
            conversation_history[sender_id] = conversation_history[sender_id][-max_history_length:]


        return Response(status_code=200)

    except Exception as e:
        print(f"An unexpected error occurred in the webhook: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
