import os
import asyncio
from fastapi import APIRouter, Request, Response, HTTPException
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv

from rag import get_rag_response

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
    Handles incoming WhatsApp messages from the Twilio webhook.
    It gets a list of responses from the RAG pipeline and sends them sequentially.
    """
    try:
        form_data = await request.form()
        user_message = form_data.get("Body")
        sender_id = form_data.get("From")

        if not user_message or not sender_id:
            raise HTTPException(status_code=400, detail="Missing 'Body' or 'From' in webhook payload.")

        print(f"Received message from {sender_id}: '{user_message}'")

        # --- Get a LIST of response messages from RAG Pipeline ---
        bot_responses = await get_rag_response(user_message)
        print(f"Generated {len(bot_responses)} sequential messages.")

        # --- Send Replies Sequentially ---
        for i, message in enumerate(bot_responses):
            send_whatsapp_message(to_number=sender_id, message_body=message)
            # Add a small delay between messages to feel more natural, but not for the last one
            if i < len(bot_responses) - 1:
                await asyncio.sleep(1.5) # 1.5-second delay

        return Response(status_code=200)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred in the webhook: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
