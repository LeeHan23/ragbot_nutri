import os
from fastapi import APIRouter, Request, Response, HTTPException
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv

from rag import get_rag_response

# --- Load Environment Variables ---
# Load credentials from a .env file for security
load_dotenv()

# --- Twilio Client Initialization ---
# It's recommended to use environment variables for sensitive data
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER") # e.g., 'whatsapp:+14155238886'

# Validate that all required environment variables are set
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER]):
    raise EnvironmentError(
        "Missing required Twilio environment variables. "
        "Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_NUMBER."
    )

# Initialize the Twilio client
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
except Exception as e:
    print(f"Error initializing Twilio client: {e}")
    twilio_client = None

# --- FastAPI Router ---
whatsapp_router = APIRouter()

# --- Helper Function to Send WhatsApp Message ---
def send_whatsapp_message(to_number: str, message_body: str):
    """
    Sends a reply message to the user's WhatsApp number using Twilio.

    Args:
        to_number (str): The recipient's WhatsApp number (e.g., 'whatsapp:+1234567890').
        message_body (str): The text of the message to send.
    """
    if not twilio_client:
        print("Twilio client not initialized. Cannot send message.")
        # In a production environment, you might want to raise an exception
        # or have a more robust fallback/notification mechanism.
        raise HTTPException(status_code=500, detail="Twilio service is not configured.")

    try:
        print(f"Sending message to {to_number}: '{message_body}'")
        message = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=to_number
        )
        print(f"Message sent successfully. SID: {message.sid}")
    except TwilioRestException as e:
        print(f"Error sending WhatsApp message via Twilio: {e}")
        # Handle specific Twilio errors if necessary
        raise HTTPException(status_code=500, detail=f"Failed to send WhatsApp message: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while sending message: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# --- Webhook Endpoint ---
@whatsapp_router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Handles incoming WhatsApp messages from the Twilio webhook.

    This endpoint expects a URL-encoded form data payload from Twilio.
    It extracts the user's message and phone number, gets a response from the RAG pipeline,
    and sends a reply back to the user.
    """
    try:
        # Twilio sends data as application/x-www-form-urlencoded
        form_data = await request.form()
        user_message = form_data.get("Body")
        sender_id = form_data.get("From")

        # --- Input Validation ---
        if not user_message or not sender_id:
            print(f"Webhook received incomplete data: Body='{user_message}', From='{sender_id}'")
            raise HTTPException(
                status_code=400,
                detail="Missing 'Body' or 'From' in webhook payload."
            )

        print(f"Received message from {sender_id}: '{user_message}'")

        # --- Get Response from RAG Pipeline ---
        # This function encapsulates the entire RAG logic
        bot_response = await get_rag_response(user_message)
        print(f"Generated RAG response: '{bot_response}'")

        # --- Send Reply ---
        send_whatsapp_message(to_number=sender_id, message_body=bot_response)

        # --- Acknowledge Receipt ---
        # Return an empty response with a 200 status code to let Twilio know
        # the message was received successfully. No XML/TwiML is needed if
        # you handle replies asynchronously like this.
        return Response(status_code=200)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to be handled by FastAPI's default error handling
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred in the webhook: {e}")
        # For security, return a generic error message to the client
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
