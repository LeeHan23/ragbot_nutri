from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import json
import asyncio

# Import the same core RAG function
from rag import get_contextual_response

website_router = APIRouter()

# A simple in-memory store for session histories for website users
# For production, this could also use Redis
website_sessions: Dict[str, Any] = {}

@website_router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    Handles the real-time chat connection from a website widget.
    
    The user_id here would correspond to your client's account, so you can
    load the correct knowledge base.
    """
    await websocket.accept()
    
    # Use the websocket's client host/port as a unique session ID
    session_id = f"{websocket.client.host}:{websocket.client.port}"
    website_sessions[session_id] = {"chat_history": []}
    
    print(f"Website client {session_id} connected for user_id: {user_id}")
    
    try:
        while True:
            # Wait for a message from the website widget
            user_message = await websocket.receive_text()
            
            # Get the user's chat history for this session
            chat_history = website_sessions[session_id]["chat_history"]
            
            # Get a response from the same RAG "brain"
            response_data = await get_contextual_response(user_message, chat_history, user_id)
            bot_response_text = response_data.get("answer", "I'm sorry, an error occurred.")
            
            # Send the response back to the website widget
            await websocket.send_text(bot_response_text)
            
            # Update the session history
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": bot_response_text})

    except WebSocketDisconnect:
        print(f"Website client {session_id} disconnected.")
        del website_sessions[session_id]
    except Exception as e:
        print(f"An error occurred in the website websocket: {e}")
        del website_sessions[session_id]
