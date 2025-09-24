import os
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
# This must be done before importing other modules that need the variables.
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Import routers from other modules
from admin import admin_router
from whatsapp_adapter import whatsapp_router # <-- 1. This line must be active

# Create the FastAPI application
app = FastAPI(
    title="Nutritionist AI Chatbot",
    description="A customer service chatbot for a nutritionist company using RAG, FastAPI, and WhatsApp.",
    version="1.0.0"
)

# --- Directory Setup ---
# (This part of your file remains the same)
@app.on_event("startup")
async def startup_event():
    """
    Create necessary data directories on application startup.
    """
    print("Application starting up...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_paths = {
        "knowledge": os.path.join(base_dir, "data/knowledge"),
        "promos": os.path.join(base_dir, "data/promos"),
        "instructions": os.path.join(base_dir, "data/instructions"),
        "sessions": os.path.join(base_dir, "data/sessions")
    }
    for name, path in data_paths.items():
        if not os.path.exists(path):
            print(f"Creating directory: {path}")
            os.makedirs(path)
    print("Startup complete. Data directories are ready.")


# --- Include Routers ---
app.include_router(admin_router, prefix="/admin", tags=["Admin"])
app.include_router(whatsapp_router, prefix="/whatsapp", tags=["WhatsApp"]) # <-- 2. This line must be active


# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint for health check.
    """
    return JSONResponse(
        content={"status": "ok", "message": "Welcome to the Nutritionist AI Chatbot API"}
    )

# --- Main Entry Point ---
if __name__ == "__main__":
    print("Starting Uvicorn server for development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

