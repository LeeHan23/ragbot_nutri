import os
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Import helper functions from other modules
from uploader import save_docx_as_text
from vector_store import reindex_knowledge_base

# --- FastAPI Router ---
admin_router = APIRouter()

# --- Constants for data paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "data", "knowledge")
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")


# --- File Upload Endpoints ---

@admin_router.post("/upload/knowledge", summary="Upload a knowledge base file")
async def upload_knowledge_file(file: UploadFile = File(...)):
    """
    Accepts a `.docx` file and saves its content as a `.txt` file
    in the `data/knowledge` directory. This content will be used
    during the next re-indexing process.

    - **file**: The `.docx` file to be uploaded.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")
    
    try:
        # The uploader function handles the conversion and saving
        saved_path = await save_docx_as_text(file, KNOWLEDGE_PATH)
        return JSONResponse(
            status_code=200,
            content={"message": "Knowledge file uploaded successfully.", "filepath": saved_path}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@admin_router.post("/upload/promo", summary="Upload a promotions file")
async def upload_promo_file(file: UploadFile = File(...)):
    """
    Accepts a `.docx` file and saves its content as a `.txt` file
    in the `data/promos` directory. The content of the most recent
    file in this directory will be dynamically injected into the prompt.

    - **file**: The `.docx` file containing promotional text.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")

    try:
        saved_path = await save_docx_as_text(file, PROMOS_PATH)
        return JSONResponse(
            status_code=200,
            content={"message": "Promotions file uploaded successfully.", "filepath": saved_path}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@admin_router.post("/upload/instructions", summary="Upload a behavior instructions file")
async def upload_instructions_file(file: UploadFile = File(...)):
    """
    Accepts a `.docx` file and saves its content as a `.txt` file
    in the `data/instructions` directory. The content of the most recent
    file will be used to guide the chatbot's behavior.

    - **file**: The `.docx` file containing behavior instructions.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")

    try:
        saved_path = await save_docx_as_text(file, INSTRUCTIONS_PATH)
        return JSONResponse(
            status_code=200,
            content={"message": "Instructions file uploaded successfully.", "filepath": saved_path}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


# --- Re-indexing Endpoint ---

def run_reindexing():
    """Wrapper function to be run in the background."""
    print("Background task started: Re-indexing knowledge base.")
    try:
        reindex_knowledge_base()
        print("Background task finished: Re-indexing successful.")
    except Exception as e:
        print(f"Background task error: Re-indexing failed with error: {e}")

@admin_router.post("/reindex", summary="Trigger re-indexing of the vector store")
async def trigger_reindexing(background_tasks: BackgroundTasks):
    """
    Triggers a full re-indexing of the knowledge base. This process
    clears the old vector store and creates a new one based on all
    the `.txt` files currently in the `data/knowledge` directory.

    The process runs in the background to avoid blocking the API.
    """
    try:
        print("Received request to re-index knowledge base.")
        # Run the potentially long-running re-indexing task in the background
        background_tasks.add_task(run_reindexing)
        
        return JSONResponse(
            status_code=202,  # Accepted
            content={
                "message": "Re-indexing process has been started in the background. "
                           "Check server logs for completion status."
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start re-indexing process: {e}")
