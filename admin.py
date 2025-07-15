import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Import helper functions from other modules
from uploader import save_uploaded_file_as_text

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
    Accepts a .docx or .pdf file and saves its content as a .txt file
    in the `data/knowledge` directory.
    
    NOTE: After uploading, you must manually run 'python create_database.py' 
    from your terminal to update the bot's knowledge.

    - **file**: The .docx or .pdf file to be uploaded.
    """
    if not (file.filename.endswith(".docx") or file.filename.endswith(".pdf")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx or .pdf file.")
    
    try:
        saved_path = await save_uploaded_file_as_text(file, KNOWLEDGE_PATH)
        return JSONResponse(
            status_code=200,
            content={
                "message": "Knowledge file uploaded successfully. Run 'python create_database.py' to re-index.", 
                "filepath": saved_path
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@admin_router.post("/upload/promo", summary="Upload a promotions file")
async def upload_promo_file(file: UploadFile = File(...)):
    """
    Accepts a .docx file and saves its content as a .txt file
    in the `data/promos` directory.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")

    try:
        saved_path = await save_uploaded_file_as_text(file, PROMOS_PATH)
        return JSONResponse(
            status_code=200,
            content={"message": "Promotions file uploaded successfully.", "filepath": saved_path}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@admin_router.post("/upload/instructions", summary="Upload a behavior instructions file")
async def upload_instructions_file(file: UploadFile = File(...)):
    """
    Accepts a .docx file and saves its content as a .txt file
    in the `data/instructions` directory.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")

    try:
        saved_path = await save_uploaded_file_as_text(file, INSTRUCTIONS_PATH)
        return JSONResponse(
            status_code=200,
            content={"message": "Instructions file uploaded successfully.", "filepath": saved_path}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")
