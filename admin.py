import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Import helper functions from the new consolidated manager
from knowledge_manager import add_pdf_to_base_db, save_instruction_file
from uploader import save_uploaded_file_as_text

# --- FastAPI Router ---
admin_router = APIRouter()

# --- Constants for data paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DOCS_DIR = os.path.join(BASE_DIR, "data", "base_documents")
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")


# --- File Upload Endpoints ---

@admin_router.post("/add-to-knowledge-base", summary="Add a PDF to the foundational knowledge base")
async def add_to_knowledge_base(file: UploadFile = File(...)):
    """
    Accepts a PDF file, saves it, and incrementally updates the foundational vector store.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .pdf file.")

    try:
        os.makedirs(BASE_DOCS_DIR, exist_ok=True)
        file_path = os.path.join(BASE_DOCS_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if add_pdf_to_base_db(file_path):
            return JSONResponse(status_code=200, content={"message": "Successfully added document to the knowledge base."})
        else:
            raise HTTPException(status_code=500, detail="Failed to process the document.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@admin_router.post("/upload/promo", summary="Upload a global promotions file")
async def upload_promo_file(file: UploadFile = File(...)):
    """
    Saves a .docx file content as a global promotions .txt file.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")
    try:
        saved_path = await save_uploaded_file_as_text(file, PROMOS_PATH)
        return JSONResponse(status_code=200, content={"message": "Promotions file uploaded successfully.", "filepath": saved_path})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@admin_router.post("/upload/instructions/global", summary="Upload a global instructions file")
async def upload_global_instructions_file(file: UploadFile = File(...)):
    """
    Saves a .docx file content as the global instructions .txt file.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")
    try:
        saved_path = await save_uploaded_file_as_text(file, INSTRUCTIONS_PATH)
        return JSONResponse(status_code=200, content={"message": "Global instructions file uploaded successfully.", "filepath": saved_path})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

@admin_router.post("/upload/instructions/{user_id}", summary="Upload user-specific instructions")
async def upload_user_instructions_file(user_id: str, file: UploadFile = File(...)):
    """
    Saves a .docx file as a specific user's persona/instruction file.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")
    try:
        saved_path = save_instruction_file(user_id, file)
        if saved_path:
            return JSONResponse(status_code=200, content={"message": f"Instructions for user {user_id} uploaded successfully.", "filepath": saved_path})
        else:
            raise HTTPException(status_code=500, detail="Failed to save user instructions.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")
