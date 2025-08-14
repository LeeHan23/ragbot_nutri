import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Import helper functions from other modules
from uploader import save_uploaded_file_as_text
from base_db_manager import add_pdf_to_base_db

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
    Accepts a PDF file, saves it to the `data/base_documents` directory,
    and then incrementally updates the foundational vector store with its content.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .pdf file.")

    try:
        # Ensure the target directory exists
        os.makedirs(BASE_DOCS_DIR, exist_ok=True)

        # Define the path to save the uploaded PDF
        file_path = os.path.join(BASE_DOCS_DIR, file.filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Saved new knowledge PDF to: {file_path}")

        # Trigger the incremental update of the vector store
        success = add_pdf_to_base_db(file_path)

        if success:
            return JSONResponse(
                status_code=200,
                content={"message": "Successfully added document to the knowledge base."}
            )
        else:
            # If add_pdf_to_base_db returns False, it means an internal error occurred
            raise HTTPException(status_code=500, detail="Failed to process the document and add it to the vector store.")

    except Exception as e:
        # Catch any other exceptions during file handling or processing
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


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
