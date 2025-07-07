import os
import docx
import aiofiles
from fastapi import UploadFile
from datetime import datetime

async def save_docx_as_text(file: UploadFile, destination_folder: str) -> str:
    """
    Reads a .docx file, extracts its text content, and saves it as a .txt file.

    The function ensures the filename is unique by appending a timestamp,
    preventing overwrites and maintaining a history of uploads.

    Args:
        file (UploadFile): The uploaded .docx file object from FastAPI.
        destination_folder (str): The directory path where the .txt file will be saved.

    Returns:
        str: The full path to the newly created .txt file.

    Raises:
        Exception: If there is an error during file processing or saving.
    """
    try:
        # --- Create a unique filename ---
        # Get the original filename without the extension
        base_filename = os.path.splitext(file.filename)[0]
        # Sanitize filename (optional, but good practice)
        sanitized_filename = "".join(c for c in base_filename if c.isalnum() or c in (' ', '_')).rstrip()
        # Create a timestamp string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Construct the new .txt filename
        new_filename = f"{sanitized_filename}_{timestamp}.txt"
        save_path = os.path.join(destination_folder, new_filename)

        print(f"Processing upload: '{file.filename}' -> '{new_filename}'")

        # --- Read and Parse the .docx file ---
        # Read the file content into memory. This is necessary because python-docx
        # needs a file-like object, and UploadFile provides a stream.
        file_content = await file.read()
        
        # Use a file-like object in memory (BytesIO) to read the content
        from io import BytesIO
        doc = docx.Document(BytesIO(file_content))
        
        # Extract text from all paragraphs in the document
        full_text = [para.text for para in doc.paragraphs]
        content = '\n'.join(full_text)

        if not content.strip():
            print(f"Warning: The uploaded document '{file.filename}' appears to be empty.")

        # --- Save the extracted text to a .txt file ---
        # Use aiofiles for asynchronous file writing to avoid blocking the event loop
        async with aiofiles.open(save_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        print(f"Successfully saved extracted text to: {save_path}")
        return save_path

    except Exception as e:
        print(f"Error processing .docx file '{file.filename}': {e}")
        # Re-raise the exception to be handled by the calling admin endpoint
        raise
