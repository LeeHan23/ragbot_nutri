import os
import docx
from fastapi import UploadFile
from datetime import datetime
from io import BytesIO
from pypdf import PdfReader

# NOTE: This function is now synchronous to be easily callable from other scripts.
def save_uploaded_file_as_text(uploaded_file, destination_folder: str) -> str:
    """
    Reads an uploaded .docx or .pdf file, extracts its text content, 
    and saves it as a .txt file.

    Args:
        uploaded_file: The file object (from Streamlit or FastAPI).
        destination_folder (str): The directory path where the .txt file will be saved.

    Returns:
        str: The full path to the newly created .txt file.
    """
    try:
        # --- Create a unique filename ---
        # Get the original filename without the extension
        base_filename = os.path.splitext(uploaded_file.name)[0]
        sanitized_filename = "".join(c for c in base_filename if c.isalnum() or c in (' ', '_')).rstrip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{sanitized_filename}_{timestamp}.txt"
        save_path = os.path.join(destination_folder, new_filename)

        print(f"Processing upload: '{uploaded_file.name}' -> '{new_filename}'")

        # --- Read file content ---
        file_content = uploaded_file.read()
        content = ""

        # --- Parse file based on its extension ---
        if uploaded_file.name.endswith(".docx"):
            print("Detected .docx file. Parsing with python-docx.")
            doc = docx.Document(BytesIO(file_content))
            full_text = [para.text for para in doc.paragraphs]
            content = '\n'.join(full_text)
        
        elif uploaded_file.name.endswith(".pdf"):
            print("Detected .pdf file. Parsing with pypdf.")
            reader = PdfReader(BytesIO(file_content))
            full_text = [page.extract_text() for page in reader.pages]
            content = '\n'.join(full_text)

        if not content.strip():
            print(f"Warning: The uploaded document '{uploaded_file.name}' appears to be empty or could not be parsed.")

        # --- Save the extracted text to a .txt file ---
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Successfully saved extracted text to: {save_path}")
        return save_path

    except Exception as e:
        print(f"Error processing uploaded file '{uploaded_file.name}': {e}")
        raise
