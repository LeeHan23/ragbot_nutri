import os
from fastapi import UploadFile
from uploader import save_uploaded_file_as_text

# Define the base directory for all user-specific instructions
BASE_INSTRUCTIONS_DIR = os.path.join("data", "instructions")

def save_instruction_file(user_id: str, uploaded_file: UploadFile):
    """
    Saves a user-uploaded .docx file as their specific instruction text.

    This function takes an uploaded file, converts it to a plain text file
    using the utility from the uploader module, and saves it to a
    user-specific directory. Each user will have their own folder,
    and the instruction file will be named consistently to allow for easy retrieval.

    Args:
        user_id (str): The unique identifier for the user.
        uploaded_file (UploadFile): The .docx file uploaded by the user.
    
    Returns:
        str: The path to the saved text file, or None if an error occurred.
    """
    if not user_id or not uploaded_file:
        print("Error: User ID and an uploaded file must be provided.")
        return None

    # Create a directory path that is unique to the user
    user_instructions_dir = os.path.join(BASE_INSTRUCTIONS_DIR, str(user_id))
    
    # Ensure the user-specific directory exists
    os.makedirs(user_instructions_dir, exist_ok=True)
    
    try:
        # Use the existing uploader function to process the file
        # and save it to the user's specific instruction directory.
        # Note: The original uploader creates a unique filename with a timestamp.
        # For instructions, we might want a consistent filename, but for now,
        # we'll leverage the existing safe-handling function.
        saved_path = save_uploaded_file_as_text(uploaded_file, user_instructions_dir)
        
        print(f"Successfully saved new instructions for user '{user_id}' at: {saved_path}")
        return saved_path
        
    except Exception as e:
        print(f"Error saving instruction file for user '{user_id}': {e}")
        return None
