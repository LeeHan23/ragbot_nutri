import os
from dotenv import load_dotenv

# --- Load environment variables from .env file FIRST ---
load_dotenv()

# --- Constants ---
# This section defines key paths and names used throughout the module.
# It ensures that the application knows where to find and store data.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMOS_PATH = os.path.join(BASE_DIR, "data", "promos")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "instructions")


# --- Prompt Loading Functions ---
# These functions are responsible for dynamically loading the AI's persona
# and any promotional information from text files.

def _get_latest_file_content(directory: str) -> str:
    """
    Finds the most recently modified .txt file in a directory, reads it,
    and returns its content as a string. This ensures the bot always
    uses the most up-to-date instructions or promotions.
    """
    try:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return ""
            
        # Get a list of all .txt files in the specified directory
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
        if not files:
            return ""
        
        # Find the file with the most recent modification time
        latest_file = max(files, key=os.path.getmtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
            
    except Exception as e:
        print(f"Error reading from {directory}: {e}")
        return ""

def get_prompts() -> tuple[str, str]:
    """
    Loads the content of the latest persona instructions and promotions files.
    This is called by the RAG chain every time a user sends a message.
    """
    print("Loading persona instructions and latest promotions...")
    
    instructions = _get_latest_file_content(INSTRUCTIONS_PATH)
    promotions = _get_latest_file_content(PROMOS_PATH)
    
    # Provide default text if the instruction or promotion files are missing or empty
    if not instructions:
        instructions = "You are a helpful general assistant. Since no specific instructions were provided, answer questions concisely."
    if not promotions:
        promotions = "There are no special promotions at this time."
        
    # Combine the instructions and promotions into a single block for the AI
    full_instructions = f"{instructions}\n\n[LATEST PROMOTIONS & OFFERS]\n{promotions}"
    
    # Return the combined instructions. The second empty string is for compatibility
    # with the function signature expected in rag.py.
    return full_instructions, ""


# This script is not intended to be run directly. It is imported by other modules.
