import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Validate Environment Variable ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")

# --- Language Model Initialization ---
def get_llm():
    """
    Initializes and returns the ChatOpenAI model instance.

    This function configures the language model to be used for generating responses.
    We use gpt-3.5-turbo as it is widely available and effective for this task.
    
    Returns:
        An instance of ChatOpenAI.
    """
    try:
        # Using gpt-3.5-turbo is more compatible with free-tier and new API keys.
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
            openai_api_key=OPENAI_API_KEY
        )
        return llm
    except Exception as e:
        print(f"Error initializing ChatOpenAI model: {e}")
        # This will prevent the application from starting if the LLM can't be initialized
        raise
