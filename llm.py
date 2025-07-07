import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- Load Environment Variables ---
# This ensures the OpenAI API key is loaded securely from a .env file
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
    It defaults to 'gpt-3.5-turbo' for a balance of performance and cost,
    but can be easily switched to 'gpt-4' or other models.

    Returns:
        An instance of ChatOpenAI.
    """
    try:
        # For higher quality responses, you can change the model to "gpt-4-turbo" or "gpt-4"
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,  # Adjust for more creative or factual responses
            max_tokens=500,
            openai_api_key=OPENAI_API_KEY
        )
        return llm
    except Exception as e:
        print(f"Error initializing ChatOpenAI model: {e}")
        # This will prevent the application from starting if the LLM can't be initialized
        raise

# --- Main LLM Interaction Function ---
async def get_llm_response(prompt: str) -> str:
    """
    Sends a prompt to the LLM and returns the generated response as a string.

    This function uses a simple chain: Prompt -> LLM -> String Output Parser.

    Args:
        prompt (str): The fully formatted prompt to be sent to the language model.

    Returns:
        str: The text response from the language model.
    """
    try:
        llm = get_llm()
        
        # Although our rag.py prepares a detailed string prompt,
        # we still wrap it for the Chat model.
        # This approach is simple and effective for this use case.
        # For more complex chat histories, you would use a message list.
        prompt_template = ChatPromptTemplate.from_template("{user_prompt}")
        
        # Create a simple chain: prompt -> model -> output parser
        chain = prompt_template | llm | StrOutputParser()
        
        print("Invoking LLM to generate response...")
        # Asynchronously invoke the chain with the prompt
        response = await chain.ainvoke({"user_prompt": prompt})
        
        return response

    except Exception as e:
        print(f"An error occurred while communicating with the LLM: {e}")
        # The calling function in rag.py will handle this error and
        # provide a user-friendly fallback message.
        raise

