import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Validate Environment Variable ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")

# --- Define the desired output structure ---
# We want the LLM to return a JSON object with a single key "messages"
# which contains a list of strings.
class Conversation(BaseModel):
    messages: List[str] = Field(description="A list of short, sequential messages to send to the user.")

# --- Language Model Initialization ---
def get_llm():
    """Initializes and returns the ChatOpenAI model instance."""
    try:
        llm = ChatOpenAI(
            # Using gpt-4 is recommended for better instruction following and JSON formatting
            model_name="gpt-4-turbo", 
            temperature=0.7,
            max_tokens=500,
            openai_api_key=OPENAI_API_KEY,
            # Important: Enable JSON mode
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        return llm
    except Exception as e:
        print(f"Error initializing ChatOpenAI model: {e}")
        raise

# --- Main LLM Interaction Function ---
async def get_llm_conversation(prompt: str) -> List[str]:
    """
    Sends a prompt to the LLM and returns a list of conversational messages.

    This function uses a chain that forces the LLM to output a JSON object
    matching the 'Conversation' model.

    Args:
        prompt (str): The fully formatted prompt.

    Returns:
        List[str]: A list of message strings from the language model.
    """
    try:
        llm = get_llm()
        
        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=Conversation)

        prompt_template = ChatPromptTemplate.from_template(
            template="{user_prompt}\n\n{format_instructions}\n",
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        # Create a simple chain: prompt -> model -> JSON output parser
        chain = prompt_template | llm | parser
        
        print("Invoking LLM to generate conversational messages...")
        response_data = await chain.ainvoke({"user_prompt": prompt})
        
        # Extract the list of messages from the parsed data
        return response_data.get("messages", ["I'm sorry, I had trouble generating a response."])

    except Exception as e:
        print(f"An error occurred while communicating with the LLM: {e}")
        raise
