#!/bin/bash

# Define the path to the base vector store on the persistent disk
# The mount path for the disk is /data as configured on Render
BASE_DB_PATH="/data/vectorstore_base"

# Check if the base vector store directory exists
if [ ! -d "$BASE_DB_PATH" ]; then
  echo "--- Foundational knowledge base not found. Building it now... ---"
  
  # Run the Python script to build the base database
  # We point the script to the persistent disk location
  python build_base_db.py
  
  echo "--- Foundational knowledge base built successfully. ---"
else
  echo "--- Foundational knowledge base already exists. Skipping build. ---"
fi

# After the check, start the Streamlit application
echo "--- Starting Streamlit UI... ---"
streamlit run ui.py --server.port=8501 --server.address=0.0.0.0
