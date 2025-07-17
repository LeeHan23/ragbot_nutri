# Use an official, lightweight Python runtime as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container first
COPY requirements.txt .

# Install the Python dependencies
# --no-cache-dir makes the final image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# --- NEW: Make the startup script executable ---
# This command gives the script permission to be run by the system
RUN chmod +x startup.sh

# Make port 8501 (the default for Streamlit) available to the host machine
EXPOSE 8501

# --- MODIFIED: Define the new startup command ---
# This now runs our smart startup script instead of directly running Streamlit
CMD ["./startup.sh"]
