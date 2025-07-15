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

# Make port 8501 (the default for Streamlit) available to the host machine
EXPOSE 8501

# Define the command to run your app when the container launches
# This starts the Streamlit UI and makes it accessible outside the container
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
