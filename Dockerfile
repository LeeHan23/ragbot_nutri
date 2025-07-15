# Use an official Python runtime as a parent image
# The 'slim' version is smaller and good for production
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 8501 available to the world outside this container
# This is the default port that Streamlit runs on
EXPOSE 8501

# Define the command to run your app
# This runs the Streamlit UI when the container launches
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
