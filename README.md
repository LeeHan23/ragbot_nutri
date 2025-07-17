# **Personalized AI Nutrition Chatbot**

This project is a sophisticated, multi-tenant AI chatbot application designed to serve as a customizable knowledge assistant. It features a user-friendly web interface built with Streamlit, a powerful backend powered by FastAPI and LangChain, and a flexible knowledge base system using ChromaDB.  
The core architecture is built around a hybrid knowledge model:

1. A **foundational knowledge base** is pre-built from core documents (e.g., official dietary guidelines), providing a consistent source of truth for all users.  
2. Users can upload their own .docx files to create a **private, custom knowledge base** that augments the foundational knowledge, allowing for a highly personalized experience.

## **✨ Features**

* **Conversational AI:** Natural, human-like chat powered by OpenAI's GPT models (gpt-3.5-turbo or gpt-4-turbo).  
* **Retrieval-Augmented Generation (RAG):** The bot answers questions based on a knowledge base created from your documents, ensuring factual and context-aware responses.  
* **Persistent Memory:** Remembers past conversations and user metadata (like visit count) to provide a personalized, stateful experience.  
* **Customizable Knowledge:** Users can upload their own .docx files through the web UI to "train" the bot on their private knowledge.  
* **Hybrid Knowledge System:** Combines a pre-built foundational knowledge base with user-specific custom knowledge.  
* **User-Friendly Interface:** A clean, mobile-friendly web UI built with Streamlit for easy interaction.  
* **Ready for Deployment:** Includes a Dockerfile for easy packaging and deployment on cloud services like Render.

## **🛠️ Tech Stack**

* **Backend:** Python, FastAPI  
* **Frontend (UI):** Streamlit  
* **AI & Orchestration:** LangChain  
* **Language Models:** OpenAI (gpt-3.5-turbo)  
* **Vector Store:** ChromaDB  
* **Document Loading:** PyMuPDF, python-docx, Docx2txt  
* **Deployment:** Docker, Render

## **📂 Project Structure**

rag\_bot/  
├── data/  
│   ├── base\_documents/     \# Core PDFs for the foundational knowledge base  
│   └── users/              \# Dynamically created folders for user-specific documents  
├── chroma\_db/              \# Dynamically created folders for user-specific vector stores  
├── vectorstore\_base/       \# The pre-built foundational vector store  
├── app.py                  \# FastAPI entry point (for optional API extensions)  
├── ui.py                   \# The main Streamlit user interface application  
├── build\_base\_db.py        \# Script to build the foundational knowledge base (run once)  
├── knowledge\_manager.py    \# Manages the creation of user-specific knowledge bases  
├── rag.py                  \# Core RAG and conversational logic  
├── vector\_store.py         \# Logic for retrieving from the vector stores  
├── llm.py                  \# OpenAI model initialization  
├── Dockerfile              \# Instructions for building the application container  
├── requirements.txt        \# Python dependencies  
└── README.md               \# This file

## **🚀 Local Setup and Installation**

Follow these steps to run the application on your local machine for development and testing.

### **1\. Prerequisites**

* Python 3.11+  
* An OpenAI API Key

### **2\. Clone the Repository**

git clone https://github.com/YourUsername/YourRepoName.git  
cd YourRepoName

### **3\. Set Up the Environment**

Create and activate a virtual environment:  
\# Create the virtual environment  
python3 \-m venv venv

\# Activate it (macOS/Linux)  
source venv/bin/activate

\# Or activate it (Windows)  
.\\venv\\Scripts\\activate

Install the required Python packages:  
pip install \-r requirements.txt

### **4\. Configure Environment Variables**

Create a .env file in the root of the project folder and add your OpenAI API key:  
OPENAI\_API\_KEY="sk-..."

### **5\. Build the Foundational Knowledge Base**

This is a one-time setup step that processes the core PDF documents.

* Place your foundational PDF files (e.g., FA-Buku-RNI.pdf) inside the data/base\_documents/ folder.  
* Run the build script from your terminal:

python build\_base\_db.py

This will take several minutes to complete. It will create a vectorstore\_base folder in your project directory.

### **6\. Run the Application**

Start the Streamlit user interface:  
streamlit run ui.py

Your web browser should open to http://localhost:8501, where you can now interact with the chatbot.

## **☁️ Deployment to Render**

This application is configured for easy deployment on Render using Docker.

1. **Push to GitHub:** Ensure your latest code, including the Dockerfile and the pre-built vectorstore\_base folder, is pushed to your GitHub repository.  
2. **Create a Web Service on Render:**  
   * Connect your GitHub repository.  
   * Choose the **Docker** runtime.  
   * In the **Settings**, set the **Start Command** to: ./startup.sh  
3. **Add Environment Variables:**  
   * In the **Environment** tab, add your OPENAI\_API\_KEY as a secret.  
4. **Create a Persistent Disk:**  
   * In the **Disks** tab, create a new disk.  
   * Set the **Mount Path** to /data.  
   * Choose a size (e.g., 1 GB).  
5. **Deploy:** Render will build the Docker image and start your service. The startup.sh script will automatically build the foundational knowledge base on the persistent disk the first time it runs.
