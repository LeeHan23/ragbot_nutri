# **Personalized AI Nutrition Chatbot**

This project is a sophisticated, multi-tenant AI chatbot application designed to serve as a customizable knowledge assistant. It features a user-friendly web interface built with Streamlit, a powerful backend powered by FastAPI and LangChain, and a flexible knowledge base system using ChromaDB.
The core architecture is built around a hybrid knowledge model:

1.  A **foundational knowledge base** is pre-built from core documents (e.g., official dietary guidelines), providing a consistent source of truth for all users.
2.  Users can upload their own .docx files to create a **private, custom knowledge base** that augments the foundational knowledge, allowing for a highly personalized experience.

## ✨ Features

* **Conversational AI:** Natural, human-like chat powered by OpenAI's GPT models (gpt-3.5-turbo or gpt-4-turbo).
* **Retrieval-Augmented Generation (RAG):** The bot answers questions based on a knowledge base created from your documents, ensuring factual and context-aware responses.
* **User Authentication & Management:** Secure sign-up and login system with first-login key verification.
* **Customizable Knowledge:** Users can upload their own .docx files through the web UI to "train" the bot on their private knowledge or persona.
* **Hybrid Knowledge System:** Combines a pre-built foundational knowledge base with user-specific custom knowledge.
* **User-Friendly Interface:** A clean, mobile-friendly web UI built with Streamlit for easy interaction.
* **Ready for Deployment:** Includes a Dockerfile for easy packaging and deployment on cloud services like Render.

---

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI
* **Frontend (UI):** Streamlit
* **Database (Users):** SQLite
* **AI & Orchestration:** LangChain
* **Language Models:** OpenAI (gpt-3.5-turbo)
* **Vector Store:** ChromaDB
* **Document Loading:** PyMuPDF, python-docx, Docx2txt
* **Deployment:** Docker, Render

---

## 📂 Project Structure

(Project Structure remains the same)

---

## 🚀 Local Setup and Installation

Follow these steps to run the application on your local machine for development and testing.

### 1. Prerequisites

* Python 3.11+
* An OpenAI API Key

### 2. Clone the Repository

```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName