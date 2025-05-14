# 🧠 Conversational RAG with LangChain, Groq, and Gemini

This project demonstrates how to build a production-ready Conversational Retrieval-Augmented Generation (RAG) system using LangChain, Groq (LLaMA 3.1), and Gemini Embeddings. It includes context-aware memory handling, persistent chat sessions, and real-time document querying via local embeddings.

## 🚀 Features

- Retrieval-Augmented Generation (RAG) pipeline
- Memory-enabled chat using session IDs
- Text splitting and embedding via Gemini
- Fast inference via LLaMA-3.1 on Groq
- Uses LangChain’s Chroma for vector storage
- Query-aware contextual reformulation

---

## 🧰 Tech Stack

- Python 3.10+
- LangChain
- Groq (LLaMA 3.1)
- Gemini Embeddings (Google)
- ChromaDB (in-memory vector DB)
- dotenv for API key management

---

## 📁 Directory Structure

├── data/
│ └── be-good.txt # Source document
├── main.py # Main logic
├── .env # Contains GOOGLE_API_KEY
├── requirements.txt # All dependencies

---

## 🧪 How It Works

1. Loads a plain text document and splits it into chunks.
2. Generates vector embeddings using Gemini (`embedding-001`).
3. Stores them in a Chroma vector store.
4. Sets up a Chat RAG chain using Groq's LLaMA-3.1 model.
5. Uses LangChain's memory to maintain a contextual chat thread.
6. Allows question-answering with chat history awareness.

---

## 🧵 Example Flow
    > What is this article about?
    AI: The article discusses how to be a better person...

    > What was my previous question?
    AI: Your previous question was asking what the article was about.
## 🔑 Environment Variables
    Create a .env file in your root directory with the following:
    GOOGLE_API_KEY=your_google_api_key_here
## 🧭 Setup Instructions
    Clone the repo
    
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    Install dependencies
    pip install -r requirements.txt
    Add your API key to the .env file
    
    Run the script
    python main.py
  ## 📦 Dependencies
    Add this to requirements.txt if not already:
    langchain
    langchain-groq
    langchain-google-genai
    langchain-chroma
    langchain-community
    python-dotenv

## 🧠 Memory & Session Support
This project uses LangChain’s RunnableWithMessageHistory to support:

Stateful conversations

Unique session IDs

Persistent chat history via ChatMessageHistory

## 📄 Reference
LangChain Docs

Groq LLaMA Models

Google Gemini API

## 📬 License
MIT License — Feel free to use, modify, and distribute.

## 🙋‍♂️ Author
Built with ❤️ by Mihir Raj Chowdhury
