# 📚 NUB AI Chatbot

A conversational AI chatbot built with **FastAPI**, **LangChain**, **FAISS**, and **OpenAI/OpenRouter** LLMs.  
The chatbot uses a local FAISS vectorstore for retrieval-augmented generation (RAG) and supports multi-session chat with memory.

---

## 🚀 Features

- FastAPI backend for chat requests
- Retrieval-Augmented Generation (RAG) with FAISS vector store
- Conversational memory per session
- HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Supports OpenAI / OpenRouter LLM models
- Simple HTML + CSS + JS chat frontend
- Session reset endpoint

---

## 📂 Project Structure

```
.
├── api/
│   ├── main.py            # FastAPI app (chat API & session management)
│   ├── ingest.py         # Script to load documents into FAISS vectorstore
│
├── static/
│   ├── chat.html         # Chat UI
│   ├── style.css         # UI styling
│   ├── app.js            # Frontend logic for calling the API
│
├── vectorstore/          # FAISS index (created after running ingest.py)
│
├── .env                  # API keys and config (ignored in git)
├── .gitignore
├── README.md
├── requirements.txt
```

---

## 🛠 Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/yourusername/nub-chatbot.git
cd nub-chatbot
```

### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### ⚙️ Environment Variables
Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_or_openrouter_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_MODEL=gpt-4o-mini
```

⚠️ Never commit `.env` to git (it’s in `.gitignore`).

### 📄 Building the Vectorstore
Before running the chatbot, ingest your documents into FAISS:

```bash
python api/ingest.py
```

This will create the `vectorstore/` folder with your embeddings.

### ▶️ Running the API
```bash
uvicorn api.main:app --reload
```

API will be available at:

```
http://localhost:8000
```

Swagger UI for testing:

```
http://localhost:8000/docs
```

### 💬 Testing the Chat API
Example request with `curl`:

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "Hello, what is NUB?", "session_id": "default"}'
```

Example response:

```json
{
  "answer": "NUB stands for ...",
  "sources": ["source1.pdf", "source2.pdf"]
}
```

### 🌐 Using the Chat UI
Place `chat.html`, `style.css`, and `app.js` in the `static/` folder.

FastAPI will serve these files automatically.

Open `http://localhost:8000/static/chat.html` in your browser.

### 🔄 Resetting a Session
```bash
curl -X POST "http://localhost:8000/reset_session" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "default"}'
```

---

## 🚀 Quick Start
To get the chatbot running in under 2 minutes:

1. Clone the repo and navigate to the project folder:
   ```bash
   git clone https://github.com/yourusername/nub-chatbot.git
   cd nub-chatbot
   ```

2. Set up and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API key:
   ```
   OPENAI_API_KEY=your_openai_or_openrouter_key
   OPENAI_API_BASE=https://openrouter.ai/api/v1
   OPENAI_MODEL=gpt-4o-mini
   ```

5. Build the vectorstore:
   ```bash
   python api/ingest.py
   ```

6. Start the FastAPI server:
   ```bash
   uvicorn api.main:app --reload
   ```

7. Open the chat UI in your browser:
   ```
   http://localhost:8000/static/chat.html
   ```

---

## 📝 Requirements File
Example `requirements.txt`:

```
fastapi
uvicorn
python-dotenv
langchain
langchain-community
langchain-openai
langchain-huggingface
faiss-cpu
pydantic
```

---

## 📜 License
MIT License © 2025 roktim ashraful