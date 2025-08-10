import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

# Import vector store and embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Define VECTOR_DIR (make sure this path is correct for your environment)
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

# Initialize embeddings (HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vectorstore with deserialization allowed
db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)

# Load OpenRouter/OpenAI API credentials
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # corrected model name

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

# Initialize the LLM client
llm = ChatOpenAI(
    temperature=0,
    model=OPENAI_MODEL,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
)

# Create a conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 6}),
    return_source_documents=True,
)

app = FastAPI(title="NUB Chatbot API")


# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; you can limit to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Session storage for chat history: session_id -> list of (user_question, bot_answer)
SESSIONS: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


INSTRUCTION = """
You are a helpful assistant of Northern University of Bangladesh (NUB). Automatically detect the language of the user's question.
- If the question is in Bangla, answer in fluent Bangla.
- If the question is in English or any other language, answer in English.
- Tone of the answer should be friendly and conversational.
- If the user asks a question that is not related to NUB, answer in English in a friendly manner.
- If the question is logical, answer in a logical and mathematical manner without any personal opinion.
"""

def prepend_instruction(question):
    return INSTRUCTION + "\n\nUser question: " + question

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    history = SESSIONS.get(session_id, [])

    try:
        result = qa_chain.invoke({
            "question": prepend_instruction(req.question),
            "chat_history": history
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chain invocation error: {str(e)}")

    answer = result.get("answer") or result.get("result") or ""
    src_docs = result.get("source_documents") or []

    # Append current question and answer to session history
    history.append((req.question, answer))
    SESSIONS[session_id] = history[-20:]  # keep last 20 entries

    # Extract unique sources from documents
    sources = []
    for doc in src_docs:
        src = None
        if hasattr(doc, "metadata") and doc.metadata:
            src = doc.metadata.get("source") or ""
        if not src and hasattr(doc, "page_content"):
            src = doc.page_content[:120]
        if src and src not in sources:
            sources.append(src)

    return ChatResponse(answer=answer, sources=sources)

@app.post("/reset_session")
async def reset_session(payload: dict):
    session_id = payload.get("session_id", "default")
    SESSIONS.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}
