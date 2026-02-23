import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware
from difflib import SequenceMatcher
import glob
load_dotenv()

# Import vector store and embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Define VECTOR_DIR (make sure this path is correct for your environment)
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

# Load faculty members data from all faculty files
def load_faculty_data():
    """Load all faculty members from data files"""
    faculty_list = []
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # Find all faculty data files
    faculty_files = glob.glob(os.path.join(data_dir, "faculty*"))
    
    for file_path in faculty_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by "Name:" to extract individual faculty records
                records = content.split("Name:")[1:]  # Skip header
                for record in records:
                    name_line = record.split('\n')[0].strip()
                    if name_line:
                        faculty_list.append({
                            'name': name_line,
                            'name_lower': name_line.lower(),
                            'file': os.path.basename(file_path)
                        })
        except Exception as e:
            print(f"Error loading faculty file {file_path}: {str(e)}")
    
    return faculty_list

# Load faculty data on startup
FACULTY_MEMBERS = load_faculty_data()

def similarity_ratio(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_similar_faculty(query_name: str, threshold: float = 0.6) -> List[Dict]:
    """
    Find similar faculty members based on query name
    Returns list of similar faculty members with similarity scores
    """
    matches = []
    
    for faculty in FACULTY_MEMBERS:
        similarity = similarity_ratio(query_name, faculty['name'])
        if similarity >= threshold:
            matches.append({
                'name': faculty['name'],
                'file': faculty['file'],
                'similarity': similarity
            })
    
    # Sort by similarity score (highest first)
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return matches


def get_faculty_full_record(faculty_entry: Dict) -> str:
    """Return the full text block for a faculty member from its source file.
    If not found, return an empty string."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    file_path = os.path.join(data_dir, faculty_entry.get('file', ''))
    if not os.path.isfile(file_path):
        return ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return ""

    # Split records by 'Name:' and find the block that matches the faculty name
    records = content.split("Name:")
    for rec in records:
        rec = rec.strip()
        if not rec:
            continue
        first_line = rec.splitlines()[0].strip()
        # Compare names with a stricter similarity
        if similarity_ratio(first_line, faculty_entry.get('name', '')) >= 0.85 or first_line.lower() == faculty_entry.get('name', '').lower():
            return "Name: " + rec.strip()

    return ""

def detect_faculty_query(question: str) -> Tuple[bool, str]:
    """
    Detect if question is about faculty members
    Returns (is_faculty_query, extracted_name)
    """
    # Keywords that indicate faculty query
    faculty_keywords = ['faculty', 'teacher', 'professor', 'instructor', 'sir', 'madam', 'miss', 'mr', 'dr', 'prof']
    question_lower = question.lower()
    
    # Check for faculty keywords
    is_faculty_query = any(keyword in question_lower for keyword in faculty_keywords)
    
    if not is_faculty_query:
        return False, ""
    
    # Words to filter out (common question words)
    stop_words = ['contact', 'details', 'information', 'about', 'tell', 'me', 'info', 'of', 'from', 
                  'who', 'what', 'where', 'when', 'how', 'is', 'are', 'the', 'a', 'an', 'din', 'do', 'bolo']
    
    # Extract potential name from question
    # Remove faculty keywords
    name = question
    for keyword in faculty_keywords:
        name = re.sub(r'\b' + keyword + r'\b\s*', '', name, flags=re.IGNORECASE)
    
    # Remove special characters but keep spaces
    name = re.sub(r'[?!.,;:\-]', '', name).strip()
    
    # Split into words and filter out stop words
    words = name.split()
    filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 1]
    
    # Take only first 4 words (typically a name is 1-4 words)
    name = ' '.join(filtered_words[:4]).strip()
    
    return True, name if name else ""



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

def detect_language(text: str) -> str:
    """Detect language of the input text.
    Returns: 'bn' for Bangla (Bengali script), 'bn_lat' for Banglish (romanized Bangla), or 'en' for English/other.
    """
    if not text:
        return "en"

    # If contains any Bengali (Bangla) Unicode characters, treat as Bangla
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"

    # Heuristic for Banglish / romanized Bangla: presence of common Bangla words in Latin script
    banglish_tokens = [
        "ami", "tumi", "ke", "kemon", "kono", "katha", "kothai", "koro", "kore",
        "bolo", "dao", "din", "ami", "apni", "bhai", "dida", "mone", "ki", "na",
    ]
    text_lower = text.lower()
    token_matches = sum(1 for t in banglish_tokens if t in text_lower)
    if token_matches >= 1:
        return "bn_lat"

    return "en"

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
You are the official AI assistant for Northern University of Bangladesh (NUB). Your primary role is to provide accurate, helpful, and friendly assistance to students, faculty, staff, and visitors.

LANGUAGE HANDLING:
- Automatically detect the language of every user question
- For Bangla questions: Comprehend in English, then respond in fluent, natural Bangla using proper grammar and cultural context
- For English or other languages: Comprehend and respond in clear, professional English
- Maintain consistency: Do not switch languages mid-response unless explicitly requested

RESPONSE TONE & STYLE:
- Always be warm, welcoming, and conversational
- Use a respectful yet approachable tone appropriate for an educational institution
- Show empathy and patience, especially with confused or frustrated users
- Avoid overly formal or robotic language while maintaining professionalism

CONTENT GUIDELINES:
- NUB-related questions: Provide accurate, detailed information about courses, admissions, campus facilities, policies, events, faculty, departments, and student services
- Faculty member queries: If a specific faculty name is provided but no exact match is found, provide a list of similar faculty members with their information
- Non-NUB questions: Politely acknowledge the question and provide a helpful response in English, but gently remind users that your primary expertise is NUB-related matters
- Logical/mathematical questions: Provide step-by-step reasoning, show your work, and use objective analysis without personal opinions
- Sensitive topics: Handle complaints, concerns, or difficult questions with care and direct users to appropriate university resources when necessary
- If a user asks about a topic that is not covered in the current data files, explain that the bot can learn from new documents and invite them to submit additional information via the `/add_data` endpoint or by adding files to the `data/` folder. The system will automatically ingest new data so the model can answer future queries.  When possible, use your general external knowledge to provide a helpful answer but make it clear which parts come from your own reasoning vs. the stored documents.

ACCURACY & RELIABILITY:
- Only provide information you are certain about regarding NUB
- If uncertain, clearly state "I don't have confirmed information about this" and suggest contacting official NUB channels
- Never fabricate details about policies, dates, fees, or official procedures
- When providing links or contact information, ensure they are accurate or indicate they should be verified

PROHIBITED ACTIONS:
- Do not make up information about NUB
- Do not provide personal opinions on university policies or controversies
- Do not share student or faculty personal information
- Do not engage in arguments or debates about the university's reputation or rankings
"""

def prepend_instruction(question: str, language: str = "en") -> str:
    """Prepare the instruction prompt including language guidance."""
    lang_directive = ""
    if language == "bn":
        lang_directive = "\n\nPLEASE RESPOND IN BANGLA (বাংলা) - understand the user's question and answer in fluent Bangla."
    elif language == "bn_lat":
        lang_directive = "\n\nUser wrote in Banglish (romanized Bangla). Comprehend it as Bangla and respond in fluent Bangla (বাংলা)."
    else:
        lang_directive = "\n\nPLEASE RESPOND IN ENGLISH - answer clearly and professionally in English."

    return INSTRUCTION + lang_directive + "\n\nUser question: " + question

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    history = SESSIONS.get(session_id, [])
    # Detect language for this request
    lang = detect_language(req.question)

    # Check if this is a faculty query and prepare candidate matches
    similar_faculty: List[Dict] = []
    is_faculty_query, extracted_name = detect_faculty_query(req.question)
    enhanced_question = req.question

    if is_faculty_query and extracted_name:
        similar_faculty = find_similar_faculty(extracted_name, threshold=0.5)
        if similar_faculty:
            faculty_list_text = "Found faculty members similar to your query:\n"
            faculty_details_text = ""
            for idx, faculty in enumerate(similar_faculty[:10], 1):  # Top 10 matches
                faculty_list_text += f"{idx}. {faculty['name']}\n"
                details = get_faculty_full_record(faculty)
                if details:
                    faculty_details_text += f"{idx}. {faculty['name']}\n{details}\n---\n"
                else:
                    # If no detailed block found, at least list the name
                    faculty_details_text += f"{idx}. {faculty['name']}\n"

            # Prefer detailed records when available, otherwise use the simple list
            enhanced_prefix = faculty_details_text.strip() if faculty_details_text.strip() else faculty_list_text
            enhanced_question = enhanced_prefix + "\n\n" + req.question

    try:
        result = qa_chain.invoke({
            "question": prepend_instruction(enhanced_question, language=lang),
            "chat_history": history
        })
    except Exception as e:
        print(f"Chain invocation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chain invocation error: {str(e)}")

    answer = result.get("answer") or result.get("result") or ""
    src_docs = result.get("source_documents") or []
    
    # If the chain returned a negative/no-info answer and we have similar faculty, prepend detailed info
    if is_faculty_query and extracted_name and similar_faculty:
        neg_phrases = [
            "don't have", "i don't have", "no information", "no contact", "can't find", "cannot find",
            "not found", "no record", "no details", "no data", "sorry, i don't"
        ]
        if any(p in answer.lower() for p in neg_phrases):
            if lang.startswith("bn"):
                faculty_response = f"আপনি যে নামটি দিয়েছেন '{extracted_name}', তার সঠিক মিল আমি পাইনি, কিন্তু সম্পর্কিত শিক্ষক(গণের) তথ্য নীচে দেয়া হলো:\n\n"
            else:
                faculty_response = f"The name '{extracted_name}' that you provided, I did not find an exact match but here is the related teachers info:\n\n"

            for idx, faculty in enumerate(similar_faculty[:10], 1):
                faculty_response += f"{idx}. {faculty['name']}\n"

            answer = faculty_response + "\n\n" + answer

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


# --- Data ingestion helpers & API -------------------------------------------------

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class AddDataRequest(BaseModel):
    filename: str  # should end in .txt
    content: str


def add_document_to_vectorstore(filename: str, text: str):
    """Split a piece of text, embed it and append it to the existing FAISS store."""
    # create document and chunk it
    docs = [Document(page_content=text, metadata={"source": filename})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    # build embeddings (reuse same model as on startup)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    global db
    db.add_documents(chunks, emb)
    db.save_local(VECTOR_DIR)


@app.post("/add_data")
async def add_data(req: AddDataRequest):
    """API endpoint to store a new text file in the data folder and update the vectorstore.

    This lets the chatbot grow over time; after adding documents the model will be
    able to answer questions about them.  Clients can POST a JSON body with the file
    name and its contents.  Only .txt files are accepted.  (In a production system
    you'd add authentication and sanitise filenames.)
    """
    # validate filename
    if not req.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="filename must end with .txt")
    # prevent path traversal
    safe_name = os.path.basename(req.filename)
    path = os.path.join(os.path.dirname(__file__), "..", "data", safe_name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(req.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    # update vectorstore immediately so that subsequent questions can see the content
    try:
        add_document_to_vectorstore(safe_name, req.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating vectorstore: {e}")

    return {"status": "ok", "filename": safe_name}
