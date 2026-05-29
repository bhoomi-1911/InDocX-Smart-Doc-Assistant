
import streamlit as st
import time
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InDocX · RAG Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
#  API KEY — backend only
#  Set before running:  set INDOCX_API_KEY=your_key_here
# =════════════════════════════════════════════════════════════════════════════
INDOCX_API_KEY = os.environ.get("INDOCX_API_KEY", "AIzaSyBSFemy5HUFSkNy_UUqVnIyci2BLZpUjKI")


# ═════════════════════════════════════════════════════════════════════════════
#  INTEGRATION STUBS
# =════════════════════════════════════════════════════════════════════════════

def process_uploaded_pdf(file_bytes, filename):
    import io, re, PyPDF2
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # Member B: Extract text
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text and len(page_text.strip()) > 10:
            text += page_text + "\n"

    if len(text.strip()) < 50:
        from pdf2image import convert_from_bytes
        import pytesseract
        images = convert_from_bytes(file_bytes)
        for image in images:
            text += pytesseract.image_to_string(image, lang="eng") + "\n"

    # Member B: Clean and chunk
    text = re.sub(r"\s+", " ", text).strip()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    # Member C: Embed and build FAISS vector store
    # langchain-google-genai v2+ requires GOOGLE_API_KEY env var only
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=INDOCX_API_KEY,
        task_type="retrieval_document",
    )
    documents = [
        Document(page_content=chunk, metadata={"chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings_model,
    )
    return vector_store

def get_rag_response(question, vector_store, model, answer_mode="Strict Document Mode"):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate

    # Member C: Get relevant chunks with scores
    results = vector_store.similarity_search_with_score(question, k=3)
    context = "\n\n---\n\n".join(doc.page_content for doc, score in results)

    if not context or context.strip() == "":
        return "I don't know based on the provided document.", [], []

    # Member D: Generate answer using Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=INDOCX_API_KEY,
        temperature=0.3,
    )

    if answer_mode == "Strict Document Mode":
        system_instructions = (
            "You are a Smart Document Assistant using RAG.\n"
            "Answer ONLY from the provided document context.\n"
            "Rules:\n"
            "1. Use only the given context.\n"
            "2. Do not use outside knowledge.\n"
            "3. If answer is missing, say: I don't know based on the provided document.\n"
            "4. Keep answers clear and concise.\n"
        )
    else:  # Flexible Mode
        system_instructions = (
            "You are a Smart Document Assistant using RAG.\n"
            "Use the provided document context as the primary source.\n"
            "Rules:\n"
            "1. Base your answer primarily on the given context.\n"
            "2. You may supplement with general knowledge if needed.\n"
            "3. Clearly label any outside information with 'General explanation:' prefix.\n"
            "4. Keep answers clear and well-structured.\n"
        )

    prompt = ChatPromptTemplate.from_template(
        "{system_instructions}\n"
        "Document Context:\n{context}\n\n"
        "User Question:\n{question}\n\n"
        "Final Answer:"
    )

    chain = prompt | llm
    response = chain.invoke({
        "system_instructions": system_instructions,
        "context": context,
        "question": question
    })

    answer = response.content if not isinstance(response.content, list) else response.content[0]["text"]

    # Return answer + evidence
    chunks = [doc.page_content for doc, score in results]
    scores = [float(score) for doc, score in results]
    return answer, chunks, scores


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE: FLASHCARD GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_flashcards(vector_store):
    from langchain_google_genai import ChatGoogleGenerativeAI
    import json

    results = vector_store.similarity_search_with_score("key concepts definitions important terms", k=3)
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=INDOCX_API_KEY,
        temperature=0.4,
    )

    prompt = (
        "You are an expert educator. Based on the document context below, generate 10 flashcards.\n"
        "Return ONLY a valid JSON array with no markdown, no code fences, no preamble.\n"
        "Format: [{\"front\": \"question or key term\", \"back\": \"answer or explanation\"}, ...]\n\n"
        f"Document Context:\n{context}\n\nJSON Array:"
    )

    response = llm.invoke(prompt)
    raw = response.content if not isinstance(response.content, list) else response.content[0]["text"]
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        cards = json.loads(raw)
        return cards
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE: QUIZ GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_quiz(vector_store):
    from langchain_google_genai import ChatGoogleGenerativeAI
    import json

    results = vector_store.similarity_search_with_score("facts details important information", k=3)
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=INDOCX_API_KEY,
        temperature=0.5,
    )

    prompt = (
        "You are an expert quiz maker. Based on the document context below, generate 7 multiple-choice questions.\n"
        "Return ONLY a valid JSON array with no markdown, no code fences, no preamble.\n"
        "Format: [{\"question\": \"...\", \"options\": [\"A. ...\", \"B. ...\", \"C. ...\", \"D. ...\"], \"answer\": \"A. ...\", \"explanation\": \"...\"}]\n"
        "The 'answer' field must be the full correct option string exactly as it appears in 'options'.\n\n"
        f"Document Context:\n{context}\n\nJSON Array:"
    )

    response = llm.invoke(prompt)
    raw = response.content if not isinstance(response.content, list) else response.content[0]["text"]
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        quiz = json.loads(raw)
        return quiz
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE: SUMMARY GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(vector_store):
    from langchain_google_genai import ChatGoogleGenerativeAI
    import json

    results = vector_store.similarity_search_with_score("main topic overview introduction conclusion", k=3)
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=INDOCX_API_KEY,
        temperature=0.3,
    )

    prompt = (
        "You are an expert academic summarizer. Based on the document context below, produce a structured summary.\n"
        "Return ONLY a valid JSON object with no markdown, no code fences, no preamble.\n"
        "Format: {\"overview\": \"2-3 sentence overview\", \"important_points\": [\"point 1\", ...], "
        "\"key_terms\": [{\"term\": \"...\", \"definition\": \"...\"}], "
        "\"exam_questions\": [\"question 1\", ...]}\n\n"
        f"Document Context:\n{context}\n\nJSON Object:"
    )

    response = llm.invoke(prompt)
    raw = response.content if not isinstance(response.content, list) else response.content[0]["text"]
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        summary = json.loads(raw)
        return summary
    except Exception:
        return {}

# ═════════════════════════════════════════════════════════════════════════════
#  CSS
#  Palette from image:
#    #044550  — darkest teal  (sidebar bg)
#    #086E77  — deep teal     (sidebar accents, buttons)
#    #199396  — mid teal      (hover states, borders)
#    #4FB3AE  — soft teal     (highlights, pills, chat input focus)
#    #9FD3CD  — lightest teal (main background, AI bubble)
#  Font: Cormorant Garamond for "InDocX" title, Figtree for body
# =════════════════════════════════════════════════════════════════════════════

def inject_css():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Figtree:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Figtree', sans-serif; }

    /* ── Main background: lightest teal #9FD3CD tinted ── */
    .stApp {
        background: #dff0ee;
        background-image:
            radial-gradient(ellipse 70% 50% at 10% 0%,  rgba(4,69,80,0.12)  0%, transparent 60%),
            radial-gradient(ellipse 55% 45% at 90% 100%, rgba(8,110,119,0.10) 0%, transparent 55%);
    }

    /* ── Streamlit chrome: hide only what's safe ── */
    #MainMenu { visibility: hidden !important; }
    footer    { visibility: hidden !important; }

    /* ── Sidebar toggle arrow — make it always visible ── */
    [data-testid="collapsedControl"] {
        background: #086E77 !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 0.5rem 0.3rem !important;
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        color: #dff0ee !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #dff0ee !important;
        stroke: #dff0ee !important;
    }
    /* Force sidebar open width and prevent it from being zero-width */
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 280px !important;
        max-width: 320px !important;
    }

    .block-container {
        padding: 3.5rem 2.5rem 2rem 2.5rem !important;
        max-width: 920px !important;
        margin: 0 auto !important;
    }

    /* ── Sidebar: darkest teal #044550 ── */
    [data-testid="stSidebar"] {
        background: #044550 !important;
        border-right: 1px solid rgba(79,179,174,0.2) !important;
    }
    [data-testid="stSidebar"] > div { padding: 1.75rem 1.4rem !important; }

    /* ── Brand ── */
    .brand {
        margin-bottom: 1.75rem;
        padding-bottom: 1.4rem;
        border-bottom: 1px solid rgba(159,211,205,0.15);
    }
    .brand-name {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #9FD3CD;
        letter-spacing: 0.04em;
        line-height: 1;
    }
    .brand-tagline {
        font-size: 0.74rem;
        color: #4FB3AE;
        margin-top: 6px;
        font-style: italic;
        letter-spacing: 0.03em;
    }
    .brand-tag {
        font-size: 0.58rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #086E77;
        margin-top: 5px;
    }

    /* ── Sidebar section labels ── */
    .s-label {
        font-size: 0.63rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4FB3AE;
        margin: 1.4rem 0 0.5rem;
    }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        background: #086E77 !important;
        border: 1px solid rgba(79,179,174,0.35) !important;
        border-radius: 8px !important;
        color: #dff0ee !important;
        font-size: 0.83rem !important;
    }
    [data-testid="stSelectbox"] label { color: #4FB3AE !important; font-size: 0.8rem !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(79,179,174,0.06) !important;
        border: 1.5px dashed rgba(79,179,174,0.35) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #4FB3AE !important;
        background: rgba(79,179,174,0.12) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] div span {
        color: #9FD3CD !important;
    }

    /* ── Status pills ── */
    .pill {
        display: inline-flex; align-items: center; gap: 5px;
        font-size: 0.7rem; font-weight: 500;
        padding: 3px 10px; border-radius: 20px; margin-top: 6px;
    }
    .pill-ok { background: rgba(25,147,150,0.2); color: #9FD3CD; border: 1px solid rgba(79,179,174,0.4); }
    .pill-go { background: rgba(8,110,119,0.2);  color: #4FB3AE; border: 1px solid rgba(79,179,174,0.25); }
    .dot { width:6px; height:6px; border-radius:50%; background:currentColor; display:inline-block; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #086E77, #199396) !important;
        color: #dff0ee !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
        transition: opacity 0.2s, transform 0.15s !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Dividers ── */
    hr { border-color: rgba(159,211,205,0.1) !important; }

    /* ── Page heading ── */
    .page-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #044550;
        letter-spacing: 0.04em;
        line-height: 1;
        margin-bottom: 0.1rem;
    }
    .page-tagline {
        font-size: 0.85rem;
        color: #199396;
        font-style: italic;
        margin-bottom: 0.25rem;
        letter-spacing: 0.02em;
    }
    .page-sub {
        font-size: 0.78rem;
        color: #4FB3AE;
        margin-bottom: 1.5rem;
    }

    /* ── Chat bubbles ── */
    .msg-wrap { display: flex; flex-direction: column; margin-bottom: 1.3rem; }
    .msg-meta {
        font-size: 0.63rem; font-weight: 600;
        letter-spacing: 0.09em; text-transform: uppercase;
        color: #199396; margin-bottom: 5px;
    }
    .msg-meta-user { text-align: right; color: #086E77; }

    .bubble {
        padding: 0.85rem 1.15rem;
        border-radius: 16px;
        font-size: 0.88rem;
        line-height: 1.7;
        max-width: 78%;
    }

    /* User bubble: deep teal */
    .bubble-user {
        background: linear-gradient(135deg, #086E77 0%, #044550 100%);
        color: #dff0ee;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }

    /* AI bubble: light teal card */
    .bubble-ai {
        background: #f0faf9;
        border: 1px solid #9FD3CD;
        color: #044550;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        box-shadow: 0 2px 10px rgba(4,69,80,0.07);
    }
    .bubble-ai strong { color: #044550; }
    .bubble-ai code {
        background: rgba(25,147,150,0.12);
        color: #086E77;
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 0.82rem;
    }

    /* ── Warning box ── */
    .warn-box {
        background: rgba(159,211,205,0.2);
        border: 1px solid #4FB3AE;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #044550;
        margin-bottom: 1rem;
    }

    /* ── Feature Cards (top buttons) ── */
    .feature-cards-row { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .feature-card {
        flex: 1; min-width: 130px;
        background: linear-gradient(135deg, #f0faf9 0%, #dff0ee 100%);
        border: 1.5px solid #9FD3CD;
        border-radius: 14px;
        padding: 1rem 0.9rem;
        text-align: center;
        cursor: pointer;
        transition: box-shadow 0.2s, border-color 0.2s, transform 0.15s;
        box-shadow: 0 2px 8px rgba(4,69,80,0.07);
    }
    .feature-card:hover {
        border-color: #4FB3AE;
        box-shadow: 0 6px 20px rgba(8,110,119,0.15);
        transform: translateY(-2px);
    }
    .feature-card-icon { font-size: 1.6rem; margin-bottom: 0.35rem; }
    .feature-card-label {
        font-size: 0.78rem; font-weight: 600;
        color: #086E77; letter-spacing: 0.02em;
    }

    /* ── Flashcard styles ── */
    .fc-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin: 1.2rem 0;
    }
    .fc-card {
        background: linear-gradient(135deg, #f0faf9, #e2f6f4);
        border: 1.5px solid #9FD3CD;
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 12px rgba(4,69,80,0.08);
        cursor: pointer;
        transition: box-shadow 0.2s, transform 0.15s;
    }
    .fc-card:hover { box-shadow: 0 6px 20px rgba(4,69,80,0.13); transform: translateY(-2px); }
    .fc-number {
        font-size: 0.6rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #4FB3AE; margin-bottom: 0.6rem;
    }
    .fc-front {
        font-weight: 700; color: #044550;
        font-size: 1rem; line-height: 1.4; flex: 1;
    }
    .fc-flip-hint {
        font-size: 0.7rem; color: #9FD3CD;
        margin-top: 0.8rem; text-align: right;
    }
    .fc-back {
        font-size: 0.88rem; color: #086E77;
        line-height: 1.6; margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(159,211,205,0.4);
    }

    /* ── Summary card ── */
    .summary-card {
        background: linear-gradient(135deg, #f0faf9, #e2f6f4);
        border: 1.5px solid #9FD3CD;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(4,69,80,0.08);
    }
    .summary-section-title {
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #086E77; margin: 1rem 0 0.4rem;
    }
    .summary-overview { font-size: 0.88rem; color: #044550; line-height: 1.7; }
    .summary-point {
        font-size: 0.84rem; color: #044550;
        padding: 0.25rem 0; border-bottom: 1px solid rgba(159,211,205,0.3);
    }
    .summary-term { font-weight: 600; color: #086E77; }
    .summary-def  { font-size: 0.82rem; color: #199396; }
    .exam-q { font-size: 0.84rem; color: #044550; padding: 0.2rem 0; font-style: italic; }

    /* ── Evidence section ── */
    .evidence-chunk {
        background: rgba(159,211,205,0.12);
        border-left: 3px solid #4FB3AE;
        border-radius: 0 8px 8px 0;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.75rem;
        font-size: 0.8rem;
        color: #044550;
        line-height: 1.6;
    }
    .evidence-meta {
        font-size: 0.65rem; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: #199396; margin-bottom: 0.3rem;
    }
    .score-bar-wrap { height: 4px; background: rgba(159,211,205,0.3); border-radius: 4px; margin-bottom: 0.5rem; }
    .score-bar { height: 4px; background: linear-gradient(90deg, #086E77, #4FB3AE); border-radius: 4px; }

    /* ── Answer mode badge ── */
    .mode-badge {
        display: inline-block;
        font-size: 0.6rem; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        padding: 2px 8px; border-radius: 20px; margin-bottom: 0.5rem;
    }
    .mode-strict { background: rgba(4,69,80,0.12); color: #044550; border: 1px solid rgba(4,69,80,0.2); }
    .mode-flex   { background: rgba(79,179,174,0.15); color: #086E77; border: 1px solid rgba(79,179,174,0.3); }

    /* ── Empty state ── */
    .empty-state { text-align: center; padding: 4rem 2rem; }
    .empty-icon  { font-size: 3rem; margin-bottom: 1rem; }
    .empty-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.3rem; font-weight: 700;
        color: #086E77; letter-spacing: 0.03em;
    }
    .empty-hint { font-size: 0.82rem; color: #199396; margin-top: 0.4rem; }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: #c2e0dc !important;
        border-radius: 16px !important;
        padding: 4px !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #e4f4f2 !important;
        border: 2px solid #199396 !important;
        border-radius: 12px !important;
        color: #044550 !important;
        font-family: 'Figtree', sans-serif !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #4FB3AE !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #086E77 !important;
        box-shadow: 0 0 0 3px rgba(8,110,119,0.2) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #4FB3AE !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# =════════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "pdf_filename": None,
        "pdf_processed": False,
        "model": "gemini-2.0-flash",
        # Feature state
        "answer_mode": "Strict Document Mode",
        "flashcards": None,
        "quiz_data": None,
        "quiz_answers": {},
        "quiz_submitted": False,
        "summary_data": None,
        "active_feature": None,          # "flashcards" | "quiz" | "summary" | None
        "evidence_store": {},            # keyed by message index
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# =════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:

        st.markdown(
            '<div class="brand">'
            '<div class="brand-name">InDocX</div>'
            '<div class="brand-tagline">Clarity from Complexity</div>'
            '<div class="brand-tag">RAG Chatbot · v1.0</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # PDF Upload
        st.markdown('<div class="s-label">📄 Document</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload PDF", type=["pdf"], label_visibility="collapsed",
        )

        if uploaded_file is not None:
            new_file = uploaded_file.name != st.session_state.pdf_filename
            if new_file:
                st.session_state.pdf_processed = False
                st.session_state.vector_store = None
                st.session_state.pdf_filename = uploaded_file.name
                st.session_state.chat_history = []

            if not st.session_state.pdf_processed:
                if st.button("⚙️ Process Document"):
                    with st.spinner("Reading & chunking PDF…"):
                        file_bytes = uploaded_file.read()
                        vector_store = process_uploaded_pdf(
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                        )
                        st.session_state.vector_store = vector_store
                        st.session_state.pdf_processed = True
                    st.rerun()

            if st.session_state.pdf_processed:
                st.markdown(
                    '<div class="pill pill-ok"><span class="dot"></span> Ready · '
                    + uploaded_file.name + '</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="pill pill-go"><span class="dot"></span> No document loaded</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.evidence_store = {}
            st.rerun()

        # ── Answer Mode Selector ──
        st.markdown('<div class="s-label">⚙️ Answer Mode</div>', unsafe_allow_html=True)
        mode_choice = st.selectbox(
            "Answer Mode",
            options=["Strict Document Mode", "Flexible Mode"],
            index=0 if st.session_state.answer_mode == "Strict Document Mode" else 1,
            label_visibility="collapsed",
        )
        if mode_choice != st.session_state.answer_mode:
            st.session_state.answer_mode = mode_choice

        if st.session_state.answer_mode == "Strict Document Mode":
            st.markdown(
                '<div style="font-size:0.7rem;color:#4FB3AE;margin-top:4px;">'
                '🔒 Answers strictly from the PDF only.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="font-size:0.7rem;color:#4FB3AE;margin-top:4px;">'
                '🌐 PDF-first, with general knowledge allowed.</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")



# ═════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT WINDOW
# =════════════════════════════════════════════════════════════════════════════

def render_feature_cards():
    """Three clickable feature cards rendered as Streamlit columns with buttons."""
    if not st.session_state.pdf_processed:
        return

    st.markdown(
        '<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#4FB3AE;margin-bottom:0.6rem;">✨ Smart Tools</div>',
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-card-icon">🃏</div>'
            '<div class="feature-card-label">Generate Flashcards</div>'
            '</div>', unsafe_allow_html=True
        )
        if st.button("Generate Flashcards", key="btn_flashcards", use_container_width=True):
            st.session_state.active_feature = "flashcards"
            st.session_state.flashcards = None
            st.rerun()

    with col2:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-card-icon">📝</div>'
            '<div class="feature-card-label">Make Quiz</div>'
            '</div>', unsafe_allow_html=True
        )
        if st.button("Make Quiz", key="btn_quiz", use_container_width=True):
            st.session_state.active_feature = "quiz"
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()

    with col3:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-card-icon">📋</div>'
            '<div class="feature-card-label">Summarize</div>'
            '</div>', unsafe_allow_html=True
        )
        if st.button("Summarize", key="btn_summary", use_container_width=True):
            st.session_state.active_feature = "summary"
            st.session_state.summary_data = None
            st.rerun()

    if st.session_state.active_feature:
        if st.button("← Back to Chat", key="btn_back"):
            st.session_state.active_feature = None
            st.rerun()


def render_flashcards_view():
    st.markdown("### 🃏 Flashcards")
    if st.session_state.flashcards is None:
        with st.spinner("Generating flashcards from your document…"):
            cards = generate_flashcards(st.session_state.vector_store)
            st.session_state.flashcards = cards

    cards = st.session_state.flashcards
    if not cards:
        st.error("Could not generate flashcards. Please try again.")
        return

    # init flipped state
    if "fc_flipped" not in st.session_state:
        st.session_state.fc_flipped = {}

    st.markdown(
        f"<div style='font-size:0.8rem;color:#199396;margin-bottom:1rem;'>"
        f"{len(cards)} flashcards · Click a card to flip it and reveal the answer.</div>",
        unsafe_allow_html=True
    )

    # Render 3 per row
    for row_start in range(0, len(cards), 3):
        row_cards = cards[row_start:row_start+3]
        cols = st.columns(len(row_cards))
        for col_idx, (col, card) in enumerate(zip(cols, row_cards)):
            card_idx = row_start + col_idx
            is_flipped = st.session_state.fc_flipped.get(card_idx, False)
            with col:
                front_html = (
                    f'<div class="fc-card">'
                    f'<div class="fc-number">Card {card_idx+1}</div>'
                    f'<div class="fc-front">{card.get("front", "")}</div>'
                    f'<div class="fc-flip-hint">tap to reveal ↓</div>'
                    f'</div>'
                )
                flipped_html = (
                    f'<div class="fc-card" style="border-color:#086E77;background:linear-gradient(135deg,#e2f6f4,#c8eeed);">'
                    f'<div class="fc-number">Card {card_idx+1} · Answer</div>'
                    f'<div class="fc-front">{card.get("front", "")}</div>'
                    f'<div class="fc-back">💡 {card.get("back", "")}</div>'
                    f'</div>'
                )
                st.markdown(flipped_html if is_flipped else front_html, unsafe_allow_html=True)
                btn_label = "Hide answer" if is_flipped else "Show answer"
                if st.button(btn_label, key=f"fc_btn_{card_idx}"):
                    st.session_state.fc_flipped[card_idx] = not is_flipped
                    st.rerun()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Regenerate Flashcards", key="btn_regen_fc"):
            st.session_state.flashcards = None
            st.session_state.fc_flipped = {}
            st.rerun()
    with col2:
        if st.button("↩ Flip All Back", key="btn_flip_all"):
            st.session_state.fc_flipped = {}
            st.rerun()


def render_quiz_view():
    st.markdown("### 📝 Quiz")
    if st.session_state.quiz_data is None:
        with st.spinner("Generating quiz questions from your document…"):
            quiz = generate_quiz(st.session_state.vector_store)
            st.session_state.quiz_data = quiz
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False

    quiz = st.session_state.quiz_data
    if not quiz:
        st.error("Could not generate quiz. Please try again.")
        return

    if not st.session_state.quiz_submitted:
        st.markdown(f"<div style='font-size:0.8rem;color:#199396;margin-bottom:1rem;'>"
                    f"{len(quiz)} questions · Select your answers then click Submit.</div>",
                    unsafe_allow_html=True)

        for i, q in enumerate(quiz):
            st.markdown(
                f'<div style="font-weight:600;color:#044550;font-size:0.9rem;margin:1rem 0 0.4rem;">'
                f'Q{i+1}. {q.get("question", "")}</div>',
                unsafe_allow_html=True
            )
            options = q.get("options", [])
            chosen = st.radio(
                f"q_{i}", options=options,
                index=None, key=f"quiz_q_{i}",
                label_visibility="collapsed"
            )
            st.session_state.quiz_answers[i] = chosen

        if st.button("✅ Submit Quiz", key="btn_submit_quiz", type="primary"):
            st.session_state.quiz_submitted = True
            st.rerun()

    else:
        # Show results
        correct_count = 0
        for i, q in enumerate(quiz):
            user_ans = st.session_state.quiz_answers.get(i)
            correct_ans = q.get("answer", "")
            is_correct = user_ans == correct_ans
            if is_correct:
                correct_count += 1

            icon = "✅" if is_correct else "❌"
            bg = "rgba(25,147,150,0.1)" if is_correct else "rgba(220,80,80,0.08)"
            border = "#4FB3AE" if is_correct else "#e57373"

            st.markdown(
                f'<div style="background:{bg};border-left:3px solid {border};'
                f'border-radius:0 10px 10px 0;padding:0.8rem 1rem;margin-bottom:0.75rem;">'
                f'<div style="font-weight:600;color:#044550;font-size:0.88rem;">'
                f'{icon} Q{i+1}. {q.get("question", "")}</div>'
                f'<div style="font-size:0.8rem;color:#199396;margin-top:0.3rem;">'
                f'Your answer: <strong>{user_ans or "Not answered"}</strong></div>'
                f'<div style="font-size:0.8rem;color:#086E77;margin-top:0.2rem;">'
                f'Correct: <strong>{correct_ans}</strong></div>'
                f'<div style="font-size:0.78rem;color:#4FB3AE;margin-top:0.3rem;font-style:italic;">'
                f'{q.get("explanation", "")}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        pct = int((correct_count / len(quiz)) * 100)
        color = "#086E77" if pct >= 70 else "#199396" if pct >= 40 else "#e57373"
        st.markdown(
            f'<div style="text-align:center;padding:1.2rem;background:rgba(159,211,205,0.2);'
            f'border:1.5px solid #9FD3CD;border-radius:14px;margin-top:1rem;">'
            f'<div style="font-size:2rem;font-weight:700;color:{color};">{correct_count}/{len(quiz)}</div>'
            f'<div style="font-size:0.9rem;color:#199396;">Score: {pct}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        if st.button("🔄 Retake Quiz", key="btn_retake"):
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()


def render_summary_view():
    st.markdown("### 📋 Document Summary")
    if st.session_state.summary_data is None:
        with st.spinner("Generating structured summary…"):
            summary = generate_summary(st.session_state.vector_store)
            st.session_state.summary_data = summary

    s = st.session_state.summary_data
    if not s:
        st.error("Could not generate summary. Please try again.")
        return

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)

    # Overview
    st.markdown('<div class="summary-section-title">📌 Overview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-overview">{s.get("overview", "")}</div>', unsafe_allow_html=True)

    # Important points
    points = s.get("important_points", [])
    if points:
        st.markdown('<div class="summary-section-title">⭐ Important Points</div>', unsafe_allow_html=True)
        for p in points:
            st.markdown(f'<div class="summary-point">• {p}</div>', unsafe_allow_html=True)

    # Key terms
    terms = s.get("key_terms", [])
    if terms:
        st.markdown('<div class="summary-section-title">🔑 Key Terms</div>', unsafe_allow_html=True)
        for t in terms:
            st.markdown(
                f'<div style="padding:0.3rem 0;">'
                f'<span class="summary-term">{t.get("term", "")}</span>'
                f' — <span class="summary-def">{t.get("definition", "")}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Exam questions
    exam_qs = s.get("exam_questions", [])
    if exam_qs:
        st.markdown('<div class="summary-section-title">🎓 Possible Exam / Viva Questions</div>', unsafe_allow_html=True)
        for eq in exam_qs:
            st.markdown(f'<div class="exam-q">❓ {eq}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Regenerate Summary", key="btn_regen_summary"):
        st.session_state.summary_data = None
        st.rerun()


def render_evidence(chunks, scores, msg_index):
    """Render expandable evidence section below a chat answer."""
    if not chunks:
        return
    with st.expander(f"🔍 Evidence Used ({len(chunks)} chunks)", expanded=False):
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            # Normalize score: FAISS L2 distance → lower = better. Convert to 0-1 similarity.
            # Simple inverse normalization capped at 1.0
            similarity = max(0.0, min(1.0, 1.0 / (1.0 + score)))
            pct = int(similarity * 100)
            bar_width = pct

            st.markdown(
                f'<div class="evidence-chunk">'
                f'<div class="evidence-meta">Chunk {i+1} &nbsp;·&nbsp; Relevance: {pct}%</div>'
                f'<div class="score-bar-wrap"><div class="score-bar" style="width:{bar_width}%;"></div></div>'
                f'{chunk[:600]}{"…" if len(chunk) > 600 else ""}'
                f'</div>',
                unsafe_allow_html=True
            )


def render_chat():
    st.markdown(
        '<div class="page-title">InDocX</div>'
        '<div class="page-tagline">Clarity from Complexity</div>'
        '<div class="page-sub">Upload a PDF and ask anything about it</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.pdf_processed:
        st.markdown(
            '<div class="warn-box">📄 <strong>Upload and process a PDF</strong> '
            'from the sidebar to start chatting.</div>',
            unsafe_allow_html=True
        )

    # ── Feature Cards ──
    render_feature_cards()

    # ── Feature Views (replace chat when active) ──
    if st.session_state.active_feature == "flashcards":
        render_flashcards_view()
        return
    elif st.session_state.active_feature == "quiz":
        render_quiz_view()
        return
    elif st.session_state.active_feature == "summary":
        render_summary_view()
        return

    # ── Chat history ──
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-title">Ask anything about your document</div>'
            '<div class="empty-hint">Upload a PDF from the sidebar, then type your question below.</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        for idx, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(
                    '<div class="msg-wrap">'
                    '<div class="msg-meta msg-meta-user">You</div>'
                    '<div class="bubble bubble-user">' + msg["content"] + '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                mode = st.session_state.answer_mode
                badge_cls = "mode-strict" if mode == "Strict Document Mode" else "mode-flex"
                badge_label = "📄 Strict Mode" if mode == "Strict Document Mode" else "🌐 Flexible Mode"
                st.markdown(
                    '<div class="msg-wrap">'
                    '<div class="msg-meta">InDocX</div>'
                    f'<span class="mode-badge {badge_cls}">{badge_label}</span>'
                    '<div class="bubble bubble-ai">' + msg["content"] + '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                # Evidence for this assistant turn
                evidence = st.session_state.evidence_store.get(idx)
                if evidence:
                    render_evidence(evidence["chunks"], evidence["scores"], idx)

    # ── Chat input ──
    user_question = st.chat_input(
        placeholder="Ask a question about your document…",
        disabled=not st.session_state.pdf_processed,
    )

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.spinner("Thinking…"):
            result = get_rag_response(
                question=user_question,
                vector_store=st.session_state.vector_store,
                model=st.session_state.model,
                answer_mode=st.session_state.answer_mode,
            )
            answer, chunks, scores = result

        ai_msg_idx = len(st.session_state.chat_history)  # index of the upcoming assistant message
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.evidence_store[ai_msg_idx] = {"chunks": chunks, "scores": scores}
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# =════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    init_session_state()
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()