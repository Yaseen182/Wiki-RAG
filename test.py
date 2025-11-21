import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import uuid
import torch
import wikipedia
import time

# --- RAG IMPORTS ---
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------- CONFIG ----------
st.set_page_config(page_title="RAG Chat", layout="wide", page_icon="✨")

# ---------- LOAD AI MODELS (CACHED) ----------
@st.cache_resource
def load_models():
    # تحميل الموديلات مرة واحدة فقط
    retriever = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        device=device
    )
    return retriever, generator

# تحميل صامت للموديلات
retriever_model, generator_model = load_models()

# ---------- RAG FUNCTIONS ----------
def split_into_chunks(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def fetch_wikipedia(query, max_articles=4):
    try:
        search_results = wikipedia.search(query, results=max_articles)
        chunks = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                paragraphs = [p for p in page.content.split("\n") if len(p) > 50]
                for p in paragraphs:
                    chunks.extend(split_into_chunks(p))
            except:
                continue
        return chunks
    except:
        return []

def ask_rag_pipeline(query):
    knowledge_base = fetch_wikipedia(query)
    if not knowledge_base:
        return "I couldn't find any relevant info on Wikipedia."
    
    knowledge_embeddings = retriever_model.encode(knowledge_base, convert_to_tensor=True)
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    
    cos_scores = util.pytorch_cos_sim(query_embedding, knowledge_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(5, len(knowledge_base)))
    
    top_chunks = [knowledge_base[i] for i in top_results.indices.tolist()]
    context = " ".join(top_chunks)
    
    result = generator_model(question=query, context=context[:2000])
    return result['answer']

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "text": "Hello! Ask me anything."}
    ]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------- STYLES ----------
logo_src = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    :root {
        --bg-dark: #1a1d21; --bg-darker: #15181c; --card-bg: #252931;
        --user-bubble: #2d4a6d; --ai-bubble: #3d2d52;
        --text-primary: #e8eaed; 
        --border: rgba(255,255,255,0.08);
    }
    
    * { font-family: 'Inter', sans-serif; }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, var(--bg-darker) 0%, var(--bg-dark) 100%);
        color: var(--text-primary);
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stTextInput > label {display: none;}
    
    .chat-container {
        max-width: 480px; margin: 20px auto; height: calc(100vh - 100px);
        border-radius: 24px; overflow: hidden; background: var(--bg-dark); border: 1px solid var(--border);
        display: flex; flex-direction: column;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    }
    
    .header {
        padding: 20px; text-align: center;
        background: rgba(37,41,49,0.95); border-bottom: 1px solid var(--border);
    }
    
    .messages {
        flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; background: var(--bg-darker);
    }

    .message-wrapper { display: flex; flex-direction: column; animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    
    .bubble { max-width: 85%; padding: 12px 16px; border-radius: 18px; line-height: 1.5; font-size: 15px; color: var(--text-primary); }
    .ai-wrapper { align-items: flex-start; }
    .ai { background: #4a3464; border-bottom-left-radius: 4px; }
    .user-wrapper { align-items: flex-end; }
    .user { background: #3d5a7e; border-bottom-right-radius: 4px; }

    /* Typing Animation Dots */
    .typing-dots {
        display: flex; gap: 4px; padding: 4px 8px; align-items: center;
    }
    .dot {
        width: 6px; height: 6px; background: rgba(255,255,255,0.6); border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

    /* Input Area */
    .input-area { padding: 16px; background: var(--card-bg); border-top: 1px solid var(--border); }
    .stTextInput input { color: white !important; }
    
    @media (max-width: 768px) { .chat-container { margin: 0; border-radius: 0; height: 100vh; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- MAIN CONTAINER ----------
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Header
    st.markdown(f'''
    <div class="header">
        <div style="width:48px; height:48px; margin:0 auto 8px; border-radius:12px; overflow:hidden;">
            <img src="{logo_src}" style="width:100%; height:100%; object-fit:cover;" />
        </div>
        <div style="font-weight:600; font-size:16px;">AI ChatBot</div>
    </div>
    ''', unsafe_allow_html=True)

    # Messages area
    messages_container = st.container()
    with messages_container:
        st.markdown('<div class="messages" id="messages">', unsafe_allow_html=True)
        
        # 1. عرض جميع الرسائل السابقة
        for msg in st.session_state.messages:
            role = msg["role"]
            text = msg["text"]
            if role == "ai":
                st.markdown(f'<div class="message-wrapper ai-wrapper"><div class="bubble ai">{text}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message-wrapper user-wrapper"><div class="bubble user">{text}</div></div>', unsafe_allow_html=True)
        
        # 2. نقطة الحسم: إذا كان هناك سؤال قيد الانتظار، نعرض النقاط فوراً هنا
        # هذا يضمن ظهور النقاط مع باقي الرسائل في نفس لحظة تحميل الصفحة
        ai_placeholder = st.empty()
        if st.session_state.pending_prompt:
            loading_html = '''
            <div class="message-wrapper ai-wrapper">
                <div class="bubble ai">
                    <div class="typing-dots">
                        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                    </div>
                </div>
            </div>
            '''
            ai_placeholder.markdown(loading_html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------- LOGIC EXECUTION ----------
# نقوم بالمعالجة هنا ولكن التحديث يتم في الـ Placeholder الذي حجزناه في الأعلى
if st.session_state.pending_prompt:
    # تأخير بسيط جداً لضمان رسم الواجهة قبل بدء المعالجة الثقيلة
    time.sleep(0.1)
    
    try:
        # تشغيل RAG
        response_text = ask_rag_pipeline(st.session_state.pending_prompt)
        
        # محاكاة الكتابة (Streaming)
        streamed_text = ""
        words = response_text.split()
        for word in words:
            streamed_text += word + " "
            time.sleep(0.05)
            
            # تحديث الـ Placeholder الموجود داخل الرسائل
            ai_placeholder.markdown(f'''
            <div class="message-wrapper ai-wrapper">
                <div class="bubble ai">{streamed_text}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # حفظ الرسالة وإلغاء حالة الانتظار
        st.session_state.messages.append({"role": "ai", "text": response_text})
        st.session_state.pending_prompt = None
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.pending_prompt = None

# ---------- INPUT LOGIC ----------
def submit():
    if st.session_state.user_input.strip():
        user_text = st.session_state.user_input.strip()
        st.session_state.messages.append({"role": "user", "text": user_text})
        st.session_state.pending_prompt = user_text
        st.session_state.user_input = ""

# ---------- INPUT UI ----------
col1, col2 = st.columns([0.88, 0.12])

with col1:
    st.text_input(
        "", 
        key="user_input", 
        placeholder="Ask something...", 
        label_visibility="collapsed",
        on_change=submit,
        disabled=st.session_state.pending_prompt is not None 
    )

with col2:
    st.button(
        "➤", 
        key="send_btn", 
        use_container_width=True, 
        on_click=submit,
        disabled=st.session_state.pending_prompt is not None
    )

# Auto Scroll script
components.html("""
<script>
    var observer = new MutationObserver(function() {
        var el = window.parent.document.getElementById("messages");
        if (el) el.scrollTop = el.scrollHeight;
    });
    var el = window.parent.document.getElementById("messages");
    if (el) observer.observe(el, {childList: true, subtree: true});
</script>
""", height=0)