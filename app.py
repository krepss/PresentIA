import streamlit as st
import os
import tempfile
import pdfplumber
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Configuração da Página ---
st.set_page_config(page_title="Tutor IA - Pro", page_icon="🎓", layout="wide")

# --- CSS Personalizado (Visual Profissional) ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #4F46E5; color: white; border-radius: 8px; font-weight: bold; padding: 0.5rem; }
    .premium-box { background-color: #f0fdf4; border: 2px solid #22c55e; padding: 20px; border-radius: 10px; text-align: center; }
    .locked-box { background-color: #fef2f2; border: 2px solid #ef4444; padding: 20px; border-radius: 10px; text-align: center; }
    .instruction-card { background-color: #f3f4f6; padding: 15px; border-radius: 8px; border-left: 5px solid #4F46E5; }
    h1, h2, h3 { color: #1e1b4b; }
    /* Estilo do Botão de Compra da Sidebar */
    .sidebar-buy-btn {
        display: block;
        width: 100%;
        background-color: #16a34a; /* Verde Venda */
        color: white;
        text-align: center;
        padding: 12px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        margin-top: 10px;
        border: 1px solid #15803d;
    }
    .sidebar-buy-btn:hover {
        background-color: #15803d;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO ---
api_key = "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"
SENHAS_VALIDAS = ["ALUNO100", "ESTUDAR2024", "PASSARAGORA"] 
LINK_PAGAMENTO = "https://buy.stripe.com/fZu7sK8nF93W1O2czp4c800" 

# --- Barra Lateral (Login e Venda) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
    st.header("🔐 Área do Aluno")
    
    # Input de Senha
    senha_user = st.text_input("Tenho uma Chave de Acesso", type="password", placeholder="Digite sua senha aqui...")
    
    # Lógica de Validação
    if senha_user in SENHAS_VALIDAS:
        st.success("✅ Acesso Premium Ativo!")
        st.session_state.is_premium = True
    elif senha_user:
        st.error("Chave inválida.")
        st.session_state.is_premium = False
    else:
        st.caption("🔒 Funções avançadas bloqueadas.")
        st.session_state.is_premium = False

    # --- BOTÃO DE VENDA LATERAL (A novidade) ---
    # Só mostramos se o usuário NÃO for premium
    if not st.session_state.get("is_premium", False):
        st.markdown("---")
        st.markdown("**Ainda não é aluno?**")
        st.markdown(f"""
        <a href="{LINK_PAGAMENTO}" target="_blank" class="sidebar-buy-btn">
            🔓 COMPRAR PREMIUM
        </a>
        <p style="text-align:center; font-size:12px; color:gray; margin-top:5px;">Acesso imediato por R$ 29,90</p>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Limpar Tudo / Reiniciar"):
        st.session_state.messages = []
        st.session_state.quiz_history = []
        st.session_state.vector_db = None
        st.rerun()

# --- Funções de IA (Backend) ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_files(uploaded_files):
    all_documents = []
    status = st.empty()
    prog = st.progress(0)
    for idx, file in enumerate(uploaded_files):
        status.text(f"🧠 Lendo arquivo: {file.name}...")
        prog.progress((idx+1)/len(uploaded_files))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        doc = Document(page_content=text, metadata={"source": file.name, "page": i+1})
                        all_documents.append(doc)
        except Exception as e: st.error(f"Erro: {e}")
        finally: 
            if os.path.exists(tmp_path): os.remove(tmp_path)
    if not all_documents: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    db = FAISS.from_documents(texts, get_embeddings())
    status.empty(); prog.empty()
    return db

def generate_quiz(topic, qtd, db):
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    prompt_template = f"""
    Crie uma prova com {qtd} questões de múltipla escolha sobre: '{topic}'.
    Baseado APENAS no contexto. Formato OBRIGATÓRIO:
    ### Questão X
    **Enunciado**
    a) ... b) ... c) ... d) ...
    > **Gabarito:** Letra - Explicação
    ---
    """
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 5}))
    return qa.invoke({"query": prompt_template})['result']

# --- Inicialização de Estado ---
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "messages" not in st.session_state: st.session_state.messages = []
if "quiz_history" not in st.session_state: st.session_state.quiz_history = []

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

# Título Principal
st.title("🎓 Tutor IA: Sua Máquina de Aprovação")

# Se NÃO tiver arquivo carregado, mostra a Landing Page
if not st.session_state.vector_db:
    
    st.markdown("### Transforme PDFs chatos em Simulados em segundos.")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="instruction-card">
            <h3>📂 1. Carregue</h3>
            <p>Arraste suas apostilas, editais ou livros em PDF aqui em cima. Pode ser mais de um!</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="instruction-card">
            <h3>🤖 2. A IA Estuda</h3>
            <p>Nossa inteligência artificial lê 100 páginas em segundos e entende tudo.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="instruction-card">
            <h3>📝 3. Pratique</h3>
            <p>Gere provas infinitas com gabarito ou tire dúvidas no chat interativo.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "👇 COMECE AGORA: Solte seus arquivos PDF aqui", 
        type="pdf", 
        accept_multiple_files=True,
        help="Arraste seus arquivos aqui. Limite de 200MB por arquivo."
    )

    if uploaded_files:
        if st.button("🚀 Processar e Começar a Estudar", use_container_width=True):
            with st.spinner("O Professor IA está lendo seu material..."):
                st.session_state.vector_db = process_files(uploaded_files)
                st.success("Tudo pronto!")
                st.rerun()

# Se JÁ tiver arquivo carregado
else:
    st.info(f"✅ Modo de Estudo Ativo. A IA já leu seus documentos.")
    
    tab1, tab2 = st.tabs(["💬 Chat Tira-Dúvidas", "📝 Gerador de Provas (Premium)"])
    
    # ABA 1: CHAT
    with tab1:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])
        if prompt := st.chat_input("Pergunte algo sobre a matéria..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Pesquisando na apostila..."):
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}))
                    resp = qa.invoke({"query": prompt})['result']
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

    # ABA 2: SIMULADO
    with tab2:
        if st.session_state.get("is_premium", False):
            st.markdown('<div class="premium-box">💎 Acesso Premium Liberado</div><br>', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1: topic = st.text_input("Sobre o que você quer ser testado?", placeholder="Ex: Prazos Processuais, Anatomia...")
            with col2: qtd = st.slider("Qtd. Questões", 1, 10, 3)
            
            if st.button("🎯 Gerar Simulado Agora", use_container_width=True):
                if topic:
                    with st.spinner(f"Criando {qtd} questões inéditas..."):
                        try:
                            raw_quiz = generate_quiz(topic, qtd, st.session_state.vector_db)
                            st.session_state.quiz_history.insert(0, {"topic": topic, "content": raw_quiz, "qtd": qtd})
                        except Exception as e: st.error(f"Erro: {e}")
            
            if st.session_state.quiz_history:
                for i, quiz in enumerate(st.session_state.quiz_history):
                    with st.expander(f"📝 Prova: {quiz['topic']} ({quiz['qtd']} questões)", expanded=(i==0)):
                        questoes = quiz['content'].split("---")
                        for q in questoes:
                            if q.strip():
                                partes = q.split("> **Gabarito:**")
                                if len(partes) == 2:
                                    st.markdown(partes[0])
                                    with st.expander("👁️ Ver Gabarito"): st.info(partes[1])
                                else: st.markdown(q)
                                st.markdown("---")
        else:
            st.markdown(f"""
            <div class="locked-box">
                <h3>🔒 Funcionalidade Premium</h3>
                <p>Estude 10x mais rápido gerando provas infinitas.</p>
                <a href="{LINK_PAGAMENTO}" target="_blank">
                    <button style="background-color:#ef4444; color:white; padding:15px 32px; font-size:16px; border:none; border-radius:8px; cursor:pointer;">
                        🔓 DESBLOQUEAR AGORA
                    </button>
                </a>
                <br><br><small>Já tem senha? Digite na barra lateral.</small>
            </div>
            """, unsafe_allow_html=True)
