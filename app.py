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

# --- CSS Personalizado ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #4F46E5; color: white; border-radius: 8px; font-weight: bold; padding: 0.5rem; }
    .premium-box { background-color: #f0fdf4; border: 2px solid #22c55e; padding: 20px; border-radius: 10px; text-align: center; }
    .locked-box { background-color: #fef2f2; border: 2px solid #ef4444; padding: 30px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .instruction-card { background-color: #f3f4f6; padding: 15px; border-radius: 8px; border-left: 5px solid #4F46E5; }
    h1, h2, h3 { color: #1e1b4b; }
    .sidebar-buy-btn {
        display: block; width: 100%; background-color: #16a34a; color: white; text-align: center;
        padding: 12px; border-radius: 8px; text-decoration: none; font-weight: bold; margin-top: 10px; border: 1px solid #15803d;
    }
    .sidebar-buy-btn:hover { background-color: #15803d; color: white; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    st.error("A chave da API não foi encontrada nos Secrets!")
    st.stop()
SENHAS_VALIDAS = ["ALUNO100", "ESTUDAR2024", "PASSARAGORA"] 
LINK_PAGAMENTO = "https://buy.stripe.com/fZu7sK8nF93W1O2czp4c800" 

# --- Barra Lateral ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
    st.header("🔐 Área do Aluno")
    
    senha_user = st.text_input("Tenho uma Chave de Acesso", type="password", placeholder="Digite sua senha...")
    
    if senha_user in SENHAS_VALIDAS:
        st.success("✅ Premium Ativo")
        st.session_state.is_premium = True
    elif senha_user:
        st.error("Senha inválida")
        st.session_state.is_premium = False
    else:
        st.session_state.is_premium = False

    if not st.session_state.get("is_premium", False):
        st.markdown("---")
        st.markdown("**Ainda não é aluno?**")
        st.markdown(f'<a href="{LINK_PAGAMENTO}" target="_blank" class="sidebar-buy-btn">🔓 COMPRAR PREMIUM</a>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Limpar Memória"):
        st.session_state.messages = []
        st.session_state.quiz_history = []
        st.session_state.vector_db = None
        st.rerun()

# --- Funções Backend ---
@st.cache_resource
def get_embeddings(): return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_files(uploaded_files):
    all_documents = []
    status = st.empty(); prog = st.progress(0)
    for idx, file in enumerate(uploaded_files):
        status.text(f"Lendo: {file.name}..."); prog.progress((idx+1)/len(uploaded_files))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue()); tmp_path = tmp.name
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text: all_documents.append(Document(page_content=text, metadata={"source": file.name, "page": i+1}))
        except: pass
        finally: os.remove(tmp_path) if os.path.exists(tmp_path) else None
    if not all_documents: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    db = FAISS.from_documents(texts, get_embeddings())
    status.empty(); prog.empty()
    return db

def generate_quiz(topic, qtd, db):
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    prompt = f"Crie uma prova com {qtd} questões sobre '{topic}'. Formato: ### Questão X\\n**Enunciado**\\na)...\\n> **Gabarito:** Letra - Explicação\\n---"
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 5}))
    return qa.invoke({"query": prompt})['result']

# --- Inicialização ---
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "messages" not in st.session_state: st.session_state.messages = []
if "quiz_history" not in st.session_state: st.session_state.quiz_history = []

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================
st.title("🎓 Tutor IA: Sua Máquina de Aprovação")

# 1. Upload Fixo no Topo
with st.expander("📂 Carregar Apostilas e PDFs (Clique Aqui)", expanded=not st.session_state.vector_db):
    uploaded_files = st.file_uploader("Solte seus arquivos PDF aqui", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("🚀 Processar Arquivos", use_container_width=True):
            with st.spinner("Lendo material..."):
                st.session_state.vector_db = process_files(uploaded_files)
                st.success("Pronto! Pode usar as abas abaixo.")
                st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# 2. As Abas (Sempre Visíveis)
tab1, tab2 = st.tabs(["💬 Chat (Grátis)", "🔒 Gerador de Provas (Premium)"])

# --- ABA 1: CHAT ---
with tab1:
    if not st.session_state.vector_db:
        # Landing Page dentro da aba Chat se não tiver arquivo
        st.info("👆 Carregue um PDF acima para habilitar o Chat.")
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown("### 📂 1. Carregue"); st.caption("Suba suas apostilas.")
        with col2: st.markdown("### 🤖 2. IA Estuda"); st.caption("Ela lê tudo em segundos.")
        with col3: st.markdown("### 📝 3. Pratique"); st.caption("Tire dúvidas e faça provas.")
    else:
        # Chat Ativo
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])
        if prompt := st.chat_input("Pergunte sobre a matéria..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Pesquisando..."):
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}))
                    resp = qa.invoke({"query": prompt})['result']
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

# --- ABA 2: SIMULADO (O PRODUTO DE VENDA) ---
with tab2:
    # A Mágica: Verificação de Premium acontece PRIMEIRO
    if not st.session_state.get("is_premium", False):
        # TELA DE BLOQUEIO (Venda) - Aparece mesmo sem arquivo
        st.markdown(f"""
        <div class="locked-box">
            <h2>🔒 Funcionalidade Exclusiva</h2>
            <p style="font-size: 18px;">Gere <b>Simulados Infinitos</b> e questões de prova das suas apostilas.</p>
            <p>Estude 10x mais rápido com a Inteligência Artificial.</p>
            <br>
            <a href="{LINK_PAGAMENTO}" target="_blank">
                <button style="background-color:#16a34a; color:white; padding:15px 40px; font-size:20px; border:none; border-radius:8px; cursor:pointer; font-weight:bold;">
                    🔓 DESBLOQUEAR POR R$ 19,90
                </button>
            </a>
            <br><br><small>Pagamento Único. Acesso Imediato.</small>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Se for Premium, mas não tiver arquivo
        if not st.session_state.vector_db:
            st.warning("⚠️ Você tem acesso Premium, mas precisa carregar um PDF lá em cima primeiro!")
        
        # Se for Premium E tiver arquivo (Funcionalidade Real)
        else:
            st.markdown('<div class="premium-box">💎 Gerador de Provas Ativo</div><br>', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1: topic = st.text_input("Tema da Prova", placeholder="Ex: Crase, Direito Penal...")
            with col2: qtd = st.slider("Questões", 1, 10, 3)
            
            if st.button("🎯 Gerar Simulado", use_container_width=True):
                if topic:
                    with st.spinner(f"Criando prova sobre {topic}..."):
                        try:
                            raw_quiz = generate_quiz(topic, qtd, st.session_state.vector_db)
                            st.session_state.quiz_history.insert(0, {"topic": topic, "content": raw_quiz, "qtd": qtd})
                        except Exception as e: st.error(f"Erro: {e}")
            
            # Histórico de Provas
            if st.session_state.quiz_history:
                st.write("---")
                for i, quiz in enumerate(st.session_state.quiz_history):
                    with st.expander(f"📝 Prova: {quiz['topic']}", expanded=(i==0)):
                        questoes = quiz['content'].split("---")
                        for q in questoes:
                            if q.strip():
                                partes = q.split("> **Gabarito:**")
                                if len(partes) == 2:
                                    st.markdown(partes[0])
                                    with st.expander("👁️ Ver Gabarito"): st.info(partes[1])
                                else: st.markdown(q)
                                st.markdown("---")
