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
st.set_page_config(page_title="Tutor AI - Pro", page_icon="🎓", layout="wide")

# --- CSS Personalizado ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #4F46E5; color: white; border-radius: 8px; font-weight: bold; }
    .premium-box { background-color: #f0fdf4; border: 2px solid #22c55e; padding: 20px; border-radius: 10px; text-align: center; }
    .locked-box { background-color: #fef2f2; border: 2px solid #ef4444; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO ---
# 1. Coloque sua API Key da Groq aqui (ou use st.secrets para segurança em produção)
api_key = "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"

# 2. Defina sua "Senha Mestra" para vender (Simplificado para MVP)
# Em produção real, você usaria um banco de dados, mas para começar isso funciona.
SENHAS_VALIDAS = ["ALUNO100", "ESTUDAR2024", "PASSARAGORA"] 

# 3. Coloque aqui seu Link do Stripe
LINK_PAGAMENTO = "https://buy.stripe.com/test_5kQ8wO0TA9JT5bQ5dC8so00" 

# --- Barra Lateral (Login) ---
with st.sidebar:
    st.header("🔐 Área do Aluno")
    
    # Sistema simples de login
    senha_user = st.text_input("Tenho uma Chave de Acesso", type="password")
    
    if senha_user in SENHAS_VALIDAS:
        st.success("✅ Acesso Premium Ativo!")
        st.session_state.is_premium = True
    elif senha_user:
        st.error("Chave inválida.")
        st.session_state.is_premium = False
    else:
        st.info("Algumas funções são exclusivas para assinantes.")
        st.session_state.is_premium = False

    st.markdown("---")
    st.header("🎓 Material")
    if st.button("🗑️ Limpar Tudo"):
        st.session_state.messages = []
        st.session_state.quiz_history = []
        st.session_state.vector_db = None
        st.rerun()

# --- Funções de IA (Mesmas de antes) ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_files(uploaded_files):
    all_documents = []
    status = st.empty()
    prog = st.progress(0)
    for idx, file in enumerate(uploaded_files):
        status.text(f"Lendo: {file.name}...")
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

# --- Interface Principal ---
st.title("🎓 Tutor IA: Plataforma de Estudos")

if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "messages" not in st.session_state: st.session_state.messages = []
if "quiz_history" not in st.session_state: st.session_state.quiz_history = []

uploaded_files = st.file_uploader("Carregar Apostilas (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("📚 Processar Material"):
        with st.spinner("Lendo..."):
            st.session_state.vector_db = process_files(uploaded_files)
            st.success("Pronto!")

if st.session_state.vector_db:
    tab1, tab2 = st.tabs(["💬 Chat (Grátis)", "📝 Gerador de Provas (Premium)"])
    
    # ABA 1: CHAT LIBERADO (A ISCA)
    with tab1:
        st.info("💡 O Chat é liberado para você testar a inteligência da IA.")
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])
        if prompt := st.chat_input("Tire sua dúvida..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("..."):
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}))
                    resp = qa.invoke({"query": prompt})['result']
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

    # ABA 2: SIMULADO BLOQUEADO (O PRODUTO)
    with tab2:
        if st.session_state.get("is_premium", False):
            # --- CONTEÚDO PREMIUM ---
            st.markdown('<div class="premium-box">💎 Acesso Premium Liberado</div><br>', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1: topic = st.text_input("Assunto da Prova")
            with col2: qtd = st.slider("Qtd.", 1, 10, 3)
            
            if st.button("🎯 Criar Prova"):
                if topic:
                    with st.spinner(f"Criando {qtd} questões..."):
                        try:
                            raw_quiz = generate_quiz(topic, qtd, st.session_state.vector_db)
                            st.session_state.quiz_history.insert(0, {"topic": topic, "content": raw_quiz, "qtd": qtd})
                        except Exception as e: st.error(f"Erro: {e}")
            
            if st.session_state.quiz_history:
                for i, quiz in enumerate(st.session_state.quiz_history):
                    with st.expander(f"📝 {quiz['topic']}", expanded=(i==0)):
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
            # --- TELA DE BLOQUEIO (PAYWALL) ---
            st.markdown(f"""
            <div class="locked-box">
                <h3>🔒 Funcionalidade Bloqueada</h3>
                <p>O Gerador de Simulados Automático é exclusivo para alunos Premium.</p>
                <p>Estude 10x mais rápido gerando provas infinitas das suas apostilas.</p>
                <a href="{LINK_PAGAMENTO}" target="_blank">
                    <button style="background-color:#ef4444; color:white; padding:15px 32px; font-size:16px; border:none; border-radius:8px; cursor:pointer;">
                        🔓 DESBLOQUEAR AGORA POR R$ 19,90
                    </button>
                </a>
                <br><br>
                <small>Já comprou? Digite sua chave de acesso na barra lateral esquerda.</small>
            </div>
            """, unsafe_allow_html=True)

elif not uploaded_files:
    st.info("👆 Carregue suas apostilas para começar.")
