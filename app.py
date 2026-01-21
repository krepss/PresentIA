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
st.set_page_config(page_title="Tutor AI - Estudos", page_icon="🎓", layout="wide")

# --- CSS Personalizado (Estilo Clean para Estudantes) ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #4F46E5; color: white; border-radius: 8px; }
    .stChatMessage { border-radius: 12px; }
    h1 { color: #4F46E5; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA CHAVE ---
api_key = "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"

# --- Barra Lateral ---
with st.sidebar:
    st.header("🎓 Material de Estudo")
    st.info("Suba apostilas, editais ou resumos. A IA vai criar questões de prova para você.")
    
    if st.button("Limpar Tudo"):
        st.session_state.messages = []
        st.session_state.quiz_history = []
        st.session_state.vector_db = None
        st.rerun()

# --- Funções de IA ---

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
        except Exception as e:
            st.error(f"Erro em {file.name}: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
    if not all_documents: return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    db = FAISS.from_documents(texts, get_embeddings())
    
    status.empty()
    prog.empty()
    return db

def generate_quiz(topic, db):
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    
    # Prompt Especializado para Criar Questões
    prompt_template = f"""
    Você é um professor universitário rigoroso. Baseado APENAS nos documentos fornecidos, crie 3 questões de múltipla escolha sobre o tópico: '{topic}'.
    
    Formato OBRIGATÓRIO para cada questão:
    **Questão X:** [Enunciado]
    a) [Opção]
    b) [Opção]
    c) [Opção]
    d) [Opção]
    
    *Resposta Correta:* [Letra] - [Explicação breve]
    ---
    """
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
    )
    
    return qa.invoke({"query": prompt_template})['result']

# --- Interface Principal ---

st.title("🎓 Tutor IA: Seu Parceiro de Aprovação")

if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "messages" not in st.session_state: st.session_state.messages = []
if "quiz_history" not in st.session_state: st.session_state.quiz_history = []

# Upload
uploaded_files = st.file_uploader("Carregar Apostilas/Editais (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("📚 Processar Material"):
        with st.spinner("Estudando o conteúdo..."):
            st.session_state.vector_db = process_files(uploaded_files)
            st.success("Material aprendido! Escolha uma aba abaixo.")

# Abas de Funcionalidade
if st.session_state.vector_db:
    tab1, tab2 = st.tabs(["💬 Chat com a Matéria", "📝 Gerador de Simulado"])
    
    # --- ABA 1: CHAT ---
    with tab1:
        st.subheader("Tire dúvidas específicas")
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])
            
        if prompt := st.chat_input("Ex: O que o texto diz sobre prazos recursais?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Consultando..."):
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}))
                    resp = qa.invoke({"query": prompt})['result']
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

    # --- ABA 2: SIMULADO (A Inovação) ---
    with tab2:
        st.subheader("Teste seu conhecimento")
        topic = st.text_input("Sobre qual tópico você quer gerar questões?", placeholder="Ex: Direito Constitucional, Anatomia do Coração...")
        
        if st.button("🎯 Gerar Questões de Prova"):
            if topic:
                with st.spinner("O Professor IA está elaborando a prova..."):
                    try:
                        quiz_content = generate_quiz(topic, st.session_state.vector_db)
                        st.session_state.quiz_history.append({"topic": topic, "content": quiz_content})
                    except Exception as e:
                        st.error(f"Erro: {e}")
            else:
                st.warning("Digite um tópico primeiro.")
        
        # Mostrar Simulados Gerados
        if st.session_state.quiz_history:
            st.write("---")
            for i, quiz in enumerate(reversed(st.session_state.quiz_history)):
                with st.expander(f"📝 Simulado: {quiz['topic']} (Clique para ver)", expanded=(i==0)):
                    st.markdown(quiz['content'])

elif not uploaded_files:
    st.info("👆 Comece enviando suas apostilas para ativar o Tutor.")
