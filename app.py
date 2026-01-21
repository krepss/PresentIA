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

# --- CSS Personalizado ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #4F46E5; color: white; border-radius: 8px; font-weight: bold; }
    .stChatMessage { border-radius: 12px; }
    h1 { color: #4F46E5; }
    .stMarkdown h3 { color: #4338ca; } /* Cor para os títulos das questões */
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA CHAVE ---
api_key = "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"

# --- Barra Lateral ---
with st.sidebar:
    st.header("🎓 Material de Estudo")
    st.success("✅ Tutor Ativo")
    st.info("Suba apostilas, editais ou resumos. A IA vai criar provas personalizadas.")
    
    if st.button("🗑️ Limpar Tudo"):
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

def generate_quiz(topic, qtd, db):
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    
    # Prompt aprimorado para formatação limpa
    prompt_template = f"""
    Você é um examinador de banca de concurso exigente. Baseado APENAS nos documentos, crie uma prova com {qtd} questões de múltipla escolha sobre: '{topic}'.
    
    IMPORTANTE: Siga ESTRITAMENTE este formato de saída para cada questão, sem textos introdutórios:
    
    ### Questão [Número]
    **[Enunciado da questão bem elaborado]**
    
    a) [Alternativa A]
    b) [Alternativa B]
    c) [Alternativa C]
    d) [Alternativa D]
    e) [Alternativa E]
    
    > **Gabarito:** [Letra Correta] - [Explicação detalhada do porquê esta é a correta e as outras não]
    
    ---
    (Repita para todas as {qtd} questões)
    """
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
    )
    
    return qa.invoke({"query": prompt_template})['result']

# --- Interface Principal ---

st.title("🎓 Tutor IA: Simulados Inteligentes")

if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "messages" not in st.session_state: st.session_state.messages = []
if "quiz_history" not in st.session_state: st.session_state.quiz_history = []

# Upload
uploaded_files = st.file_uploader("Carregar Apostilas (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("📚 Processar e Estudar Material"):
        with st.spinner("O Professor está lendo seus arquivos..."):
            st.session_state.vector_db = process_files(uploaded_files)
            st.success("Tudo pronto! Vamos estudar.")

# Abas de Funcionalidade
if st.session_state.vector_db:
    tab1, tab2 = st.tabs(["💬 Chat / Dúvidas", "📝 Gerador de Provas"])
    
    # --- ABA 1: CHAT ---
    with tab1:
        st.subheader("Tire suas dúvidas")
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])
            
        if prompt := st.chat_input("Pergunte algo..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Consultando..."):
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}))
                    resp = qa.invoke({"query": prompt})['result']
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

    # --- ABA 2: SIMULADO MELHORADO ---
    with tab2:
        st.subheader("Montar Simulado")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            topic = st.text_input("Assunto da Prova", placeholder="Ex: Direito Constitucional, Anatomia...")
        with col2:
            # NOVO: Slider para escolher quantidade
            qtd_questoes = st.slider("Qtd. Questões", min_value=1, max_value=10, value=3)
        
        if st.button("🎯 Criar Prova Agora"):
            if topic:
                with st.spinner(f"Elaborando {qtd_questoes} questões inéditas..."):
                    try:
                        raw_quiz = generate_quiz(topic, qtd_questoes, st.session_state.vector_db)
                        # Salva no histórico
                        st.session_state.quiz_history.insert(0, {"topic": topic, "content": raw_quiz, "qtd": qtd_questoes})
                    except Exception as e:
                        st.error(f"Erro: {e}")
            else:
                st.warning("Diga qual o assunto da prova.")
        
        # Renderização Inteligente das Questões
        if st.session_state.quiz_history:
            st.markdown("---")
            st.write("### 📂 Seus Simulados Gerados")
            
            for i, quiz in enumerate(st.session_state.quiz_history):
                # Expander para cada simulado gerado (Histórico)
                with st.expander(f"📝 Prova de {quiz['topic']} ({quiz['qtd']} questões)", expanded=(i==0)):
                    
                    # Aqui está o truque para separar o Gabarito
                    # Vamos tentar separar as questões visualmente
                    questoes = quiz['content'].split("---")
                    
                    for q in questoes:
                        if q.strip(): # Se não for vazio
                            # Tenta separar a resposta do enunciado (Baseado no prompt "> Gabarito:")
                            partes = q.split("> **Gabarito:**")
                            
                            if len(partes) == 2:
                                enunciado = partes[0]
                                resposta = partes[1]
                                
                                st.markdown(enunciado)
                                # Esconde a resposta num botão "Ver Resposta"
                                with st.expander("👁️ Ver Gabarito e Explicação"):
                                    st.info(f"**Resposta Correta:** {resposta}")
                            else:
                                # Se a IA não formatou perfeito, mostra tudo
                                st.markdown(q)
                            
                            st.markdown("---") # Linha separadora entre questões

elif not uploaded_files:
    st.info("👆 Carregue seus PDFs para começar.")
