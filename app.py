import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
# CORREÇÃO AQUI: Importação atualizada para novas versões
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Tenta importar do caminho antigo ou do novo para garantir
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_community.chains import RetrievalQA
# --- Configuração da Página ---
st.set_page_config(page_title="Chat com PDF (RAG)", page_icon="🧠", layout="wide")

# --- CSS Personalizado ---
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 10px; }
    .stButton>button { width: 100%; background-color: #7c3aed; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Barra Lateral ---
with st.sidebar:
    st.header("🧠 Configuração")
    
    # Verifica se a chave está nos segredos ou pede ao usuário
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ Chave de API detectada")
    else:
        api_key = st.text_input("Digite sua Groq API Key", type="password")
        if not api_key:
            st.warning("⚠️ Insira a chave para começar.")

    st.markdown("---")
    st.info("Este sistema lê seu PDF, cria um índice de busca e usa IA para responder perguntas baseadas no documento.")
    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

# --- Funções de RAG (Cérebro do App) ---

@st.cache_resource
def get_embeddings():
    # Usa modelo gratuito e leve rodando na CPU
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    # Cria arquivo temporário para leitura
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 1. Carregar
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 2. Dividir (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # 3. Criar Banco Vetorial
        embeddings = get_embeddings()
        db = FAISS.from_documents(texts, embeddings)
        
        return db
    finally:
        # Garante que o arquivo temporário seja deletado
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Interface Principal ---
st.title("🧠 Converse com seus Documentos")
st.markdown("Faça upload de um **PDF** e tire dúvidas com a Inteligência Artificial.")

# Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Área de Upload
uploaded_file = st.file_uploader("Carregar Documento", type="pdf")

if uploaded_file:
    # Processa o arquivo apenas se mudou ou se ainda não foi processado
    if st.button("🚀 Processar Documento"):
        with st.spinner("Lendo e indexando..."):
            try:
                st.session_state.vector_db = process_pdf(uploaded_file)
                st.success("Documento pronto! Pergunte abaixo.")
            except Exception as e:
                st.error(f"Erro ao processar: {e}")

# Área de Chat
if st.session_state.vector_db and api_key:
    # Exibe histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Pergunte sobre o arquivo..."):
        # Adiciona pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera resposta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192")
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    
                    response = qa_chain.invoke({"query": prompt})
                    answer = response['result']
                    
                    st.markdown(answer)
                    
                    # Fontes
                    with st.expander("📚 Fontes Consultadas"):
                        for doc in response['source_documents']:
                            st.caption(f"Conteúdo: {doc.page_content[:150]}...")
                            st.caption(f"Página: {doc.metadata.get('page', 'N/A')}")
                            st.divider()
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {e}")

elif not uploaded_file:
    st.info("👆 Comece enviando um arquivo PDF.")
