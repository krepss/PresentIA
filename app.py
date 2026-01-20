import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
import tempfile

# --- Configuração ---
st.set_page_config(page_title="Chat com PDF (RAG)", page_icon="🧠", layout="wide")

# CSS para estilo de chat
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("🧠 Configuração do Cérebro")
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("Chave de API Carregada")
    except:
        api_key = st.text_input("Sua Groq API Key", type="password")

    st.markdown("---")
    st.info("Este sistema lê seu PDF, 'quebra' ele em pedaços pequenos, indexa o conteúdo e permite que a IA consulte esses pedaços para responder.")

# --- Funções de RAG ---

@st.cache_resource
def get_embeddings():
    # Usa um modelo gratuito e leve do HuggingFace para criar os vetores
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    # Salva o arquivo temporariamente para o LangChain ler
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 1. Carregar
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # 2. Dividir (Chunking) - Crucial para RAG
    # Quebramos o texto em blocos de 1000 caracteres
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 3. Vetorizar e Criar Banco de Dados
    embeddings = get_embeddings()
    db = FAISS.from_documents(texts, embeddings)
    
    os.remove(tmp_path) # Limpa o arquivo temporário
    return db

# --- Interface Principal ---
st.title("🧠 Converse com seus Documentos")
st.markdown("Faça upload de um contrato, manual ou livro e **tire dúvidas com a IA**.")

# Estado da Sessão (Para manter o histórico do chat)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Upload
uploaded_file = st.file_uploader("Carregar Documento (PDF)", type="pdf")

if uploaded_file and st.session_state.vector_db is None:
    if st.button("Processar Documento"):
        with st.spinner("Indexando conhecimento... (Isso pode levar alguns segundos)"):
            try:
                st.session_state.vector_db = process_pdf(uploaded_file)
                st.success("Documento aprendido! Agora você pode perguntar.")
            except Exception as e:
                st.error(f"Erro ao processar: {e}")

# Área de Chat
if st.session_state.vector_db and api_key:
    # Mostra histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Pergunte algo sobre o documento..."):
        # 1. Adiciona pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Processa a resposta
        with st.chat_message("assistant"):
            with st.spinner("Consultando o documento..."):
                # Configura o LLM (Groq Llama 3)
                llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192")
                
                # Cria a cadeia de recuperação (Retrieval Chain)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}), # Busca os 3 trechos mais relevantes
                    return_source_documents=True
                )
                
                # Gera resposta
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                
                st.markdown(answer)
                
                # Feature Inovadora: Mostrar Fonte
                with st.expander("📚 Ver Fontes Consultadas"):
                    for doc in response['source_documents']:
                        st.caption(f"Conteúdo: {doc.page_content[:200]}...")
                        st.caption(f"Página: {doc.metadata.get('page', 'Desconhecida')}")
                        st.divider()

        # 3. Salva resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif not uploaded_file:
    st.info("👆 Comece enviando um arquivo PDF lá em cima.")
