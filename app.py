import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Configuração da Página ---
st.set_page_config(page_title="Chat Multi-PDFs", page_icon="📚", layout="wide")

# --- CSS Personalizado ---
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 10px; }
    .stButton>button { width: 100%; background-color: #7c3aed; color: white; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA CHAVE ---
api_key = "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"

# --- Barra Lateral ---
with st.sidebar:
    st.header("📚 Biblioteca")
    st.success("✅ Modelo Llama 3.3 Ativo")
    
    st.markdown("---")
    st.info("Agora você pode enviar **vários arquivos** de uma vez! A IA lerá todos eles.")
    if st.button("Limpar Memória"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.rerun()

# --- Funções de RAG (Cérebro do App) ---

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_multiple_pdfs(uploaded_files):
    # Lista para guardar todo o texto de todos os arquivos
    all_documents = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Lendo arquivo {index + 1}/{total_files}: {uploaded_file.name}")
        progress_bar.progress((index + 1) / total_files)

        # Cria arquivo temporário para cada PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            reader = PdfReader(tmp_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # AGORA ADICIONAMOS O NOME DO ARQUIVO (SOURCE) NOS DADOS
                    doc = Document(
                        page_content=text, 
                        metadata={"source": uploaded_file.name, "page": i + 1}
                    )
                    all_documents.append(doc)
        except Exception as e:
            st.error(f"Erro ao ler {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    status_text.text("Organizando conhecimento e criando índices...")
    
    if not all_documents:
        raise ValueError("Nenhum texto foi extraído dos arquivos.")

    # Dividir e Vetorizar (Tudo junto)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    embeddings = get_embeddings()
    db = FAISS.from_documents(texts, embeddings)
    
    progress_bar.empty()
    status_text.empty()
    return db

# --- Interface Principal ---
st.title("📚 Chat com Múltiplos Arquivos")
st.markdown("Envie contratos, manuais e relatórios. A IA cruza as informações de todos eles.")

# Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Área de Upload (MODIFICADA PARA MULTIPLOS ARQUIVOS)
uploaded_files = st.file_uploader(
    "Carregar Documentos (Segure Ctrl para selecionar vários)", 
    type="pdf", 
    accept_multiple_files=True # <--- A MÁGICA ACONTECE AQUI
)

if uploaded_files:
    if st.button("🚀 Processar Todos os Arquivos"):
        # Só processa se o banco de dados estiver vazio ou se o usuário pedir
        with st.spinner("Processando biblioteca..."):
            try:
                st.session_state.vector_db = process_multiple_pdfs(uploaded_files)
                st.success(f"{len(uploaded_files)} documentos processados com sucesso!")
            except Exception as e:
                st.error(f"Erro crítico: {e}")

# Área de Chat
if st.session_state.vector_db:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pergunte sobre os documentos..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pesquisando nos arquivos..."):
                try:
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 4}), # Busca 4 trechos para ter mais contexto
                        return_source_documents=True
                    )
                    
                    response = qa_chain.invoke({"query": prompt})
                    answer = response['result']
                    
                    st.markdown(answer)
                    
                    # Fontes Melhoradas (Mostra qual arquivo)
                    with st.expander("📚 Fontes Consultadas"):
                        for doc in response['source_documents']:
                            # Mostra o nome do arquivo original
                            nome_arquivo = doc.metadata.get('source', 'Desconhecido')
                            pagina = doc.metadata.get('page', 'N/A')
                            
                            st.markdown(f"**Arquivo:** `{nome_arquivo}` | **Pág:** {pagina}")
                            st.caption(f"Trecho: {doc.page_content[:150]}...")
                            st.divider()
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erro: {e}")

elif not uploaded_files:
    st.info("👆 Selecione seus arquivos PDF acima para começar.")
