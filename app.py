import streamlit as st
import os
import tempfile
import pdfplumber  # <--- NOVA BIBLIOTECA (Mais robusta)
from langchain_groq import ChatGroq
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
    st.info("Usando motor **pdfplumber** para máxima compatibilidade.")
    if st.button("Limpar Memória"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.rerun()

# --- Funções de RAG ---

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_multiple_pdfs(uploaded_files):
    all_documents = []
    status_text = st.empty()
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Lendo arquivo {index + 1}/{total_files}: {uploaded_file.name}")
        progress_bar.progress((index + 1) / total_files)

        # Cria arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # --- MUDANÇA PRINCIPAL: USANDO PDFPLUMBER ---
            with pdfplumber.open(tmp_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            doc = Document(
                                page_content=text, 
                                metadata={"source": uploaded_file.name, "page": i + 1}
                            )
                            all_documents.append(doc)
                    except Exception as e:
                        print(f"Erro na página {i} do arquivo {uploaded_file.name}: {e}")
                        continue
                        
        except Exception as e:
            st.error(f"Erro crítico ao abrir {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    status_text.text("Indexando conteúdo...")
    
    if not all_documents:
        # Se falhar tudo, não trava o app, só avisa
        return None

    # Dividir e Vetorizar
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    embeddings = get_embeddings()
    db = FAISS.from_documents(texts, embeddings)
    
    progress_bar.empty()
    status_text.empty()
    return db

# --- Interface Principal ---
st.title("📚 Chat com Múltiplos Arquivos")
st.markdown("Envie seus PDFs complexos. O sistema usa leitura avançada.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

uploaded_files = st.file_uploader(
    "Carregar Documentos", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🚀 Processar Arquivos"):
        with st.spinner("Processando..."):
            try:
                db_result = process_multiple_pdfs(uploaded_files)
                if db_result:
                    st.session_state.vector_db = db_result
                    st.success(f"Processamento concluído!")
                else:
                    st.error("Não foi possível ler texto dos arquivos enviados. Eles podem ser imagens escaneadas.")
            except Exception as e:
                st.error(f"Erro: {e}")

# Área de Chat
if st.session_state.vector_db:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pergunte..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 4}),
                        return_source_documents=True
                    )
                    
                    response = qa_chain.invoke({"query": prompt})
                    answer = response['result']
                    
                    st.markdown(answer)
                    
                    with st.expander("📚 Ver Fontes"):
                        for doc in response['source_documents']:
                            st.caption(f"Arquivo: {doc.metadata.get('source')} | Pág: {doc.metadata.get('page')}")
                            st.caption(f"...{doc.page_content[:150]}...")
                            st.divider()
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erro: {e}")

elif not uploaded_files:
    st.info("👆 Envie arquivos para começar.")
