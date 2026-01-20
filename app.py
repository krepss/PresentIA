import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
# Mudança: Vamos usar o pypdf direto em vez do loader do langchain
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Configuração da Página ---
st.set_page_config(page_title="Chat com PDF (RAG)", page_icon="🧠", layout="wide")

# --- CSS Personalizado ---
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 10px; }
    .stButton>button { width: 100%; background-color: #7c3aed; color: white; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA CHAVE ---
# A chave que você forneceu anteriormente
api_key = "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"

# --- Barra Lateral ---
with st.sidebar:
    st.header("🧠 Configuração")
    st.success("✅ Chave de API Embutida")
    
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
    # Cria arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 1. Carregar (MÉTODO ROBUSTO - MANUAL)
        # Substituímos o PyPDFLoader por uma leitura direta para evitar erro de 'bbox'
        reader = PdfReader(tmp_path)
        documents = []
        
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    # Criamos o objeto Document manualmente
                    doc = Document(page_content=text, metadata={"page": i + 1})
                    documents.append(doc)
            except Exception as e:
                # Se uma página der erro, pulamos ela e avisamos, mas não travamos o app
                print(f"Erro ao ler página {i+1}: {e}")
                continue

        if not documents:
            raise ValueError("Não foi possível extrair texto deste PDF. Ele pode ser uma imagem digitalizada.")

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
    # Processa o arquivo apenas se o botão for clicado
    if st.button("🚀 Processar Documento"):
        with st.spinner("Lendo e indexando..."):
            try:
                st.session_state.vector_db = process_pdf(uploaded_file)
                st.success("Documento pronto! Pergunte abaixo.")
            except Exception as e:
                st.error(f"Erro ao processar: {e}")
                st.warning("Dica: Se o PDF for uma imagem digitalizada (scanner), este método não funcionará. Use PDFs nativos.")

# Área de Chat
if st.session_state.vector_db:
    # Exibe histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Pergunte sobre o arquivo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

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
