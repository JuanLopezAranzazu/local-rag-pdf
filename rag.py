import chromadb
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

import config

def is_indexed(pdf_path: str) -> bool:
    """
    Determina si el PDF ya ha sido indexado verificando la existencia de la colección en ChromaDB y si contiene documentos.
    """
    db_path = Path(config.CHROMA_DIR)
    if not db_path.exists():
        return False
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(_collection_name(pdf_path))
        return collection.count() > 0
    except Exception:
        return False
    

def index_pdf(pdf_path: str) -> Chroma:
    """Carga el PDF, lo fragmenta y lo guarda en ChromaDB."""
    docs = PyPDFLoader(pdf_path).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    ).split_documents(docs)

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_name=_collection_name(pdf_path),
    )
    vectordb.persist()
    return vectordb


def load_index(pdf_path: str) -> Chroma:
    """Carga un índice ya existente para el PDF dado."""
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    return Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=_collection_name(pdf_path),
    )


def build_qa(vectordb: Chroma) -> tuple[Chroma, Ollama]:
    """Construye la cadena de preguntas y respuestas."""
    retriever = vectordb.as_retriever(
        search_kwargs={"k": config.K_DOCS}
    )

    llm = Ollama(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE
    )

    return retriever, llm


def ask(qa, question: str) -> tuple[str, list]:
    """Realiza una pregunta y devuelve la respuesta junto con los documentos relevantes."""
    retriever, llm = qa

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = config.PROMPT.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    return response, docs


def _collection_name(pdf_path: str) -> str:
    from pathlib import Path
    return Path(pdf_path).stem.replace(" ", "_")