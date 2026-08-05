import os
import re
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "treinamento_rag"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
K = 10

ABBREVIATIONS = {
    "iss": "ISS",
    "iptu": "IPTU",
    "itbi": "ITBI",
    "ufir": "UFIR",
}

def normalize_query(query):
    normalized = query
    for abbr, upper in ABBREVIATIONS.items():
        normalized = re.sub(rf"\b{abbr}\b", upper, normalized, flags=re.IGNORECASE)
    return normalized

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore():
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )

def ensure_indexed():
    from src import ingest
    try:
        count = get_vectorstore()._collection.count()
        if count == 0:
            raise ValueError("Base vetorial vazia.")
    except Exception:
        print("Base vetorial ausente ou vazia. Indexando documentos...")
        ingest.main_indexing()

def ask(query, k=K):
    vectorstore = get_vectorstore()
    search_result = vectorstore.similarity_search(query=normalize_query(query), k=k)

    context = ""
    mappings = {}
    list_res = []
    for i, res in enumerate(search_result):
        context += f"{i}\n{res.page_content}\n\n"
        path = res.metadata.get("path")
        mappings[i] = path
        list_res.append({"id": i, "path": path, "content": res.page_content})

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-v4-flash"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
    )

    rolemsg = {
        "role": "system",
        "content": (
            "Você é um assistente técnico especializado. Responda à pergunta do usuário "
            "usando estritamente os documentos fornecidos no contexto. No contexto estão "
            "documentos que devem conter a resposta. Sempre faça referência ao ID do "
            "documento (entre colchetes, por exemplo [0],[1]) do documento que foi usado "
            "para responder. Use quantas citações e documentos forem necessários para "
            "responder à pergunta. Se a informação não estiver no contexto, diga "
            "claramente que não possui essa informação na base de dados."
        ),
    }

    messages = [
        rolemsg,
        {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {query}"},
    ]

    resposta = llm.invoke(messages)
    return {"context": list_res, "answer": resposta.content}
