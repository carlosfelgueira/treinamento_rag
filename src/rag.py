import os
import re
import threading
import functools

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "treinamento_rag"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
K = 10

SYNONYMS = {
    "iptu": "IPTU Imposto Predial e Territorial Urbano",
    "imposto predial e territorial urbano": "IPTU Imposto Predial e Territorial Urbano",
    "imposto predial": "Imposto Predial IPTU",
    "imposto territorial urbano": "Imposto Territorial Urbano IPTU",
    "imposto imobiliário": "Imposto Imobiliário IPTU",
    "iss": "ISS Imposto Sobre Serviços",
    "imposto sobre serviços": "ISS Imposto Sobre Serviços",
    "imposto sobre servico": "ISS Imposto Sobre Serviços",
    "itbi": "ITBI Imposto sobre Transmissão de Bens Imóveis",
    "imposto sobre transmissão": "ITBI Imposto sobre Transmissão de Bens Imóveis",
    "transmissão de bens imóveis": "ITBI Transmissão de Bens Imóveis",
    "ufir": "UFIR Unidade Fiscal de Referência",
    "unidade fiscal de referência": "UFIR",
    "ccm": "CCM Cadastro de Contribuintes Mobiliários",
    "cadastro imobiliário": "Cadastro Imobiliário IPTU",
    "taxa de fiscalização": "Taxa de Fiscalização de Localização, Instalação e Funcionamento",
    "divida ativa": "Dívida Ativa Cobrança",
    "dívida ativa": "Dívida Ativa Cobrança",
    "divida": "Dívida",
    "aliquota": "alíquota",
    "construcao": "construção",
    "taxas imobiliarias": "Taxas Imobiliárias IPTU",
    "taxas imobiliárias": "Taxas Imobiliárias IPTU",
    "imobiliarias": "imobiliárias",
}

_index_lock = threading.Lock()
_index_tried = False

def normalize_query(query):
    normalized = query
    placeholders = []
    for term in sorted(SYNONYMS, key=len, reverse=True):
        pattern = rf"\b{re.escape(term)}\b"
        marker = f"@@{len(placeholders)}@@"
        updated, n = re.subn(pattern, marker, normalized, flags=re.IGNORECASE)
        if n:
            normalized = updated
            placeholders.append(SYNONYMS[term])
    for i, value in enumerate(placeholders):
        normalized = normalized.replace(f"@@{i}@@", value)
    return normalized

@functools.lru_cache(maxsize=1)
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
    global _index_tried
    if _index_tried:
        return
    with _index_lock:
        if _index_tried:
            return
        try:
            count = get_vectorstore()._collection.count()
            if count == 0:
                raise ValueError("Base vetorial vazia.")
        except Exception:
            print("Base vetorial ausente ou vazia. Indexando documentos...")
            from src import ingest
            ingest.main_indexing()
        finally:
            _index_tried = True

def ask(query, k=K, history=None):
    ensure_indexed()
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

    messages = [rolemsg]
    for turn in (history or []):
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append(
        {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {query}"}
    )

    resposta = llm.invoke(messages)
    return {"context": list_res, "answer": resposta.content}
