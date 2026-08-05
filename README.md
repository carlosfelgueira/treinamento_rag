# TREINAMENTO RAG

Sistema de **Busca Inteligente com IA Generativa e RAG** (Retrieval-Augmented Generation) para consulta em documentos legais/técnicos. O usuário faz uma pergunta em linguagem natural e o sistema recupera os trechos mais relevantes da base de documentos, envia o contexto para um LLM e retorna uma resposta com citação das fontes.

## Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.14 |
| Framework de LLM | LangChain |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace, 384 dims) |
| Banco vetorial | ChromaDB (similaridade por cosseno) |
| LLM | DeepSeek (`deepseek-v4-flash`) via API |
| API | FastAPI + Uvicorn |
| Web App | Streamlit |
| Documentos suportados | PDF, TXT, MD, DOCX, PPTX |

## Arquitetura

```
docs/  -->  src/ingest.py  -->  ChromaDB (chunks + embeddings)
                                     |
                    src/rag.py <-----+  (busca semântica k=10, normalização de siglas)
                        |                |__ src/query.py (CLI)
                        |__ src/api.py (FastAPI POST /query)
                                     |
                              src/web_app.py (Streamlit)
```

1. **Ingestão** (`src/ingest.py`): lê os documentos da pasta `docs/`, divide em chunks (500 tokens com overlap de 50) e gera embeddings locais via HuggingFace, gravando no ChromaDB.
2. **Busca** (`src/rag.py`): normaliza siglas da pergunta (`iss` → `ISS`), faz busca semântica no ChromaDB (k=10) e monta um contexto numerado.
3. **Geração**: o contexto é enviado ao DeepSeek com um prompt que instrui o modelo a responder **estritamente** com base nos documentos e a **citar as fontes** entre colchetes, ex. `[0]`.
4. **Apresentação**: via CLI (`query.py`), API (`api.py`) ou web app (`web_app.py`).

## Estrutura

```
├── docs/              # documentos-fonte (PDF, DOCX, PPTX, TXT, MD)
├── src/
│   ├── ingest.py      # indexação dos documentos no ChromaDB
│   ├── rag.py         # pipeline RAG (busca + contexto + LLM)
│   ├── query.py       # consulta via linha de comando
│   ├── api.py         # API FastAPI
│   └── web_app.py     # interface Streamlit
├── requirements.txt
└── .env               # credenciais (NÃO commitar)
```

## Como executar

1. Crie e ative o ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   .\venv\Scripts\activate       # Windows
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure as credenciais no arquivo `.env`:
   ```
   OPENAI_API_BASE="https://api.deepseek.com/v1"
   OPENAI_API_KEY="sua-chave-aqui"
   MODEL_NAME="deepseek-v4-flash"
   ```

4. Adicione os documentos em `docs/` e indexe:
   ```bash
   python -m src.ingest
   ```

5. Consulte:
   ```bash
   # CLI
   python -m src.query "Qual a alíquota do ISS para construção civil?"

   # API (em outro terminal)
   python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

   # Web App (em outro terminal)
   streamlit run src/web_app.py
   ```

> **Dica Windows:** defina `PYTHONIOENCODING=utf-8` antes de rodar para evitar erros com acentos/emoji no console.

## Exemplo de pergunta

> "Qual a alíquota do ISS para construção civil?"

Resposta obtida a partir dos documentos:
> O documento estabelece alíquotas mínima e máxima do ISS de **2% e 5%** (art. 88), com redução para **3%** no subitem 8.01 da Lista de Serviços (exercício de 2018) [0] e isenções para subitens específicos [1].
