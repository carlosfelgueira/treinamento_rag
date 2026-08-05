import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.rag import ask, ensure_indexed

warnings.filterwarnings("ignore")

for key, value in st.secrets.items():
    if isinstance(value, str) and not os.environ.get(key):
        os.environ[key] = value

st.set_page_config(page_title="TREINAMENTO RAG", page_icon=":100:", layout="centered")

st.title("_:green[TREINAMENTO RAG]_")

try:
    ensure_indexed()
except Exception as e:
    st.error(f"Erro ao preparar a base de dados: {e}")
    st.stop()

question = st.text_input("Digite Uma Pergunta Para a IA Executar Consulta nos Documentos:", "")

if st.button("Enviar"):

    st.write("A pergunta foi: \"" + question + "\"")

    with st.spinner("Consultando os documentos..."):
        result = ask(question)
        st.markdown(result["answer"])
