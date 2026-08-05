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
st.caption("Tire suas dúvidas sobre o código tributário do município de São Sebastião.")

try:
    ensure_indexed()
except Exception as e:
    st.error(f"Erro ao preparar a base de dados: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite uma pergunta sobre os documentos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Consultando os documentos..."):
        history = st.session_state.messages[:-1]
        result = ask(prompt, history=history)
        answer = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
