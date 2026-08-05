import sys
from src.rag import ask

def main():
    if len(sys.argv) < 2:
        print('Uso: python -m src.query "Sua pergunta aqui"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"\n🔍 Buscando no ChromaDB e consultando DeepSeek para: '{query}'\n")

    result = ask(query)
    print("🤖 Resposta:")
    print(result["answer"])

if __name__ == "__main__":
    main()
