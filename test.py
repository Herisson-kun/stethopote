import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereRerank
from langchain_core.documents import Document # Nouvel import recommandé

# 1. Charger les clés
load_dotenv()

def test_services():
    print("🚀 Démarrage du test des API...")

    # TEST OPENAI
    try:
        llm = ChatOpenAI(model="gpt-4o-mini")
        res = llm.invoke("Réponds 'OpenAI OK' en un mot.")
        print(f"✅ OpenAI : {res.content}")
    except Exception as e:
        print(f"❌ Erreur OpenAI : {e}")

    # TEST COHERE
    try:
        # On simule un petit reranking
        reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=2)
        docs = [
            Document(page_content="Le chat mange du poisson"),
            Document(page_content="La pneumonie")
        ]
        # On cherche quel doc parle de médecine
        query = "Infection pulmonaire"
        compressed_docs = reranker.compress_documents(docs, query)
        
        if "pneumonie" in compressed_docs[0].page_content:
            print("✅ Cohere Rerank : OK")
        print(compressed_docs)
    except Exception as e:
        print(f"❌ Erreur Cohere : {e}")

if __name__ == "__main__":
    test_services()