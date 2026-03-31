import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pickle

load_dotenv()

# 1. Configuration des Embeddings (doit être identique à l'indexation)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Re-connexion à Chroma (Le magasin des "children")
vectorstore = Chroma(
    collection_name="stethopote_med",
    embedding_function=embeddings,
    persist_directory="store/chroma_db"
)

# --- TEST DE RECHERCHE SUR LES CHILDREN ---
query = "Quelle est la prise en charge des tuberculoses latentes?"
print(f"🩺 Question : {query}")
print(f"🔍 Recherche des 20 meilleurs 'children' (chunks de ~200 tokens)...\n")

try:
    # On utilise similarity_search_with_score pour voir la "distance"
    # Plus le score est bas, plus c'est proche sémantiquement
    results = vectorstore.similarity_search_with_score(query, k=20)

    if results:
        print(f"✅ Top 20 des segments trouvés :\n")
        for i, (doc, score) in enumerate(results, 1):
            # Nettoyage rapide du texte pour l'affichage (enlève les retours à la ligne trop brusques)
            content_preview = doc.page_content.replace('\n', ' ')
            
            print(f"[{i}] Score: {score:.4f}")
            print(f"    📄 Extrait : {content_preview}...")
            # On affiche la source si elle est dans les métadonnées
            source = doc.metadata.get('source', 'Inconnue')
            print(f"    📍 Source : {source}")
            print("-" * 50)
    else:
        print("❌ Aucun petit chunk trouvé dans Chroma.")

except Exception as e:
    print(f"❌ Erreur lors de la recherche des enfants : {e}")