import os
from dotenv import load_dotenv
import cohere
from unittest.mock import patch # La librairie magique pour simuler des pannes

# ==========================================
# 0. CHARGEMENT DES CLÉS ET CLIENTS
# ==========================================
load_dotenv()

TRIAL_KEY = os.getenv("COHERE_TRIAL_API_KEY")
PROD_KEY = os.getenv("COHERE_PROD_API_KEY")

co_client = cohere.Client(TRIAL_KEY)

# ==========================================
# 1. PRÉPARATION DU PIÈGE
# ==========================================
query = "Quels sont les traitements immédiats de la crise d'asthme sévère ?"
documents = [
    "L'appendicite aiguë se caractérise par une douleur en fosse iliaque droite, des nausées et une fébricule.",
    "Le diagnostic de l'asthme repose sur l'exploration fonctionnelle respiratoire (EFR) qui met en évidence un trouble ventilatoire obstructif réversible.",
    "Le traitement de fond de l'asthme léger repose sur l'inhalation quotidienne de corticoïdes à faible dose pour contrôler l'inflammation de fond.",
    "En urgence, la crise d'asthme sévère nécessite une oxygénothérapie, l'administration de bêta-2 mimétiques d'action courte en nébulisation (salbutamol) et des corticoïdes par voie systémique.",
    "Le traitement de l'exacerbation de BPCO sévère nécessite de l'oxygène à faible débit, des bronchodilatateurs et parfois une antibiothérapie."
]

# ==========================================
# 2. FONCTION DE TEST (Inchongée)
# ==========================================
def tester_reranker(query, docs):
    print(f"🔍 Requête : '{query}'\n")
    
    try:
        print("⚖️ Lancement du Reranking (Clé TRIAL)...")
        response = co_client.rerank(
            model="rerank-v4.0-pro",
            query=query,
            documents=docs,
            top_n=2,
            return_documents=True
        )
        print("✅ Réussite avec la clé Trial !")
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Le signal magique : Erreur 429
        if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
            print(f"⚠️ DÉTECTION 429 : Clé Trial épuisée ou Rate Limit atteint.")
            
            if not PROD_KEY:
                raise ValueError("❌ Clé Trial épuisée, mais aucune clé PROD n'est configurée !")
            
            print("🔄 Bascule automatique sur la clé de PRODUCTION...")
            co_prod = cohere.Client(PROD_KEY)
            
            # APPEL RÉEL AVEC LA CLÉ PROD
            response = co_prod.rerank(
                model="rerank-v4.0-pro",
                query=query,
                documents=docs,
                top_n=2,
                return_documents=True
            )
            print("✅ Réussite avec la clé Production !")
        else:
            raise e

    print("\n🏆 Résultats :")
    for hit in response.results:
        print(f"Score: {hit.relevance_score:.4f} -> {hit.document.text[:50]}...")

# ==========================================
# 3. LABORATOIRE DE TESTS (Avec Mock)
# ==========================================
if __name__ == "__main__":
    
    # --- TEST 1 : FONCTIONNEMENT NORMAL ---
    print("=========================================")
    print("🔵 TEST 1 : COMPORTEMENT NORMAL")
    print("=========================================")
    tester_reranker(query, documents)
    print("\n\n")

    # --- TEST 2 : SIMULATION DE CRASH (CODE 429) ---
    print("=========================================")
    print("🔴 TEST 2 : SIMULATION DE CRASH (FALLBACK)")
    print("=========================================")
    
    # 1. On crée une fausse fonction qui lève instantanément une erreur 429
    def fake_429_error(*args, **kwargs):
        raise Exception("Cohere API Error: status_code: 429, message: Too many requests")

    # 2. Le 'patch' remplace temporairement la vraie fonction 'rerank' du client Trial par notre fausse fonction
    with patch.object(co_client, 'rerank', side_effect=fake_429_error):
        # Quand tester_reranker va appeler co_client.rerank(), ça va crasher direct avec un 429.
        # Le 'except' va s'activer, instancier la clé PROD, et faire un VRAI appel !
        tester_reranker(query, documents)