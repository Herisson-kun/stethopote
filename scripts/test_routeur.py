import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, List
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================================
# 1. DÉFINITION DU SCHÉMA DE SORTIE (ÉPURÉ STRICT)
# ==========================================
class RouterDecision(BaseModel):
    intention: Literal["question_specifique", "resume_item", "resume_maladie", "hors_sujet"] = Field(
        description="L'intention de l'utilisateur. ATTENTION : 'resume_item' est réservé UNIQUEMENT quand l'utilisateur demande le cours COMPLET d'un item. S'il pose une question sur un point précis, même en citant un item, c'est 'question_specifique'."
    )
    items_cibles: List[str] = Field(
        description="Liste exacte des numéros d'items mentionnés pour le filtrage. Toujours formater avec une majuscule (ex: ['Item 156', 'Item 37']). Laisser vide si aucun n'est précisé."
    )
    # 🗑️ Le champ mots_cles a été supprimé pour éviter la pollution et économiser des tokens.
    requete_optimisee: str = Field(
        description="La requête reformulée de manière autonome. SANS les mentions de l'item (puisqu'ils sont filtrés à part dans items_cibles). SANS le jargon abrégé."
    )

# ==========================================
# 2. LE PROMPT SYSTÈME DU ROUTEUR (CHIRURGICAL)
# ==========================================
SYSTEM_PROMPT = """Tu es le Cerveau Routeur de "Stéthopote", un assistant médical RAG pour étudiants en médecine.
Ton rôle n'est PAS de répondre à la question médicale. Ton rôle est d'analyser la requête utilisateur pour préparer le moteur de recherche.
Tu es expert en abréviations médicales françaises, utilise le contexte pour comprendre les abreviations (ex: ttt = traitement, proba = probabiliste, gpe = groupe, bcp = beaucoup, pt = patient).

RÈGLES DE REFORMULATION ET D'EXTRACTION :
1. CORÉFÉRENCE : Si l'utilisateur dit "quel est son ttt ?", utilise l'historique pour remplacer "son" par la maladie concernée.
2. ACRONYMES : Développe TOUS les acronymes et le jargon médical en gardant l'abréviation entre parenthèses (ex: "ttt de l'EP" -> "traitement (ttt) de l'embolie pulmonaire (EP)"). Attention au contexte pour les termes métiers (ex: "ttt proba" = "traitement probabiliste").
3. NETTOYAGE : NE MENTIONNE JAMAIS les mots "item", "chapitre" ou leurs numéros dans la 'requete_optimisee'. Ces métadonnées sont gérées exclusivement par 'items_cibles'. La requête optimisée doit rester purement clinique.
4. FILTRAGE PAR ITEM : Si l'utilisateur mentionne explicitement un ou plusieurs items (ex: "dans l'item 156", "items 36 et 37", "résumé item 230"), tu DOIS extraire ces numéros et les formater strictement sous la forme "Item XXX" dans la liste `items_cibles`.
"""

# ==========================================
# 3. LA FONCTION DU ROUTEUR
# ==========================================
def analyser_requete(historique: list, nouvelle_question: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(historique)
    messages.append({"role": "user", "content": nouvelle_question})

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=RouterDecision,
        temperature=0.0 
    )
    
    return completion.choices[0].message.parsed

# ==========================================
# 4. LABORATOIRE DE TESTS (PLAYGROUND)
# ==========================================
if __name__ == "__main__":
    print("🚀 DÉMARRAGE DU TEST DU ROUTEUR LLM...\n")

    # TEST 1 : La coréférence et l'acronyme ("ttt" et "ce truc")
    print("--- TEST 1 : Suivi de conversation et Acronymes ---")
    historique_1 = [
        {"role": "user", "content": "C'est quoi l'endocardite infectieuse ?"},
        {"role": "assistant", "content": "C'est une infection de l'endocarde, souvent bactérienne, diagnostiquée notamment avec les critères de Duke."}
    ]
    question_1 = "quel est le ttt proba de cette pathologie ?"
    
    resultat_1 = analyser_requete(historique_1, question_1)
    print(f"Question brute : {question_1}")
    print(json.dumps(resultat_1.model_dump(), indent=2, ensure_ascii=False))
    print("\n")


    # TEST 2 : La demande de cours complet (Intention)
    print("--- TEST 2 : Demande de résumé d'un Item ---")
    historique_2 = [] # Pas d'historique
    question_2 = "Fais moi un résumé complet de l'item 230 stp"
    
    resultat_2 = analyser_requete(historique_2, question_2)
    print(f"Question brute : {question_2}")
    print(json.dumps(resultat_2.model_dump(), indent=2, ensure_ascii=False))
    print("\n")


    # TEST 3 : Le Hors Sujet
    print("--- TEST 3 : Hors Sujet ---")
    historique_3 = []
    question_3 = "Salut Stéthopote, tu vas bien ?"
    
    resultat_3 = analyser_requete(historique_3, question_3)
    print(f"Question brute : {question_3}")
    print(json.dumps(resultat_3.model_dump(), indent=2, ensure_ascii=False))
    print("\n")

    # TEST 4 : Demande spécifique avec filtrage multiple d'items
    print("--- TEST 4 : Filtrage sur plusieurs Items ---")
    historique_4 = []
    question_4 = "Quels sont les signes cliniques de l'insuffisance cardiaque ? Ne cherche que dans les items 234 et 235 stp."
    
    resultat_4 = analyser_requete(historique_4, question_4)
    print(f"Question brute : {question_4}")
    print(json.dumps(resultat_4.model_dump(), indent=2, ensure_ascii=False))
    print("\n")
    
    # TEST 5 : Suite de conversation avec item précisé
    print("--- TEST 5 : Historique + Filtrage Item ---")
    historique_5 = [
        {"role": "user", "content": "Quelles sont les étiologies de l'anémie ?"},
        {"role": "assistant", "content": "L'anémie peut être centrale ou périphérique..."}
    ]
    question_5 = "Fais un focus sur la macrocytaire d'après l'item 214"
    
    resultat_5 = analyser_requete(historique_5, question_5)
    print(f"Question brute : {question_5}")
    print(json.dumps(resultat_5.model_dump(), indent=2, ensure_ascii=False))
    print("\n")