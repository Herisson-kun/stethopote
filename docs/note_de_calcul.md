# 📄 NOTE DE DÉCISION ARCHITECTURALE (FINOPS)

**Objet :** Évaluation des coûts d'API (OpenAI GPT-5.6) et justification de la politique de mémorisation (Sliding Window vs Stateless).  
**Projet :** Stéthopote (RAG Médical)  
**Date :** 15 Août 2026  

---

## 1. Hypothèses de Dimensionnement (Payload par Requête)

Pour chaque interaction utilisateur, l'architecture "Dual-Brain" (Routeur + Générateur) traitera les volumes de tokens suivants :
- **Historique Glissant (2 tours) :** ~500 tokens (Questions + Réponses précédentes).
- **Nouveau Contexte RAG (8 Parents Chunks) :** ~5 000 tokens.
- **Nouvelle Requête Utilisateur :** ~50 tokens.
- **Output du Routeur (JSON structuré) :** ~50 tokens.
- **Output du Générateur (Réponse finale) :** ~200 tokens.
- **Volume d'utilisation estimé :** 20 requêtes/jour (600 requêtes/mois).

---

## 2. Grille Tarifaire (Famille GPT-5.6)

*Tarifs exprimés en USD pour 1 Million de tokens.*

| Modèle | Input | Output | Cached Input | Rôle dans l'architecture |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-5.6-Luna** | 0.20 $ | 1.20 $ | 0.02 $ | Moteur de Routage (Rapide & Logique) |
| **GPT-5.6-Terra** | 2.00 $ | 12.00 $ | 0.20 $ | Moteur de Génération (Expert & Empathique) |

---

## 3. Modélisation des Coûts Mensuels (Base 600 requêtes)

### Scénario A : Architecture Hybride SOTA (Luna Routeur + Terra Générateur)
*Ici, les DEUX modèles reçoivent l'historique des 2 derniers tours pour garantir une fluidité conversationnelle absolue.*
- **Coût Routeur (Luna) :** Input (550) + Output (50) = 0,00017 $ / req.
- **Coût Générateur (Terra) :** Input (5 550) + Output (200) = 0,01350 $ / req.
- **Total par requête :** 0,01367 $
- **Total Mensuel : 8,20 $ / mois**

### Scénario B : Architecture Hybride Stateless (Mode Amnésique)
*Aucun historique n'est conservé ni envoyé. Le système agit comme un moteur de recherche strict.*
- **Coût Routeur (Luna) :** Input (50) + Output (50) = 0,00007 $ / req.
- **Coût Générateur (Terra) :** Input (5 050) + Output (200) = 0,01250 $ / req.
- **Total par requête :** 0,01257 $
- **Total Mensuel : 7,54 $ / mois**

### Scénario C : Dégradation "Full Luna" avec Historique (Luna + Luna)
*Évaluation d'une version "Low-Cost" de Stéthopote, en gardant l'historique premium.*
- **Coût Routeur (Luna) :** 0,00017 $ / req.
- **Coût Générateur (Luna) :** Input (5 550) + Output (200) = 0,00135 $ / req.
- **Total par requête :** 0,00152 $
- **Total Mensuel : 0,91 $ / mois**

---

## 4. Analyse du Delta ("Le Coût de l'UX")

La différence de coût mathématique entre une application amnésique (Scénario B) et une application dotée d'une intelligence conversationnelle premium (Scénario A) s'élève à :
> **(8,20 $ - 7,54 $) = + 0,66 $ par mois.**

Ce delta de 66 centimes représente **exclusivement** le coût de facturation des 500 tokens d'historique envoyés aux modèles. Ce qui représente le centre de coût principal de l'architecture n'est pas la mémoire conversationnelle, mais la lourdeur du Contexte RAG (les 5 000 tokens des 8 Parent Chunks, représentant ~80% de la facture du modèle Terra).

---

## 5. Décision Architecturale Finale

**✅ Recommandation retenue : DÉPLOIEMENT DU SCÉNARIO A (Hybride avec Historique Glissant de 2 tours).**

**Justification :** 
1. **Bénéfice Utilisateur (UX) :** L'historique permet la résolution des coréférences (ex: *"Quelles sont les contre-indications pour ce traitement ?"*) et offre un ton empathique et suivi, crucial pour l'adoption et la rétention de l'outil par l'étudiante.
2. **Rationalité Financière :** Priver le LLM de l'historique conversationnel ne permet d'économiser que ~8% de la facture totale (0,66 $). Une telle mutilation de l'expérience utilisateur n'est pas justifiable par une économie aussi marginale. 
3. **Sécurité Scalable :** En limitant la fenêtre glissante à 2 tours stricts (les 4 derniers messages), nous créons une borne asymptotique financière. Le payload d'entrée n'explosera jamais, garantissant la maîtrise des coûts quelle que soit la durée de la session.