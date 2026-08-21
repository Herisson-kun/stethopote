import streamlit as st
import time
import random
import os
import sys
import subprocess
import re
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

# --- DEBUG DIAGNOSTIQUE PINECONE ---
try:
    print("=== DÉBUT DIAGNOSTIQUE PACKAGES ===")
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
    print(result.stdout)
    print("=== FIN DIAGNOSTIQUE PACKAGES ===")
except Exception as e:
    print(f"Erreur pip list: {e}")
# -----------------------------------

from pinecone import Pinecone
from supabase import create_client, Client
from openai import OpenAI
import cohere
import tiktoken

# ==========================================
# 1. CONFIGURATION & CSS ADAPTATIF
# ==========================================
st.set_page_config(page_title="Stéthopote", page_icon="🩺", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stChatMessage { font-size: 1.05rem !important; }
    .source-box { 
        background-color: var(--secondary-background-color); 
        color: var(--text-color);
        padding: 12px; 
        border-radius: 8px; 
        border-left: 4px solid #4A90E2; 
        margin-bottom: 10px; 
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INITIALISATION DES CLIENTS (Mise en cache)
# ==========================================
# On charge d'abord les secrets de Streamlit, puis on fallback sur .env si en local
load_dotenv()

@st.cache_resource
def init_clients():
    # Helper function pour chercher d'abord dans st.secrets, puis dans os.environ
    def get_secret(key):
        return st.secrets.get(key) if key in st.secrets else os.environ.get(key)
    
    openai_client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    supabase = create_client(get_secret("SUPABASE_URL"), get_secret("SUPABASE_KEY"))
    pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
    pinecone_index = pc.Index(get_secret("PINECONE_INDEX_NAME") or "stethopote")
    
    # NOUVEAU : On gère les deux clés Cohere
    cohere_client_trial = cohere.Client(get_secret("COHERE_TRIAL_API_KEY"))
    cohere_prod_key = get_secret("COHERE_PROD_API_KEY")
    
    return openai_client, supabase, pinecone_index, cohere_client_trial, cohere_prod_key

openai_client, supabase, pinecone_index, cohere_client_trial, cohere_prod_key = init_clients()

# ==========================================
# 2.5 FONCTIONS DE MONITORING & COÛTS
# ==========================================
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 0.25},
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 2.50}
}

def calculate_cost(model_name: str, input_tok: int, cached_tok: int, output_tok: int) -> float:
    if model_name not in PRICING:
        return 0.0
    p = PRICING[model_name]
    uncached_input = max(0, input_tok - cached_tok)
    return (uncached_input * p["input"] + cached_tok * p["cached_input"] + output_tok * p["output"]) / 1_000_000

def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))

# ==========================================
# 3. STRUCTURE DU ROUTER
# ==========================================
class QueryAnalysis(BaseModel):
    optimized_query: str = Field(
        description="Réécris la requête en une question clinique claire pour un moteur de recherche. RÈGLES ABSOLUES : 1. Retire les formules de politesse. 2. N'invente AUCUN diagnostic. 3. NE MENTIONNE JAMAIS LE NUMÉRO DE L'ITEM."
    )   
    category: str = Field(description="Catégorie de la question : 'specific_question', 'full_item_review', 'full_disease_review', ou 'non_medical_or_smalltalk' (pour les salutations, remerciements ou questions hors sujet médical).")
    item_filter: Optional[str] = Field(description="Si l'étudiant mentionne un numéro d'item, extraire la valeur exacte format 'Item XXX'. Sinon, null.")
    hypothetical_answer: str = Field(description="Rédige une réponse hypothétique TRÈS COURTE (2-3 phrases) à la question.")

# ==========================================
# 4. MOTEUR RAG (Les vraies fonctions)
# ==========================================
def rag_pipeline(user_query, status_container):
    metrics = {"latencies": {}, "tokens": {"router": {"input": 0, "cached": 0, "output": 0}, "generator": {"input": 0, "cached": 0, "output": 0}}}
    t_global_start = time.time()
    
    ROUTER_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    RERANK_MODEL = "rerank-v4.0-pro"
    
    # --- ETAPE 1 : ROUTER ---
    t0 = time.time()
    if st.session_state.use_router:
        status_container.write("🔀 Analyse de la requête...")
        router_prompt = f"Tu es l'agent d'analyse de requêtes de Stéthopote.\n\nRequête : {user_query}"
        router_response = openai_client.responses.parse(
            model=ROUTER_MODEL, input=router_prompt, text_format=QueryAnalysis, temperature=0.0
        )
        analysis = router_response.output_parsed
        
        # Tracking Tokens Routeur
        usage = getattr(router_response, 'usage', None)
        if usage:
            metrics["tokens"]["router"]["input"] = getattr(usage, 'input_tokens', 0)
            metrics["tokens"]["router"]["output"] = getattr(usage, 'output_tokens', 0)
            details = getattr(usage, 'input_tokens_details', None)
            if details:
                metrics["tokens"]["router"]["cached"] = getattr(details, 'cached_tokens', 0)
        if metrics["tokens"]["router"]["input"] == 0:
            metrics["tokens"]["router"]["input"] = count_tokens(router_prompt, ROUTER_MODEL)
            metrics["tokens"]["router"]["output"] = count_tokens(analysis.model_dump_json(), ROUTER_MODEL)
    else:
        analysis = QueryAnalysis(optimized_query=user_query, category="unknown", item_filter=None, hypothetical_answer="")
    metrics["latencies"]["router"] = time.time() - t0
    metrics["optimized_query"] = analysis.optimized_query
    metrics["item_filter"] = analysis.item_filter

    # 🛑 LE COURT-CIRCUIT (SMALL TALK)
    if analysis.category == "non_medical_or_smalltalk":
        status_container.write("👋 Discussion classique détectée (pas de recherche RAG)...")
        metrics["latencies"]["vectorization"] = 0
        metrics["latencies"]["pinecone"] = 0
        metrics["latencies"]["supabase"] = 0
        metrics["latencies"]["cohere"] = 0
        metrics["latencies"]["total_retrieval"] = time.time() - t_global_start
        return "", [], metrics

    # --- ETAPE 2 : VECTORISATION ---
    t0 = time.time()
    status_container.write("🧬 Vectorisation de la requête...")
    text_for_dense = analysis.hypothetical_answer if st.session_state.use_hyde and analysis.hypothetical_answer else analysis.optimized_query
    query_dense = openai_client.embeddings.create(input=text_for_dense, model=EMBEDDING_MODEL).data[0].embedding
    filter_kwargs = {"filter": {"items_r2c": {"$in": [analysis.item_filter]}}} if analysis.item_filter else {}
    metrics["latencies"]["vectorization"] = time.time() - t0

    # --- ETAPE 3 : RECHERCHE PINECONE ---
    t0 = time.time()
    status_container.write(f"🔍 Recherche vectorielle ({st.session_state.top_k_children} vecteurs)...")
    res_dense = pinecone_index.query(vector=query_dense, top_k=st.session_state.top_k_children, include_metadata=True, **filter_kwargs)
    candidats_fusion = [{"child_id": m.id, "parent_id": m.metadata.get("parent_id"), "texte": m.metadata.get("text", "")} for m in res_dense.matches]
    pids_fusion = list(set([c["parent_id"] for c in candidats_fusion]))
    metrics["latencies"]["pinecone"] = time.time() - t0

    # --- ETAPE 4 : SUPABASE & RERANKING ---
    t0 = time.time()
    status_container.write("💾 Récupération des blocs parents...")
    parents_map = {}
    if pids_fusion:
        supa_res = supabase.table("parent_chunks").select("id, titre_h1, titre_h2, content, pages, college, ordre_lecture").in_("id", pids_fusion).execute()
        parents_map = {row["id"]: row for row in supa_res.data}
    metrics["latencies"]["supabase"] = time.time() - t0

    t0 = time.time()
    status_container.write(f"⚖️ Reranking ({st.session_state.top_n_parents} parents cibles)...")
    docs_to_rerank = [e["texte"] for e in candidats_fusion]
    winning_parents = {}
    
    if docs_to_rerank:
        # 👈 NOUVEAU : Le système de double clé (Fallback)
        try:
            rerank_res = cohere_client_trial.rerank(model=RERANK_MODEL, query=analysis.optimized_query, documents=docs_to_rerank, top_n=len(docs_to_rerank))
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                status_container.write("⚠️ Limite gratuite atteinte. Bascule sur l'API Pro...")
                co_prod = cohere.Client(cohere_prod_key)
                rerank_res = co_prod.rerank(model=RERANK_MODEL, query=analysis.optimized_query, documents=docs_to_rerank, top_n=len(docs_to_rerank))
            else:
                raise e # Si c'est une autre erreur (ex: problème réseau), on laisse crasher

        # Déduplication classique
        for hit in rerank_res.results:
            enfant = candidats_fusion[hit.index]
            pid = enfant["parent_id"]
            if pid not in winning_parents:
                if len(winning_parents) >= st.session_state.top_n_parents:
                    continue
                winning_parents[pid] = []
            winning_parents[pid].append(enfant["texte"])
            
    metrics["latencies"]["cohere"] = time.time() - t0

    # --- ETAPE 5 : FORMATAGE ---
    parents_to_format = []
    for pid, child_texts in winning_parents.items():
        if pid in parents_map:
            p_data = parents_map[pid]
            p_data['child_texts'] = child_texts
            p_data['college'] = p_data.get('college') or "Z_Inconnu"
            p_data['ordre_lecture'] = p_data.get('ordre_lecture') or 0
            parents_to_format.append(p_data)
            
    parents_to_format.sort(key=lambda x: (x['college'], x['ordre_lecture']))
    
    contextes = []
    for p_data in parents_to_format:
        pid = p_data["id"]
        titre = f"{p_data.get('titre_h1', '')} > {p_data.get('titre_h2', '')}"
        content = p_data.get("content", "")
        for c_text in p_data["child_texts"]:
            clean_c = re.sub(r"^\[Contexte\s*:[^\]]+\]\n", "", c_text.strip()).strip()
            if clean_c and clean_c in content:
                content = content.replace(clean_c, f"<passage_cle>\n{clean_c}\n</passage_cle>")
            else:
                content = f"<passage_cle>\n{clean_c}\n</passage_cle>\n\n" + content
        pages_str = ", ".join(p_data.get("pages", [])) if p_data.get("pages") else "N/A"
        contextes.append(f"--- SOURCE : {titre} (Pages: {pages_str}) | ID: {pid} ---\n{content}\n")

    metrics["latencies"]["total_retrieval"] = time.time() - t_global_start
    return "\n".join(contextes), parents_to_format, metrics

def llm_stream(user_query, contexte_final, model_choice, metrics):
    LLM_MODEL = "gpt-5.6-luna" if model_choice == "🌙 Luna" else "gpt-5.6-terra"
    t0 = time.time()
    
    # Choix du prompt selon qu'il y a eu un court-circuit ou non
    if not contexte_final:
        system_prompt = (
            "Tu es Stéthopote, un tuteur médical amical pour les étudiants en médecine.\n"
            "L'étudiant te parle de manière informelle (salutations, blague, hors sujet).\n"
            "Réponds-lui avec sympathie et humour en quelques phrases, et rappelle-lui que tu es là pour l'aider à réviser ses EDN.\n\n"
            f"MESSAGE DE L'ÉTUDIANT : \n{user_query}"
        )
    else:
        system_prompt = (
            "Tu es Stéthopote, un tuteur médical expert et bienveillant, conçu pour aider les étudiants en médecine français à préparer les EDN.\n\n"
            "RÈGLES DE LECTURE DU CONTEXTE :\n"
            "1. Les documents sont classés par ordre de lecture logique.\n"
            "2. Porte une attention toute particulière aux textes encadrés par les balises <passage_cle>...</passage_cle>.\n\n"
            "RÈGLES DE RÉDACTION :\n"
            "- FIABILITÉ ABSOLUE : N'invente jamais rien.\n"
            "- STRUCTURE : Utilise le Markdown (listes à puces, gras).\n"
            "- SOURÇAGE IN-LINE : Insère des références courtes entre crochets de manière parcimonieuse (ex: [VIII. > A.]).\n"
            "- N'ajoute PAS de bibliographie à la fin de ton texte.\n\n"
            f"CONTEXTE OFFICIEL :\n{contexte_final}\n\n"
            f"QUESTION DE L'ÉTUDIANT : \n{user_query}"
        )

    response_stream = openai_client.responses.create(
        model=LLM_MODEL, input=system_prompt, reasoning={"effort": "none"}, stream=True
    )

    reponse_complete = []
    for event in response_stream:
        if event.type == "response.output_text.delta":
            reponse_complete.append(event.delta)
            yield event.delta
            
        elif event.type == "response.completed":
            usage = None
            if hasattr(event, 'response') and getattr(event.response, 'usage', None):
                usage = event.response.usage
            elif getattr(event, 'usage', None):
                usage = event.usage
                
            if usage:
                metrics["tokens"]["generator"]["input"] = getattr(usage, 'input_tokens', 0)
                metrics["tokens"]["generator"]["output"] = getattr(usage, 'output_tokens', 0)
                details = getattr(usage, 'input_tokens_details', None)
                if details:
                    metrics["tokens"]["generator"]["cached"] = getattr(details, 'cached_tokens', 0)

    # Fallback si l'API ne renvoie pas l'usage
    if metrics["tokens"]["generator"]["input"] == 0:
        metrics["tokens"]["generator"]["input"] = count_tokens(system_prompt, LLM_MODEL)
        metrics["tokens"]["generator"]["output"] = count_tokens("".join(reponse_complete), LLM_MODEL)
        
    metrics["latencies"]["generation"] = time.time() - t0

# ==========================================
# 5. INTERFACE UTILISATEUR (Sidebar & Main)
# ==========================================
with st.sidebar:
    st.title("⚙️ Stéthopote")
    st.caption("Ton binôme de révision médical.")
    if st.button("🔄 Effacer l'historique", type="primary", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Pose-moi une question médicale."}]
        # On supprime la question en mémoire pour forcer un nouveau tirage aléatoire
        if "placeholder_q" in st.session_state:
            del st.session_state["placeholder_q"]
        st.rerun()

    st.divider()
    with st.expander("🛠️ Options RAG"):
        st.session_state.top_k_children = st.slider("🔍 Enfants", 10, 100, 50, 10)
        st.session_state.top_n_parents = st.slider("⚖️ Parents (Top N)", 1, 10, 5)
        st.session_state.use_router = st.toggle("🔀 Routeur LLM", value=True)
        st.session_state.use_hyde = st.toggle("✨ HyDE", value=True)

col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
with col1:
    st.title("🩺 Stéthopote", anchor=False)
with col2:
    st.session_state.model_choice = st.selectbox("Modèle", ["🌙 Luna", "🌍 Terra"], label_visibility="collapsed")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Pose-moi une question médicale."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources médicales consultées"):
                for src in msg["sources"]:
                    college = src.get("college", "Inconnu")
                    titre = f"{src.get('titre_h1', '')} > {src.get('titre_h2', '')}".strip(" >")
                    pages_str = ", ".join(src.get("pages", [])) if src.get("pages") else "N/A"
                    st.markdown(f'<div class="source-box"><strong>📖 {college}</strong> (Page(s) {pages_str})<br><em>{titre}</em></div>', unsafe_allow_html=True)

QUESTIONS_EXEMPLES = [
    "Quels sont les signes à l'ECG d'une hyperkaliémie menaçante ?",
    "Prise en charge thérapeutique immédiate d'une crise d'asthme sévère ?",
    "Comment faire le diagnostic d'une endocardite infectieuse (Critères de Duke) ?",
    "Quelles sont les indications de l'oxygénothérapie longue durée (OLD) dans la BPCO ?",
    "Quelles sont les causes d'insuffisance rénale aiguë fonctionnelle ?",
    "Quels sont les drapeaux rouges (red flags) devant une céphalée aiguë ?",
    "Quel est le bilan de première intention devant une anémie microcytaire ?",
    "Quelles sont les complications microvasculaires du diabète de type 2 ?",
    "Quels sont les critères de gravité d'une pancréatite aiguë ?",
    "Quel est le traitement probabiliste d'une méningite bactérienne communautaire ?",
    "Quelles sont les contre-indications absolues à la thrombolyse dans l'AVC ischémique ?",
    "Quels sont les signes cliniques et biologiques d'une hypothyroïdie fruste ?",
    "Comment évaluer le risque cardiovasculaire global (SCORE) ?",
    "Quel est l'antibiothérapie d'une pyélonéphrite aiguë simple chez la femme ?",
    "Quels sont les signes cliniques d'une occlusion intestinale aiguë ?",
    "Score de Wells et démarche diagnostique devant une suspicion d'embolie pulmonaire ?",
    "Quels sont les traitements antalgiques recommandés dans la colique néphrétique ?",
    "Quels sont les signes cliniques de l'insuffisance cardiaque gauche ?",
    "Prise en charge d'une fibrillation atriale (FA) mal tolérée sur le plan hémodynamique ?",
    "Diagnostic et prise en charge d'un état de mal épileptique de l'adulte ?",
    "Quels sont les signes cliniques d'hypertension intracrânienne (HTIC) ?",
    "Règle ABCDE pour le dépistage clinique du mélanome ?",
    "Quels sont les signes de gravité d'une pneumonie aiguë communautaire (score CRB-65) ?"
]

# --- GESTION DU TEXTE DE LA BARRE DE RECHERCHE ---
if len(st.session_state.messages) <= 1:
    # C'est le début de la conversation : on fige une question aléatoire
    if "placeholder_q" not in st.session_state:
        st.session_state.placeholder_q = random.choice(QUESTIONS_EXEMPLES)
    texte_barre = f"Ex: {st.session_state.placeholder_q}"
else:
    # La conversation a commencé : texte classique
    texte_barre = "Demander à Stéthopote..."

# --- LA BARRE DE CHAT ---
if prompt := st.chat_input(texte_barre):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🧠 Analyse de la littérature médicale...", expanded=True) as status:
            contexte_final, parents_to_format, metrics = rag_pipeline(prompt, status)
            status.update(label="Réponse prête !", state="complete", expanded=False)

        # On stream la réponse en lui passant metrics par référence
        full_response = st.write_stream(llm_stream(prompt, contexte_final, st.session_state.model_choice, metrics))

        # Affichage des sources dans l'UI (uniquement s'il y a eu une recherche RAG)
        if parents_to_format:
            sources_uniques = []
            with st.expander("📚 Sources médicales consultées", expanded=True):
                for src in parents_to_format:
                    college = src.get("college", "Inconnu")
                    titre = f"{src.get('titre_h1', '')} > {src.get('titre_h2', '')}".strip(" >")
                    pages_str = ", ".join(src.get("pages", [])) if src.get("pages") else "N/A"
                    source_html = f'<div class="source-box"><strong>📖 {college}</strong> (Page(s) {pages_str})<br><em>{titre}</em></div>'
                    if source_html not in sources_uniques:
                        sources_uniques.append(source_html)
                        st.markdown(source_html, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": parents_to_format})

        # ==========================================
        # LOGS SERVEUR (Console Streamlit)
        # ==========================================
        LLM_MODEL = "gpt-5.6-luna" if st.session_state.model_choice == "🌙 Luna" else "gpt-5.6-terra"
        c_router = calculate_cost("gpt-4o-mini", metrics["tokens"]["router"]["input"], metrics["tokens"]["router"]["cached"], metrics["tokens"]["router"]["output"])
        c_gen = calculate_cost(LLM_MODEL, metrics["tokens"]["generator"]["input"], metrics["tokens"]["generator"]["cached"], metrics["tokens"]["generator"]["output"])
        
        print("\n" + "="*50)
        print("🚀 NOUVELLE REQUÊTE :", prompt)
        print("⏱️ LATENCES:")
        print(f"  - Routeur  : {metrics['latencies'].get('router', 0):.2f}s")
        print(f"  - Cohere   : {metrics['latencies'].get('cohere', 0):.2f}s")
        print(f"  - Retrieval: {metrics['latencies'].get('total_retrieval', 0):.2f}s")
        print(f"  - LLM Gen  : {metrics['latencies'].get('generation', 0):.2f}s")
        print("💰 COÛTS:")
        print(f"  - Routeur ({c_router:.6f}$) | In: {metrics['tokens']['router']['input']} (Cache: {metrics['tokens']['router']['cached']}) | Out: {metrics['tokens']['router']['output']}")
        print(f"  - LLM Gen ({c_gen:.6f}$) | In: {metrics['tokens']['generator']['input']} (Cache: {metrics['tokens']['generator']['cached']}) | Out: {metrics['tokens']['generator']['output']}")
        print(f"💸 TOTAL : ${c_router + c_gen:.6f}")
        print("="*50 + "\n")

        # ==========================================
        # 💾 ENREGISTREMENT ANALYTIQUE DANS SUPABASE
        # ==========================================
        try:
            log_data = {
                "user_query": prompt,
                "optimized_query": metrics.get("optimized_query", prompt),
                "item_filter": metrics.get("item_filter", None),
                "model_used": LLM_MODEL,
                
                # Latences
                "latency_router": round(metrics['latencies'].get('router', 0), 3),
                "latency_cohere": round(metrics['latencies'].get('cohere', 0), 3),
                "latency_retrieval": round(metrics['latencies'].get('total_retrieval', 0), 3),
                "latency_generation": round(metrics['latencies'].get('generation', 0), 3),
                "latency_total": round(metrics['latencies'].get('total_retrieval', 0) + metrics['latencies'].get('generation', 0), 3),
                
                # Tokens Routeur
                "router_input_tokens": metrics["tokens"]["router"]["input"],
                "router_cached_tokens": metrics["tokens"]["router"]["cached"],
                "router_output_tokens": metrics["tokens"]["router"]["output"],
                
                # Tokens Générateur
                "gen_input_tokens": metrics["tokens"]["generator"]["input"],
                "gen_cached_tokens": metrics["tokens"]["generator"]["cached"],
                "gen_output_tokens": metrics["tokens"]["generator"]["output"],
                
                # Coûts
                "cost_router": round(c_router, 6),
                "cost_gen": round(c_gen, 6),
                "cost_total": round(c_router + c_gen, 6),
                
                "nb_parents_utilises": len(parents_to_format)
            }
            # L'appel à Supabase
            res = supabase.table("rag_logs").insert(log_data).execute()
            print(f"✅ Log sauvegardé dans Supabase (ID: {res.data[0]['id']})")
            
        except Exception as e:
            print(f"⚠️ Erreur lors de l'enregistrement du log Supabase : {e}")