import streamlit as st
import time
import random

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

QUESTIONS_EXEMPLES = [
    "Signes de la pneumonie lobaire aiguë ?", "Prise en charge de l'infarctus (SCA ST+) ?",
    "Étiologies d'une hypercalcémie ?", "Critères de Duke (endocardite) ?",
    "Traitement de la crise d'asthme sévère ?", "Complications microvasculaires du diabète ?",
    "Signes ECG de l'embolie pulmonaire ?", "Bilan d'une anémie macrocytaire ?",
    "Critères de gravité de la pancréatite ?", "Traitement de la méningite communautaire ?",
    "Indications de l'oxygénothérapie (BPCO) ?", "Effets indésirables des corticoïdes ?",
    "Diagnostic de la sclérose en plaques ?", "Prise en charge d'un choc septique ?",
    "Contre-indications absolues des AINS ?", "Signes biologiques de l'hypothyroïdie ?",
    "Évaluation du risque CV (SCORE) ?", "Traitement de la pyélonéphrite simple ?",
    "Causes d'insuffisance rénale fonctionnelle ?", "Signes de l'occlusion intestinale aiguë ?",
    "Diagnostic d'une appendicite aiguë ?", "Médicaments de la colique néphrétique ?",
    "Signes de l'insuffisance cardiaque gauche ?", "Traitement de la FA mal tolérée ?",
    "Signes ECG d'une hyperkaliémie ?", "Diagnostic de la tuberculose pulmonaire ?",
    "Prise en charge de l'état de mal épileptique ?", "Signes d'hypertension intracrânienne ?",
    "Dépistage du mélanome (règle ABCDE) ?", "Red flags devant une céphalée aiguë ?"
]

# ==========================================
# 2. FONCTIONS MOCKS (Labo RAG)
# ==========================================
def step_1_pinecone_search(query, top_k, use_filter):
    time.sleep(0.5)
    return [f"child_{i}" for i in range(1, top_k + 1)]

def step_2_cohere_rerank(query, child_docs, top_n):
    time.sleep(0.8)
    return child_docs[:top_n]

def step_3_supabase_fetch(top_child_ids):
    time.sleep(0.5)
    return [{
        "college": "Cardiologie", "pages": ["cardiologie_3e_154"],
        "titre": "Item 230 > Syndrome Coronarien Aigu",
        "texte": "Le dosage de la troponine ultrasensible est inutile dans la prise en charge des SCA ST+."
    }]

def step_4_llm_stream(query, parent_docs, model_choice):
    modele_name = "Terra 🌍" if model_choice == "🌍 Terra" else "Luna 🌙"
    reponse = f"*(Généré par {modele_name})* D'après les recommandations, la prise en charge d'un SCA ST+ est une urgence absolue. Il ne faut pas attendre la troponine."
    for mot in reponse.split(" "):
        yield mot + " "
        time.sleep(0.04)

# ==========================================
# 3. PANNEAU LATÉRAL (Labo RAG)
# ==========================================
with st.sidebar:
    st.title("⚙️ Stéthopote")
    st.caption("Ton binôme de révision médical, sourcé à 100% sur les référentiels nationaux.")
    
    if st.button("🔄 Effacer l'historique en cours", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    with st.expander("🛠️ Options Avancées (Labo RAG)"):
        st.session_state.use_reranker = st.toggle("Activer Reranker (Cohere)", value=True)
        st.session_state.top_k_children = st.slider("🔍 Enfants récupérés", min_value=10, max_value=200, value=50, step=10)
        
        if st.session_state.use_reranker:
            st.session_state.top_n_parents = st.slider("⚖️ Parents (Top N Rerank)", min_value=1, max_value=10, value=3)
        else:
            st.session_state.top_n_parents = st.slider("⚖️ Parents (Sans Rerank)", min_value=1, max_value=10, value=5)
            
        st.session_state.use_router = st.toggle("🔀 Routeur LLM d'intentions", value=False)
        st.session_state.use_filter = st.toggle("🏷️ Filtrage par Items R2C", value=True)
        st.session_state.use_reformulation = st.toggle("🔄 Reformulation de requête", value=False)

# ==========================================
# 4. ZONE PRINCIPALE (Modèle & Chat)
# ==========================================
col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
with col1:
    st.title("🩺 Stéthopote", anchor=False)
with col2:
    st.session_state.model_choice = st.selectbox("Modèle", ["🌍 Terra", "🌙 Luna"], label_visibility="collapsed")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Pose-moi une question médicale."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Sources médicales consultées"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>📖 {src['college']}</strong> (Pages {", ".join([p.split('_')[-1] for p in src['pages']])})<br>
                        <em>{src['titre']}</em>
                    </div>
                    """, unsafe_allow_html=True)

placeholder_q = random.choice(QUESTIONS_EXEMPLES)

if prompt := st.chat_input(f"Ex: {placeholder_q}"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🧠 Analyse de la littérature médicale...", expanded=True) as status:
            if st.session_state.use_reformulation:
                st.write("🔄 Reformulation de la question...")
                time.sleep(0.3)
            if st.session_state.use_router:
                st.write("🔀 Routage de l'intention...")
                time.sleep(0.3)

            st.write(f"🔍 Recherche (Récupération de {st.session_state.top_k_children} vecteurs)...")
            enfants = step_1_pinecone_search(prompt, st.session_state.top_k_children, st.session_state.use_filter)
            
            if st.session_state.use_reranker:
                st.write(f"⚖️ Reranking (Sélection des {st.session_state.top_n_parents} meilleurs)...")
                top_enfants = step_2_cohere_rerank(prompt, enfants, st.session_state.top_n_parents)
            else:
                top_enfants = enfants[:st.session_state.top_n_parents]
            
            st.write("📚 Récupération des fragments complets (Supabase)...")
            parents = step_3_supabase_fetch(top_enfants)
            status.update(label="Réponse prête !", state="complete", expanded=False)

        full_response = st.write_stream(step_4_llm_stream(prompt, parents, st.session_state.model_choice))

        with st.expander("📚 Sources médicales consultées"):
            for src in parents:
                st.markdown(f"""
                <div class="source-box">
                    <strong>📖 {src['college']}</strong> (Pages {", ".join([p.split('_')[-1] for p in src['pages']])})<br>
                    <em>{src['titre']}</em>
                </div>
                """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant", "content": full_response, "sources": parents
        })