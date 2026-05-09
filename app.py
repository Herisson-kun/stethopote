import streamlit as st
import os
import random
import json
from pathlib import Path
from dotenv import load_dotenv

# --- Imports LangChain Officiels & Robustes ---
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever # Import standard
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURATION INITIALE & STYLE
# ==========================================
load_dotenv()
st.set_page_config(page_title="Stéthopote 🩺", page_icon="🩺", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stChatMessage { font-size: 1.1rem !important; }
    .stMarkdown { line-height: 1.6; }
    div[data-baseweb="input"] > div:focus-within {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 0 1px #4A90E2 !important;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2C3E50;
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
# 2. INITIALISATION DU CERVEAU (Cloud + Local)
# ==========================================
@st.cache_resource
def init_stethopote():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docstore_path = os.path.join(current_dir, "store_with_page", "docstore.jsonl")
    
    # 1. Pinecone
    vectorstore = PineconeVectorStore(
        index_name="stethopote",
        embedding=embeddings
    )
    
    # 2. Docstore
    store = InMemoryStore()
    try:
        if os.path.exists(docstore_path):
            with open(docstore_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        doc = Document(
                            page_content=data["content"],
                            metadata={
                                "document_name": data["document_name"],
                                "page_start": data["page_start"],
                                "page_end": data["page_end"]
                            }
                        )
                        store.mset([(data["id"], doc)])
        else:
            st.error(f"Fichier docstore.jsonl introuvable à : {docstore_path}")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du docstore : {e}")
    
    # 3. Splitters (Requis par le Retriever)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    
    # 4. Retriever
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

# ==========================================
# 3. INTERFACE & LOGIQUE
# ==========================================

with st.sidebar:
    st.title("⚙️ Paramètres")
    st.info("Assistant de révision basé strictement sur tes cours.")
    st.divider()
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🩺 Stéthopote", anchor=False)
st.caption("Ton binôme de révision médical")

try:
    retriever = init_stethopote()
except Exception as e:
    st.error(f"Erreur de base : {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

placeholder_q = random.choice(QUESTIONS_EXEMPLES)

if prompt := st.chat_input(f"Ex: {placeholder_q}"):
    
    # Affichage utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse Assistant
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les cours... 📚"):
            try:
                # ÉTAPE A : Retrieval
                relevant_docs = retriever.invoke(prompt)
                
                if not relevant_docs:
                    st.warning("Désolé, je ne trouve pas d'info à ce sujet dans tes collèges.")
                else:
                    context = "\n\n---\n\n".join([d.page_content for d in relevant_docs])
                    
                    # ÉTAPE B : LLM & Prompt
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "Tu es Stéthopote, un assistant médical. Réponds UNIQUEMENT en utilisant le contexte fourni. Si l'info manque, dis-le. Structure avec des puces et du gras."),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "Contexte :\n{context}\n\nQuestion : {query}")
                    ])
                    
                    chain = prompt_template | llm
                    
                    # ÉTAPE C : Historique formaté
                    history = []
                    for m in st.session_state.messages[:-1]:
                        role = "human" if m["role"] == "user" else "ai"
                        history.append((role, m["content"]))
                    
                    # ÉTAPE D : Streaming
                    full_response = st.write_stream(
                        chunk.content for chunk in chain.stream({
                            "context": context,
                            "query": prompt,
                            "chat_history": history
                        })
                    )
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                    # ÉTAPE E : Sources
                    with st.expander("📚 Sources consultées"):
                        for i, d in enumerate(relevant_docs, 1):
                            name = d.metadata.get('document_name', 'Inconnu')
                            p_s = d.metadata.get('page_start', '?')
                            p_e = d.metadata.get('page_end', '?')
                            p_info = f"Page {p_s}" if p_s == p_e else f"Pages {p_s} à {p_e}"
                            st.markdown(f"**{i}. {name}** — *{p_info}*")
                            
            except Exception as e:
                st.error(f"Erreur technique : {e}")