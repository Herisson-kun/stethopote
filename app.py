import streamlit as st
import os
import random # <--- NOUVEAU
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
import json

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage.file_system import LocalFileStore
from langchain_classic.storage.encoder_backed import EncoderBackedStore
import pickle
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURATION INITIALE & STYLE (Sophistiqué)
# ==========================================
load_dotenv()
st.set_page_config(page_title="Stéthopote 🩺", page_icon="🩺", layout="centered", initial_sidebar_state="expanded")

# --- INJECTION CSS ---
st.markdown("""
    <style>
    /* Taille de la police du chat pour le confort de lecture */
    .stChatMessage { font-size: 1.1rem !important; }
    .stMarkdown { line-height: 1.6; }
    
    /* 1. Remplacer le rouge agressif par un bleu "médical" (#4A90E2) lors du clic sur la barre de texte */
    div[data-baseweb="input"] > div:focus-within {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 0 1px #4A90E2 !important;
    }
    
    /* Optionnel : styliser un peu les expanders pour qu'ils fassent plus "Pro" */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2C3E50;
    }
    </style>
""", unsafe_allow_html=True)

# --- LISTE DES 30 QUESTIONS ALÉATOIRES ---
QUESTIONS_EXEMPLES = [
    "Quels sont les signes de la pneumonie franche lobaire aiguë ?",
    "Quelle est la prise en charge d'un infarctus du myocarde (SCA ST+) ?",
    "Quelles sont les étiologies principales d'une hypercalcémie ?",
    "Comment diagnostiquer une endocardite infectieuse (critères de Duke) ?",
    "Quel est le traitement de première intention d'une crise d'asthme aiguë sévère ?",
    "Quelles sont les complications microvasculaires du diabète de type 2 ?",
    "Quels sont les signes cliniques et ECG d'une embolie pulmonaire ?",
    "Quelle est la démarche diagnostique devant une anémie macrocytaire ?",
    "Quels sont les critères de gravité cliniques d'une pancréatite aiguë ?",
    "Comment prendre en charge une méningite bactérienne communautaire ?",
    "Quelles sont les indications de l'oxygénothérapie longue durée dans la BPCO ?",
    "Quels sont les effets indésirables des corticoïdes au long cours ?",
    "Comment diagnostiquer une sclérose en plaques ?",
    "Quelle est la prise en charge hémodynamique d'un choc septique ?",
    "Quelles sont les contre-indications absolues des AINS ?",
    "Quels sont les signes cliniques et biologiques d'une hypothyroïdie ?",
    "Comment évaluer le risque cardiovasculaire global (SCORE) ?",
    "Quel est le traitement d'une pyélonéphrite aiguë simple chez la femme ?",
    "Quelles sont les causes d'une insuffisance rénale aiguë fonctionnelle ?",
    "Quels sont les signes cliniques d'une occlusion intestinale aiguë ?",
    "Comment diagnostiquer une appendicite aiguë ?",
    "Quelle est la prise en charge médicamenteuse d'une colique néphrétique ?",
    "Quels sont les signes d'une insuffisance cardiaque gauche ?",
    "Quel est le traitement de première ligne d'une fibrillation atriale mal tolérée ?",
    "Quelles sont les causes et signes ECG d'une hyperkaliémie ?",
    "Comment diagnostiquer une tuberculose pulmonaire maladie ?",
    "Quelle est la prise en charge d'une crise convulsive généralisée (état de mal) ?",
    "Quels sont les signes cliniques d'une hypertension intracrânienne ?",
    "Comment diagnostiquer un mélanome (règle ABCDE) ?",
    "Quels sont les signes d'alerte (red flags) devant une céphalée aiguë ?"
]

# ==========================================
# 2. INITIALISATION DU CERVEAU 
# ==========================================
@st.cache_resource
def init_stethopote():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Les nouveaux chemins !
    chroma_dir = "store_with_page/chroma_db"
    docstore_path = "store_with_page/docstore.jsonl"
    
    # 1. Chargement de la base vectorielle ChromaDB
    vectorstore = Chroma(
        collection_name="stethopote_children",
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )
    
    # 2. Chargement du DocStore depuis le fichier JSONL
    store = InMemoryStore()
    try:
        with open(docstore_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # On recrée l'objet Document que LangChain attend
                    doc = Document(
                        page_content=data["content"],
                        metadata={
                            "document_name": data["document_name"],
                            "page_start": data["page_start"],
                            "page_end": data["page_end"]
                        }
                    )
                    store.mset([(data["id"], doc)])
    except FileNotFoundError:
        st.error(f"Fichier introuvable : {docstore_path}. As-tu bien lancé l'ingestion ?")
    
    # 3. Les splitters "décoratifs" pour la validation LangChain
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o", chunk_size=1000, chunk_overlap=100
    )
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o", chunk_size=200, chunk_overlap=20
    )
    
    # 4. Le Retriever final
    base_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    return base_retriever

# ==========================================
# 3. INTERFACE UTILISATEUR & LOGIQUE RAG
# ==========================================

# --- LA BARRE LATÉRALE (Le côté sophistiqué) ---
with st.sidebar:
    st.title("⚙️ Paramètres")
    st.info("Stéthopote est ton assistant de révision. Ses réponses sont générées strictement à partir de tes collèges de médecine sourcés.")
    
    st.divider()
    
    # Un bouton propre pour vider la mémoire de la conversation
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- TITRE PRINCIPAL ---
# On ajoute anchor=False pour désactiver le petit logo "lien" agaçant !
st.title("🩺 Stéthopote", anchor=False)
st.caption("Ton binôme de révision médical")

# --- CHARGEMENT DE LA BASE ---
try:
    retriever = init_stethopote()
except Exception as e:
    st.error(f"Erreur fatale lors du chargement de la mémoire : {e}")
    st.stop()

# --- GESTION DE L'HISTORIQUE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHOIX ALÉATOIRE DE LA QUESTION DE PLACEHOLDER ---
question_aleatoire = random.choice(QUESTIONS_EXEMPLES)

# --- ZONE DE CHAT ---
if prompt := st.chat_input(f"Pose ta question (ex: {question_aleatoire})"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Je fouille dans tes collèges de médecine... 📚"):
            try:
                # ÉTAPE A : Recherche
                relevant_docs = retriever.invoke(prompt)
                
                if not relevant_docs:
                    st.warning("Je n'ai rien trouvé d'assez précis dans tes collèges pour cette question.")
                else:
                    # ÉTAPE B : Contexte
                    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # ÉTAPE C : LLM
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    
                    system_prompt = """Tu es Stéthopote, un assistant médical bienveillant conçu pour aider une étudiante en médecine.
                    
Pour répondre aux questions médicales, tu dois IMPÉRATIVEMENT utiliser le CONTEXTE fourni. Si l'information n'y est pas, dis-le.
Tu as aussi accès à l'HISTORIQUE de la conversation pour le contexte.

Structure ta réponse avec des listes à puces et du gras pour les mots-clés."""

                    full_prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("user", "Contexte des cours :\n{context}\n\nQuestion de l'étudiante : {query}")
                    ])
                    
                    chain = full_prompt | llm
                    
                    historique_langchain = []
                    for msg in st.session_state.messages:
                        historique_langchain.append((msg["role"], msg["content"]))
                    
                    stream = chain.stream({
                        "context": context, 
                        "query": prompt,
                        "chat_history": historique_langchain
                    })
                    
                    full_response = st.write_stream(chunk.content for chunk in stream)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                    # ÉTAPE E : NOUVEL AFFICHAGE DES SOURCES (Nom du collège et Pages)
                    with st.expander("📚 Voir les sources consultées"):
                        for i, doc in enumerate(relevant_docs, 1):
                            # On récupère les métadonnées injectées lors de l'ingestion
                            doc_name = doc.metadata.get('document_name', 'Collège inconnu')
                            page_start = doc.metadata.get('page_start', '?')
                            page_end = doc.metadata.get('page_end', '?')
                            
                            # Formatage propre : si start et end sont identiques, on n'affiche qu'une page
                            if page_start == page_end:
                                pages_info = f"Page {page_start}"
                            else:
                                pages_info = f"Pages {page_start} à {page_end}"
                                
                            # On affiche uniquement le collège et la page
                            st.markdown(f"**{i}. {doc_name}** — *{pages_info}*")
                            
            except Exception as e:
                st.error(f"Oups, j'ai eu un petit souci technique : {e}")
