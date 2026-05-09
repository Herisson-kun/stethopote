import streamlit as st
import os
import random
import json
from pathlib import Path
from dotenv import load_dotenv

# --- Imports LangChain Modernes ---
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
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

# Tes 30 questions courtes (Parfait pour éviter le scroll !)
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
    
    # Chemin absolu blindé pour Streamlit
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docstore_path = os.path.join(current_dir, "store_with_page", "docstore.jsonl")
    
    # 1. Le Cloud : Pinecone
    vectorstore = PineconeVectorStore(
        index_name="stethopote",
        embedding=embeddings
    )
    
    # 2. Le Local : Docstore en JSONL
    store = InMemoryStore()
    try:
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
    except FileNotFoundError:
        st.error(f"⚠️ Docstore introuvable. Assure-toi que le fichier existe ici : {docstore_path}")
    
    # 3. Les splitters (Placeholders obligatoires)
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=20)
    
    # 4. Le Retriever unifié
    base_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    return base_retriever

# ==========================================
# 3. INTERFACE UTILISATEUR & LOGIQUE DE CHAT
# ==========================================

with st.sidebar:
    st.title("⚙️ Paramètres")
    st.info("Stéthopote est ton assistant de révision. Ses réponses sont générées strictement à partir de tes collèges de médecine sourcés.")
    st.divider()
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🩺 Stéthopote", anchor=False)
st.caption("Ton binôme de révision médical")

# Chargement de la base
try:
    retriever = init_stethopote()
except Exception as e:
    st.error(f"Erreur fatale lors du chargement de la base : {e}")
    st.stop()

# Initialisation de la mémoire
if "messages" not in st.session_state:
    st.session_state.messages = []

# Réaffichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question_aleatoire = random.choice(QUESTIONS_EXEMPLES)

# Gestion de la nouvelle question
if prompt := st.chat_input(f"Ex: {question_aleatoire}"):
    
    # 1. Afficher et sauvegarder la question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Préparer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Je fouille dans tes collèges de médecine... 📚"):
            try:
                # ÉTAPE A : Recherche dans Pinecone + Jsonl
                relevant_docs = retriever.invoke(prompt)
                
                if not relevant_docs:
                    st.warning("Je n'ai rien trouvé d'assez précis dans tes collèges pour cette question.")
                else:
                    # ÉTAPE B : Compilation du contexte
                    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # ÉTAPE C : Préparation du LLM
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    system_prompt = """Tu es Stéthopote, un assistant médical bienveillant conçu pour aider une étudiante en médecine.
Pour répondre aux questions médicales, tu dois IMPÉRATIVEMENT utiliser le CONTEXTE fourni. Si l'information n'y est pas, dis-le.
Structure ta réponse avec des listes à puces et du gras pour les mots-clés."""

                    full_prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "Contexte des cours :\n{context}\n\nQuestion de l'étudiante : {query}")
                    ])
                    
                    chain = full_prompt | llm
                    
                    # ÉTAPE D : Traduction de l'historique pour LangChain (CORRECTION CRUCIALE ICI)
                    historique_langchain = []
                    # On ne prend PAS le dernier message (qui est la question actuelle) pour ne pas faire de doublon !
                    for msg in st.session_state.messages[:-1]:
                        role = "human" if msg["role"] == "user" else "ai"
                        historique_langchain.append((role, msg["content"]))
                    
                    # ÉTAPE E : Génération en streaming
                    stream = chain.stream({
                        "context": context, 
                        "query": prompt,
                        "chat_history": historique_langchain
                    })
                    
                    full_response = st.write_stream(chunk.content for chunk in stream)
                    
                    # Sauvegarde de la réponse dans la mémoire
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                    # ÉTAPE F : Affichage des sources
                    with st.expander("📚 Voir les sources consultées"):
                        for i, doc in enumerate(relevant_docs, 1):
                            doc_name = doc.metadata.get('document_name', 'Collège inconnu')
                            page_start = doc.metadata.get('page_start', '?')
                            page_end = doc.metadata.get('page_end', '?')
                            
                            if page_start == page_end:
                                pages_info = f"Page {page_start}"
                            else:
                                pages_info = f"Pages {page_start} à {page_end}"
                                
                            st.markdown(f"**{i}. {doc_name}** — *{pages_info}*")
                            
            except Exception as e:
                # Si jamais ça plante, ça n'effacera plus l'écran, ça affichera l'erreur en rouge !
                st.error(f"Une erreur est survenue pendant la réflexion de l'IA : {e}")