import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage.file_system import LocalFileStore
from langchain_classic.storage.encoder_backed import EncoderBackedStore
import pickle
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURATION INITIALE & STYLE
# ==========================================
load_dotenv()
st.set_page_config(page_title="Stéthopote 🩺", page_icon="🩺", layout="centered")

# Style CSS pour optimiser l'affichage sur iPad
st.markdown("""
    <style>
    .stChatMessage { font-size: 1.1rem !important; }
    .stMarkdown { line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INITIALISATION DU CERVEAU (Mise en cache)
# ==========================================
@st.cache_resource
def init_stethopote():
    """Initialise la mémoire (DB + Retriever). Exécuté UNE SEULE FOIS au démarrage."""
    
    # 1. Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Chemins vers ta base de données (générée par database.py)
    chroma_dir = "store/chroma_db"
    docstore_dir = "store/docstore_db"
    
    # 3. Chargement du VectorStore (Enfants - Recherche mathématique)
    vectorstore = Chroma(
        collection_name="stethopote_med",
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )
    
    # 4. Chargement du DocStore (Parents - Texte complet avec Pickle)
    underlying_store = LocalFileStore(docstore_dir)
    store = EncoderBackedStore(
        store=underlying_store,
        key_encoder=lambda k: k,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )
    
    # 5. Splitters (Obligatoires pour la validation Pydantic, même en lecture seule)
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o", chunk_size=1000, chunk_overlap=100
    )
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o", chunk_size=200, chunk_overlap=20
    )
    
    # 6. Création du Retriever principal
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
st.title("🩺 Stéthopote")
st.caption("Ton binôme de révision médical (Mode Rapide)")

# --- CHARGEMENT DE LA BASE ---
# Se fait ici, hors de la zone de saisie, pour éviter les lags à chaque question
try:
    retriever = init_stethopote()
except Exception as e:
    st.error(f"Erreur fatale lors du chargement de la mémoire : {e}")
    st.stop()

# --- GESTION DE L'HISTORIQUE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des anciens messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ZONE DE CHAT ---
if prompt := st.chat_input("Pose ta question (ex: Signes de la pneumonie franche lobaire aiguë ?)"):
    
    # 1. Afficher et sauvegarder la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Générer et afficher la réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Je fouille dans tes collèges de médecine... 📚"):
            try:
                # ÉTAPE A : Recherche dans ChromaDB
                relevant_docs = retriever.invoke(prompt)
                
                if not relevant_docs:
                    st.warning("Je n'ai rien trouvé d'assez précis dans les cours pour cette question.")
                else:
                    # ÉTAPE B : Préparation du contexte
                    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # ÉTAPE C : Configuration du LLM
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    
                    system_prompt = """Tu es Stéthopote, un assistant médical bienveillant conçu pour aider une étudiante en médecine.
                    
Pour répondre aux questions médicales, tu dois IMPÉRATIVEMENT utiliser le CONTEXTE fourni. Si l'information n'y est pas, dis-le.
Cependant, tu as aussi accès à l'HISTORIQUE de la conversation. Tu peux t'en servir pour faire le lien avec les questions précédentes ou répondre aux questions d'ordre général de l'étudiante.

Structure ta réponse avec des listes à puces et du gras pour les mots-clés."""

                    # On crée la recette du Prompt en incluant l'historique
                    full_prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"), # <-- L'historique s'insère ici !
                        ("user", "Contexte des cours :\n{context}\n\nQuestion de l'étudiante : {query}")
                    ])
                    
                    chain = full_prompt | llm
                    
                    # On traduit l'historique de Streamlit dans un format que LangChain comprend
                    historique_langchain = []
                    for msg in st.session_state.messages: # On prend les anciens messages
                        historique_langchain.append((msg["role"], msg["content"]))
                    
                    # On lance la génération (Streaming) en lui passant les 3 variables !
                    stream = chain.stream({
                        "context": context, 
                        "query": prompt,
                        "chat_history": historique_langchain
                    })
                    
                    full_response = st.write_stream(chunk.content for chunk in stream)
                    
                    # Sauvegarde dans l'historique
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                    # ÉTAPE E : Affichage des sources (Pratique pour vérifier d'où vient l'info)
                    with st.expander("Voir les sources consultées (Parents)"):
                        for i, doc in enumerate(relevant_docs, 1):
                            source = doc.metadata.get('source', 'Source inconnue')
                            st.write(f"**Extrait {i} ({source})**")
                            st.caption(doc.page_content[:300] + "...")
                            
            except Exception as e:
                st.error(f"Oups, j'ai eu un petit souci technique : {e}")