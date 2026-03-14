import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.storage._lc_store_adapter import StorageStoreAdapter
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuration initiale
load_dotenv()
st.set_page_config(page_title="Stéthopote 🩺", page_icon="🩺", layout="centered")

# Style CSS pour optimiser l'affichage sur iPad (plus de contraste et de lisibilité)
st.markdown("""
    <style>
    .stChatMessage { font-size: 1.1rem !important; }
    .stMarkdown { line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_stethopote():
    """Initialise le cerveau de l'assistant (DB + Retriever)"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Chemins des bases (pré-calculées avec ton script database.py)
    chroma_dir = "chroma_db"
    docstore_dir = "docstore_db"
    
    # Chargement du VectorStore (Enfants)
    vectorstore = Chroma(
        collection_name="stethopote_med",
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )
    
    # Chargement du DocStore (Parents)
    fs = LocalFileStore(docstore_dir)
    store = StorageStoreAdapter(fs)
    
    # Recréation du Retriever Parent-Document
    # Note: On n'a pas besoin des splitters ici car la DB est déjà peuplée
    base_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
    )
    
    # Ajout du Reranker Cohere pour la précision chirurgicale
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=4)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever

# 2. Interface Utilisateur
st.title("🩺 Stéthopote")
st.caption("Ton binôme de révision médical boosté par GPT-5-mini")

# Initialisation de l'historique du chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Pose ta question (ex: Signes de l'insuffisance cardiaque...)"):
    # Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Je fouille dans tes collèges de médecine..."):
            try:
                # 3. Récupération des données
                retriever = init_stethopote()
                relevant_docs = retriever.invoke(prompt)
                
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # 4. Génération de la réponse
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Remplace par gpt-5-mini quand dispo
                
                system_prompt = """Tu es Stéthopote, un assistant médical bienveillant pour une étudiante. 
                Utilise uniquement le contexte fourni (issus des collèges officiels) pour répondre. 
                Si tu ne sais pas, dis-le. Structure ta réponse avec des listes à puces pour la clarté."""
                
                full_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("user", "Contexte :\n{context}\n\nQuestion : {query}")
                ])
                
                chain = full_prompt | llm
                response = chain.invoke({"context": context, "query": prompt})
                
                # Affichage
                st.markdown(response.content)
                
                # Sources (optionnel, pour vérification)
                with st.expander("Voir les sources consultées"):
                    for doc in relevant_docs:
                        st.write(f"- {doc.metadata.get('source', 'Source inconnue')}")
                
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
            except Exception as e:
                st.error(f"Oups, j'ai eu un petit souci : {e}")