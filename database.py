import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain & Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage.file_system import LocalFileStore
from langchain_classic.storage.encoder_backed import EncoderBackedStore
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import pickle # Pour la traduction Document -> Bytes

load_dotenv()

def build_vector_db():
    # 1. Configuration des chemins
    input_dir = Path("data_processed")
    store_dir = Path("store")
    chroma_dir = store_dir / "chroma_db"
    docstore_dir = store_dir / "docstore_db"
    
    store_dir.mkdir(parents=True, exist_ok=True)
    docstore_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialisation des Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 3. Setup du Vectorstore (Chroma)
    vectorstore = Chroma(
        collection_name="stethopote_med",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir)
    )

    # 4. Setup du Docstore avec ENCODEUR (Version corrigée 🛠️)
    underlying_store = LocalFileStore(str(docstore_dir))
    
    # On définit les fonctions de traduction
    def encode_value(doc) -> bytes:
        return pickle.dumps(doc)
        
    def decode_value(value: bytes):
        return pickle.loads(value)

    # LangChain attend : (store, key_encoder, value_encoder, value_deserializer)
    # Pour un ParentDocumentRetriever, on n'encode que les VALEURS (le document)
    docstore = EncoderBackedStore(
        store=underlying_store,
        key_encoder=lambda k: k,        # Les clés restent des chaînes (ID)
        value_serializer=encode_value,
        value_deserializer=decode_value
    )

    # 5. Définition des Splitters
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o", chunk_size=1000, chunk_overlap=100
    )
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o", chunk_size=200, chunk_overlap=20
    )

    # 6. Le Retriever (il utilise maintenant le store encodé)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore, # Utilise le store traduit
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 7. Ingestion Incrémentale
    md_files = list(input_dir.glob("*.md"))
    log_file = store_dir / "processed_files.txt"
    processed_files = []
    if log_file.exists():
        processed_files = log_file.read_text().splitlines()

    print(f"🧐 Analyse des nouveaux Markdowns...")
    
    new_docs_count = 0
    for md_path in md_files:
        if md_path.name in processed_files:
            print(f"⏩ Déjà indexé : {md_path.name}")
            continue

        print(f"🧠 Indexation de : {md_path.name}...")
        loader = TextLoader(str(md_path), encoding="utf-8")
        docs = loader.load()
        
        # Ajout à la base
        retriever.add_documents(docs)
        
        # Mise à jour du log
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(md_path.name + "\n")
        
        new_docs_count += 1
        print(f"✅ {md_path.name} ajouté au store.")

    if new_docs_count == 0:
        print("✨ Tout est déjà à jour. Rien à faire !")
    else:
        print(f"\n🚀 Terminé ! {new_docs_count} nouveau(x) collège(s) ajouté(s) au store.")

if __name__ == "__main__":
    build_vector_db()