import os
import json
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---> NOUVEAU : Les outils Pinecone et OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ---------------------------------------------------------
# 1. LE COUTURIER & INJECTEUR CLOUD
# ---------------------------------------------------------
class MedicalHierarchyBuilder:
    def __init__(self, doc_id, doc_name, vectorstore, docstore_file):
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.parent_counter = 1
        
        self.vectorstore = vectorstore # On range directement dans Pinecone
        self.docstore_file = docstore_file
        
        self.current_text = ""
        self.current_pages = set()
        
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        self.child_ids = []
        self.child_docs = []
        self.child_metas = []
        self.total_inserted = 0 

    def add_page(self, text, page_num):
        filigrane = "Ce livre a été acheté, scanné et publié par Faille du forum Amis-Med. Toute copie, redistribution, vente ou vol est strictement interdite, en particulier par le groupe du Dr Voleur Pour avoir plus d'exclusivités rejoindre nous sur www.amis-med.com et sur https://t.me/Faille_V2"
        text = text.replace(filigrane, "")

        if not text.strip(): 
            return

        self.current_text += text.strip() + "\n\n"
        self.current_pages.add(page_num)

        if len(self.current_text) >= 3000:
            self._process_buffer(flush=False)

    def _process_buffer(self, flush=False):
        if not self.current_text.strip(): 
            self.current_text = ""
            self.current_pages = set() 
            return

        parents_text = self.parent_splitter.split_text(self.current_text)
        
        if not flush and len(parents_text) > 1:
            parents_to_process = parents_text[:-1]
            self.current_text = parents_text[-1] 
        else:
            parents_to_process = parents_text
            self.current_text = ""

        pages_list = sorted(list(self.current_pages))
        p_start = pages_list[0] if pages_list else -1
        p_end = pages_list[-1] if pages_list else -1
        
        for p_text in parents_to_process:
            parent_id = f"{self.doc_id}::parent{self.parent_counter:04d}"
            
            # Sauvegarde locale du Parent (Ça ira sur GitHub, c'est tout léger)
            parent_data = {
                "id": parent_id,
                "type": "parent",
                "document_name": self.doc_name,
                "page_start": p_start,
                "page_end": p_end,
                "content": p_text
            }
            self.docstore_file.write(json.dumps(parent_data, ensure_ascii=False) + "\n")

            # Préparation des enfants pour le Cloud
            children_text = self.child_splitter.split_text(p_text)
            for i, c_text in enumerate(children_text):
                child_id = f"{parent_id}::child{i+1:04d}"
                self.child_ids.append(child_id)
                self.child_docs.append(c_text)
                self.child_metas.append({
                    "parent_id": parent_id,
                    "document_name": self.doc_name,
                    "page_start": p_start,
                    "page_end": p_end
                })
            
            self.parent_counter += 1

            if len(self.child_docs) >= 500:
                self._flush_pinecone()
        
        if flush or len(self.current_text) == 0:
            self.current_pages = set()
        else:
            derniere_page = max(self.current_pages)
            self.current_pages = {derniere_page}

    def _flush_pinecone(self):
        if self.child_docs:
            nb_docs = len(self.child_docs)
            self.total_inserted += nb_docs
            tqdm.write(f"      ➔ [Pinecone] Envoi de {nb_docs} morceaux dans le Cloud... (Total : {self.total_inserted})")
            
            # ---> NOUVEAU : LangChain envoie tout à Pinecone d'un seul coup
            self.vectorstore.add_texts(
                texts=self.child_docs,
                metadatas=self.child_metas,
                ids=self.child_ids
            )
            
            self.child_ids.clear()
            self.child_docs.clear()
            self.child_metas.clear()

# ---------------------------------------------------------
# 2. PIPELINE PRINCIPAL 
# ---------------------------------------------------------
def process_and_store_medical_pdfs():
    load_dotenv() 
    
    input_dir = Path("data")
    output_dir = Path("store_with_page")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    docstore_path = output_dir / "docstore.jsonl"
    
    # 🚨 ON NE CRÉE PLUS DE DOSSIER CHROMA_DB 🚨

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ Aucun fichier PDF trouvé dans 'data/'.")
        return

    print("="*60)
    print("🔌 DÉMARRAGE DE L'INGESTION VERS PINECONE CLOUD")
    print(f"📚 {len(pdf_files)} fichier(s) détecté(s)")
    print("="*60)

    # ---> NOUVEAU : Connexion à Pinecone via LangChain
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name="stethopote", # Le nom exact de l'index que tu as créé à l'étape 1
        embedding=embeddings
    )

    with open(docstore_path, "a", encoding="utf-8") as f_docstore:
        
        for file_idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n▶️ [{file_idx}/{len(pdf_files)}] Traitement de : {pdf_path.name}")
            start_time = time.time()
            
            try:
                doc = fitz.open(pdf_path)
                total_pages = doc.page_count
                doc_uuid = str(uuid.uuid4())
                
                builder = MedicalHierarchyBuilder(doc_uuid, pdf_path.name, vectorstore, f_docstore)

                for page_num in tqdm(range(total_pages), desc="   📖 Lecture", unit="page", ascii=False):
                    page = doc.load_page(page_num)
                    texte_page = page.get_text("text") 
                    real_page_num = page_num + 1 
                    builder.add_page(texte_page, real_page_num)

                doc.close()

                print("   🧵 Couture finale et dernier envoi Cloud...")
                builder._process_buffer(flush=True)
                builder._flush_pinecone() 
                
                elapsed = time.time() - start_time
                print(f"✅ FINI : {pdf_path.name} ({builder.total_inserted} sous-parties envoyées !)")

            except Exception as e:
                print(f"❌ Erreur critique sur {pdf_path.name} : {e}")

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    
    if Path("store_with_page/docstore.jsonl").exists():
        os.remove("store_with_page/docstore.jsonl")
    
    process_and_store_medical_pdfs()