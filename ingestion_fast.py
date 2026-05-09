import os
import json
import uuid
import time
from pathlib import Path

# On utilise le moteur natif, adieu les pertes de texte !
import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChromaDB
import chromadb

# ---------------------------------------------------------
# 1. LE COUTURIER & INJECTEUR DIRECT
# ---------------------------------------------------------
class MedicalHierarchyBuilder:
    def __init__(self, doc_id, doc_name, chroma_collection, docstore_file):
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.parent_counter = 1
        
        self.collection = chroma_collection
        self.docstore_file = docstore_file
        
        self.current_text = ""
        self.current_pages = set()
        
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        self.child_ids = []
        self.child_docs = []
        self.child_metas = []

    def add_page(self, text, page_num):
        # 1. Nettoyage du filigrane
        filigrane = "Ce livre a été acheté, scanné et publié par Faille du forum Amis-Med. Toute copie, redistribution, vente ou vol est strictement interdite, en particulier par le groupe du Dr Voleur Pour avoir plus d'exclusivités rejoindre nous sur www.amis-med.com et sur https://t.me/Faille_V2"
        text = text.replace(filigrane, "")

        # Si la page est VRAIMENT vide, on l'ignore sans rien casser
        if not text.strip(): 
            return

        self.current_text += text.strip() + "\n\n"
        self.current_pages.add(page_num)

        # On coupe si on atteint 3000 caractères
        if len(self.current_text) >= 3000:
            self._process_buffer(flush=False)

    def _process_buffer(self, flush=False):
        # 🚨 LE FIX DU TROU NOIR EST ICI 🚨
        if not self.current_text.strip(): 
            self.current_text = ""
            self.current_pages = set() # On purge l'historique !
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
            
            # Écriture DocStore
            parent_data = {
                "id": parent_id,
                "type": "parent",
                "document_name": self.doc_name,
                "page_start": p_start,
                "page_end": p_end,
                "content": p_text
            }
            self.docstore_file.write(json.dumps(parent_data, ensure_ascii=False) + "\n")

            # Préparation Chroma
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
                self._flush_chroma()
        
        # Reset propre des pages
        if flush or len(self.current_text) == 0:
            self.current_pages = set()
        else:
            derniere_page = max(self.current_pages)
            self.current_pages = {derniere_page}

    def _flush_chroma(self):
        if self.child_docs:
            print(f"   ➔ Insertion de {len(self.child_docs)} morceaux dans ChromaDB...")
            self.collection.add(
                ids=self.child_ids,
                documents=self.child_docs,
                metadatas=self.child_metas
            )
            self.child_ids.clear()
            self.child_docs.clear()
            self.child_metas.clear()

# ---------------------------------------------------------
# 2. PIPELINE PRINCIPAL (ULTRA RAPIDE)
# ---------------------------------------------------------
def process_and_store_medical_pdfs():
    input_dir = Path("data")
    output_dir = Path("store_with_page")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    docstore_path = output_dir / "docstore.jsonl"
    chroma_path = output_dir / "chroma_db"

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ Aucun fichier PDF trouvé.")
        return

    print("🔌 Démarrage de ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(name="stethopote_children")

    with open(docstore_path, "a", encoding="utf-8") as f_docstore:
        
        for pdf_path in pdf_files:
            print(f"\n⚡ Lancement du traitement : {pdf_path.name}")
            start_time = time.time()
            
            try:
                # Ouverture native avec PyMuPDF
                doc = fitz.open(pdf_path)
                total_pages = doc.page_count
                print(f"📄 Lecture brute de {total_pages} pages en cours...")
                
                doc_uuid = str(uuid.uuid4())
                builder = MedicalHierarchyBuilder(doc_uuid, pdf_path.name, collection, f_docstore)

                # Lecture instantanée page par page (plus besoin de batchs complexes)
                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    # get_text("text") extrait 100% du texte sélectionnable, sans rien jeter
                    texte_page = page.get_text("text") 
                    
                    real_page_num = page_num + 1 
                    builder.add_page(texte_page, real_page_num)

                doc.close()

                print("🧵 Couture finale...")
                builder._process_buffer(flush=True)
                builder._flush_chroma() 
                
                elapsed = time.time() - start_time
                print(f"✅ Terminé avec succès : {pdf_path.name} en {elapsed:.2f} secondes !")

            except Exception as e:
                print(f"❌ Erreur critique sur {pdf_path.name} : {e}")

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    
    # REINITIALISATION : On efface tes anciens essais ratés pour repartir à zéro
    if Path("store_with_page/docstore.jsonl").exists():
        os.remove("store_with_page/docstore.jsonl")
    
    process_and_store_medical_pdfs()