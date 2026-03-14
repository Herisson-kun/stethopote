import os
import shutil
import gc  # Pour le nettoyage forcé de la mémoire
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

def split_pdf(input_path, output_folder, chunk_size=10): # CHUNK DE 10 PAGES
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)
    chunk_paths = []
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for i in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        end_page = min(i + chunk_size, total_pages)
        for page_num in range(i, end_page):
            writer.add_page(reader.pages[page_num])
        
        chunk_filename = output_folder / f"temp_part_{i//chunk_size}.pdf"
        with open(chunk_filename, "wb") as f:
            writer.write(f)
        chunk_paths.append(chunk_filename)
    return chunk_paths

def process_medical_pdfs():
    input_dir = Path("data")
    output_dir = Path("data_processed")
    temp_dir = Path("data/temp_chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CONFIGURATION DE SURVIE
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    # On limite l'usage des ressources
    pipeline_options.accelerator_options.num_threads = 1

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    pdf_files = list(input_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        if "temp_part_" in pdf_path.name: continue
        
        md_final_path = output_dir / f"{pdf_path.stem}.md"
        if md_final_path.exists():
            print(f"⏩ Déjà fait : {pdf_path.name}")
            continue

        print(f"✂️ Découpage de {pdf_path.name} en mini-morceaux (10 pages)...")
        try:
            chunks = split_pdf(pdf_path, temp_dir, chunk_size=10)
            all_markdown = []

            for i, chunk_path in enumerate(chunks):
                print(f"⏳ Traitement du mini-morceau {i+1}/{len(chunks)}...")
                try:
                    result = converter.convert(chunk_path)
                    all_markdown.append(result.document.export_to_markdown())
                    
                    # --- NETTOYAGE AGRESSIF ---
                    del result
                    gc.collect() # On force Python à vider la RAM
                except Exception as chunk_err:
                    print(f"⚠️ Erreur sur le morceau {i+1} (page {i*10}), on continue quand même : {chunk_err}")

            with open(md_final_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_markdown))
            
            print(f"✅ Terminé : {pdf_path.stem} fusionné !")
            shutil.rmtree(temp_dir)

        except Exception as e:
            print(f"❌ Erreur critique : {e}")

if __name__ == "__main__":
    process_medical_pdfs()