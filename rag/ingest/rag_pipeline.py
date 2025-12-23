# Pipeline completo: PDFs -> Chunks -> Vectorstores

import os
from ingest_pdfs import process_pdf_folder
from chunker import chunk_text_folder
from build_vectorstores import create_vectorstore


RAW_PDF_PATHS = {
    "FAA_ACs": r"C:\Users\David\Documents\AeroGPT\data\raw\FAA_ACs",
    "EASA": r"C:\Users\David\Documents\AeroGPT\data\raw\EASA",
    "Airbus_FAST": r"C:\Users\David\Documents\AeroGPT\data\raw\Airbus_FAST"
}

PROCESSED_PATH = r"C:\Users\David\Documents\AeroGPT\data\processed"
METADATA_PATH = r"C:\Users\David\Documents\AeroGPT\data\metadata"
VECTORSTORE_PATH = r"C:\Users\David\Documents\AeroGPT\data\vectorStores"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(METADATA_PATH, exist_ok=True)
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

# Extraer PDFs a TXT

print("==> Paso 1: Extracción de PDFs a TXT")
for key, path in RAW_PDF_PATHS.items():
    out_folder = os.path.join(PROCESSED_PATH, key)
    print(f"Procesando {key}...")
    process_pdf_folder(path, out_folder)

# Chunking
print("\n==> Paso 2: Crear chunks de los TXT")
chunk_files = {}
for key in RAW_PDF_PATHS.keys():
    input_folder = os.path.join(PROCESSED_PATH, key)
    output_pickle = os.path.join(METADATA_PATH, f"{key.lower()}_chunks.pkl")
    chunk_text_folder(input_folder, output_pickle)
    chunk_files[key] = output_pickle

# Construir vectorstores FAISS
print("\n==> Paso 3: Crear vectorstores FAISS")

# Vectorstore regulatorio: FAA_ACs + EASA
create_vectorstore(
    chunk_files["FAA_ACs"],
    chunk_files["EASA"],
    vectorstore_path=os.path.join(VECTORSTORE_PATH, "regulatory_store")
)

# Vectorstore técnico: Airbus_FAST
create_vectorstore(
    chunk_files["Airbus_FAST"],
    vectorstore_path=os.path.join(VECTORSTORE_PATH, "technical_store")
)

print("\n✅ Pipeline completo finalizado. Vectorstores listos para RAG.")
