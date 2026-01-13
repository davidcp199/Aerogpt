import os
from pathlib import Path
from ingest_pdfs import process_pdf_folder
from chunker import chunk_text_folder


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_TECH_PATH = BASE_DIR / "data" / "raw" / "Airbus_FAST"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "TECH"
METADATA_PATH = BASE_DIR / "data" / "metadata"

TECH_PICKLE = METADATA_PATH / "tech_chunks.pkl"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(METADATA_PATH, exist_ok=True)

# PASO 1: PDFs -> TXT
print("==> Paso 1: Extracción de PDFs técnicos a TXT")

process_pdf_folder(RAW_TECH_PATH, PROCESSED_PATH)

# PASO 2: Chunking -> Pickle
print("==> Paso 2: Chunking técnico")

chunk_text_folder(PROCESSED_PATH, TECH_PICKLE)

print(f"[OK] Pickle TECH generado: {TECH_PICKLE}")
print("Pipeline TECH completado correctamente.")
