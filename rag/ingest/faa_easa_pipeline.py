import os
from pathlib import Path
from ingest_pdfs import process_pdf_folder
from chunker import chunk_text_folder

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PDF_PATHS = {
    "FAA_ACs": BASE_DIR / "data" / "raw" / "FAA_ACs",
    "EASA": BASE_DIR / "data" / "raw" / "EASA"
}

PROCESSED_PATH = BASE_DIR / "data" / "processed"
METADATA_PATH = BASE_DIR / "data" / "metadata"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(METADATA_PATH, exist_ok=True)

# ============================================================
# Convertir PDFs a TXT
# ============================================================
print("==> Paso 1: Extracción de PDFs a TXT")
for key, path in RAW_PDF_PATHS.items():
    out_folder = PROCESSED_PATH / key
    os.makedirs(out_folder, exist_ok=True)
    print(f"Procesando {key}...")
    process_pdf_folder(path, out_folder)

# ============================================================
# Crear chunks y guardar pickles
# ============================================================
print("\n==> Paso 2: Chunking y creación de pickles")
for key in RAW_PDF_PATHS.keys():
    input_folder = PROCESSED_PATH / key
    output_pickle = METADATA_PATH / f"{key.lower()}_chunks.pkl"
    chunk_text_folder(input_folder, output_pickle)
    print(f"[OK] Pickle generado: {output_pickle}")

print("\nPipeline FAA/EASA completado. Pickles listos para Criticidad Agent.")
