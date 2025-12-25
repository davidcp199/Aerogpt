import pandas as pd
from pathlib import Path
import pickle
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from utils.llm_provider import paths_config
from langchain_openai import OpenAIEmbeddings

# ====================================================
# CONFIGURACIÓN DE RUTAS
# ====================================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "ASRS"
VECTOR_DIR = BASE_DIR / "data" / "vectorStores" / "asrs_store"
OUT_DOCS = VECTOR_DIR / "asrs_docs.pkl"

EMBEDDINGS = OpenAIEmbeddings()

# ====================================================
# COLUMNAS RELEVANTES
# ====================================================
FIELDS = {
    "acn": "ACN",
    "aircraft": "Aircraft 1.Make Model Name",
    "far": "Aircraft 1.Operating Under FAR Part",
    "phase": "Aircraft 1.Flight Phase",
    "system": "Aircraft Component",
    "primary_problem": "Events.Primary Problem",
    "anomaly": "Assessments.Anomaly",
    "detector": "Events.Detector",
    "result": "Events.Result",
    "narrative": "Report 1.Narrative",
    "synopsis": "Report 1.Synopsis",
}

# ====================================================
# HELPERS
# ====================================================
def load_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=[0, 1], engine="openpyxl")
    df.columns = [f"{a}.{b}" if str(b) != "nan" else f"{a}" for a, b in df.columns]
    return df

def safe_get(row, col):
    if col not in row:
        return ""
    val = row[col]
    return "" if pd.isna(val) else str(val).strip()

def build_document(row) -> Document | None:
    narrative = safe_get(row, FIELDS["narrative"])
    synopsis = safe_get(row, FIELDS["synopsis"])
    if not narrative and not synopsis:
        return None

    text = (
        f"Aircraft: {safe_get(row, FIELDS['aircraft'])}\n"
        f"FAR Part: {safe_get(row, FIELDS['far'])}\n"
        f"Flight Phase: {safe_get(row, FIELDS['phase'])}\n"
        f"System / Component: {safe_get(row, FIELDS['system'])}\n"
        f"Primary Problem: {safe_get(row, FIELDS['primary_problem'])}\n"
        f"Anomaly: {safe_get(row, FIELDS['anomaly'])}\n\n"
        f"NARRATIVE:\n{narrative}\n\nSYNOPSIS:\n{synopsis}"
    )

    metadata = {
        "source": "ASRS",
        "source_type": "regulatory",
        "acn": safe_get(row, FIELDS["acn"]),
        "aircraft": safe_get(row, FIELDS["aircraft"]),
        "far_part": safe_get(row, FIELDS["far"]),
        "flight_phase": safe_get(row, FIELDS["phase"]),
        "system": safe_get(row, FIELDS["system"]),
        "primary_problem": safe_get(row, FIELDS["primary_problem"]),
        "anomaly": safe_get(row, FIELDS["anomaly"]),
    }

    return Document(page_content=text, metadata=metadata)

# ====================================================
# MAIN
# ====================================================
def main():
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    docs = []

    for file in RAW_DIR.glob("*.xlsx"):
        df = load_excel(file)
        for _, row in df.iterrows():
            doc = build_document(row)
            if doc:
                docs.append(doc)

    if not docs:
        raise RuntimeError("No se generó ningún documento ASRS.")

    # Guardar documentos
    with open(OUT_DOCS, "wb") as f:
        pickle.dump(docs, f)

    # Crear vectorstore
    vectorstore = FAISS.from_documents(docs, EMBEDDINGS)
    vectorstore.save_local(VECTOR_DIR)
    print(f"[OK] ASRS vectorstore creado: {VECTOR_DIR}, documentos: {len(docs)}")

if __name__ == "__main__":
    main()
