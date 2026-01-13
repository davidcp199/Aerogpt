import pandas as pd
from pathlib import Path
import pickle
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from utils.llm_provider import paths_config
from langchain_openai import OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "SDR"
VECTOR_DIR = BASE_DIR / "data" / "vectorStores" / "sdr_store"
OUT_DOCS = VECTOR_DIR / "sdr_docs.pkl"

EMBEDDINGS = OpenAIEmbeddings()

# COLUMNAS RELEVANTES SDR
FIELDS = {
    "ocn": "OperatorControlNumber",
    "submission_date": "SubmissionDate",
    "aircraft_make": "AircraftMake",
    "aircraft_model": "AircraftModel",
    "component": "ComponentName",
    "part_number": "PartNumber",
    "discrepancy": "Discrepancy",
    "stage_operation": "StageOfOperationCode",
    "how_discovered": "HowDiscoveredCode",
}


def load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")

def safe_get(row, col):
    if col not in row:
        return ""
    val = row[col]
    return "" if pd.isna(val) else str(val).strip()

def build_document(row) -> Document | None:
    discrepancy = safe_get(row, FIELDS["discrepancy"])
    if not discrepancy:
        return None

    text = (
        f"Operator Control Number: {safe_get(row, FIELDS['ocn'])}\n"
        f"Submission Date: {safe_get(row, FIELDS['submission_date'])}\n"
        f"Aircraft: {safe_get(row, FIELDS['aircraft_make'])} {safe_get(row, FIELDS['aircraft_model'])}\n"
        f"Component: {safe_get(row, FIELDS['component'])}\n"
        f"Part Number: {safe_get(row, FIELDS['part_number'])}\n"
        f"Stage of Operation: {safe_get(row, FIELDS['stage_operation'])}\n"
        f"How Discovered: {safe_get(row, FIELDS['how_discovered'])}\n\n"
        f"Discrepancy:\n{discrepancy}"
    )

    metadata = {
        "source": "SDR",
        "source_type": "regulatory",
        "ocn": safe_get(row, FIELDS["ocn"]),
        "submission_date": safe_get(row, FIELDS["submission_date"]),
        "aircraft_make": safe_get(row, FIELDS["aircraft_make"]),
        "aircraft_model": safe_get(row, FIELDS["aircraft_model"]),
        "component": safe_get(row, FIELDS["component"]),
        "part_number": safe_get(row, FIELDS["part_number"]),
        "stage_operation": safe_get(row, FIELDS["stage_operation"]),
        "how_discovered": safe_get(row, FIELDS["how_discovered"]),
    }

    return Document(page_content=text, metadata=metadata)



def main():
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    docs = []

    excel_files = list(RAW_DIR.glob("*.xlsx"))
    if not excel_files:
        raise RuntimeError("No se encontraron archivos SDR .xlsx")

    for file in excel_files:
        df = load_excel(file)
        for _, row in df.iterrows():
            doc = build_document(row)
            if doc:
                docs.append(doc)

    if not docs:
        raise RuntimeError("No se generó ningún documento SDR. Revisar columnas.")

    # Guardar documentos
    with open(OUT_DOCS, "wb") as f:
        pickle.dump(docs, f)

    # Crear vectorstore
    vectorstore = FAISS.from_documents(docs, EMBEDDINGS)
    vectorstore.save_local(VECTOR_DIR)
    print(f"[OK] SDR vectorstore creado: {VECTOR_DIR}, documentos: {len(docs)}")

if __name__ == "__main__":
    main()
