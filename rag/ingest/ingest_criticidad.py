# ============================================================
# Ingestión global Criticidad Agent: ASRS, SDR, FAA, EASA, TECH
# ============================================================

import pickle
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from utils.llm_provider import paths_config
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parents[2]
METADATA_DIR = BASE_DIR / "data" / "metadata"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorStores"
CRITICIDAD_PATH = VECTORSTORE_DIR / "criticidad_store"

EMBEDDINGS = OpenAIEmbeddings()

# Pickles
PICKLES = {
    "ASRS": VECTORSTORE_DIR / "asrs_store" / "asrs_docs.pkl",
    "SDR": VECTORSTORE_DIR / "sdr_store" / "sdr_docs.pkl",
    "FAA": METADATA_DIR / "faa_acs_chunks.pkl",
    "EASA": METADATA_DIR / "easa_chunks.pkl",
    "TECH": METADATA_DIR / "tech_chunks.pkl"
}


def load_docs_from_pickle(pickle_path: Path, source_name: str):
    """Carga documentos desde pickle y añade metadata de origen"""
    if not pickle_path.exists():
        print(f"[WARNING] Pickle no encontrado: {pickle_path}")
        return []

    with open(pickle_path, "rb") as f:
        items = pickle.load(f)

    docs = []
    # Diferenciar si el pickle contiene objetos Document o dicts de chunks
    for item in items:
        if isinstance(item, Document):
            # Si ya es Document
            doc = Document(page_content=item.page_content,
                           metadata={**item.metadata, "source": source_name})
        elif isinstance(item, dict) and "text" in item:
            doc = Document(page_content=item["text"], metadata={"source": source_name})
        else:
            # Ignorar elementos no reconocidos
            continue
        docs.append(doc)
    return docs

def create_global_vectorstore(documents, out_path: Path):
    """Crea un vectorstore FAISS a partir de la lista completa de documentos"""
    if not documents:
        raise RuntimeError("No hay documentos para crear criticidad_store")

    out_path.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(documents, EMBEDDINGS)
    vectorstore.save_local(out_path)
    print(f"[OK] Criticidad vectorstore creado: {out_path}, documentos: {len(documents)}")



def main():
    print("=== Ingestión global para Criticidad Agent ===")
    all_docs = []

    for source, path in PICKLES.items():
        docs = load_docs_from_pickle(path, source)
        print(f"Documentos {source}: {len(docs)}")
        all_docs.extend(docs)

    print(f"Total documentos cargados: {len(all_docs)}")

    if not all_docs:
        raise RuntimeError("No se cargó ningún documento. Revisa los pickles.")

    create_global_vectorstore(all_docs, CRITICIDAD_PATH)

if __name__ == "__main__":
    main()
