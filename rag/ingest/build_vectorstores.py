# Crea vectorstore FAISS para RAG a partir de chunks almacenados en archivos pickle.
import pickle
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


def create_vectorstore(*pickle_files, vectorstore_path):
    os.makedirs(vectorstore_path, exist_ok=True)
    all_docs = []
    for pkl_file in pickle_files:
        with open(pkl_file, "rb") as pf:
            chunks = pickle.load(pf)
            for c in chunks:
                all_docs.append(Document(page_content=c["text"], metadata={"source": c["source"]}))
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(all_docs, embeddings)
    store.save_local(vectorstore_path)
    print(f"[OK] Vectorstore creado en {vectorstore_path}, documentos: {len(all_docs)}")

if __name__ == "__main__":
    create_vectorstore(
        "C:/Users/David/Documents/AeroGPT/data/metadata/faa_ac_chunks.pkl",
        "C:/Users/David/Documents/AeroGPT/data/metadata/easa_cs_chunks.pkl",
        vectorstore_path="C:/Users/David/Documents/AeroGPT/data/vectorStores/regulatory_store"
    )

    create_vectorstore(
        "C:/Users/David/Documents/AeroGPT/data/metadata/airbus_fast_chunks.pkl",
        vectorstore_path="C:/Users/David/Documents/AeroGPT/data/vectorStores/technical_store"
    )
