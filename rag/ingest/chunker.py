# Divide texto en chunks y los guarda en pickle

import os
import pickle

def chunk_text(text, chunk_size=1200, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_text_folder(input_folder, output_pickle):
    all_chunks = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(".txt"):
                txt_path = os.path.join(root, f)
                with open(txt_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    chunks = chunk_text(text)
                    all_chunks.extend([{"source": f, "text": c} for c in chunks])
    with open(output_pickle, "wb") as pkl_file:
        pickle.dump(all_chunks, pkl_file)
    print(f"[OK] Chunks guardados en {output_pickle}, total chunks: {len(all_chunks)}")
