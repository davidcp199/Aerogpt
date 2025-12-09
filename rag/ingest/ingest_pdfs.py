# ingest_pdfs.py
# =========================
# Funciones para procesar PDFs y guardarlos como .txt
# =========================

import os
from extract_text import extract_text_from_pdf

def process_pdf_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, f)
                txt_name = f.replace(".pdf", ".txt")
                txt_path = os.path.join(output_folder, txt_name)
                if not os.path.exists(txt_path):
                    try:
                        text = extract_text_from_pdf(pdf_path)
                        with open(txt_path, "w", encoding="utf-8") as out_file:
                            out_file.write(text)
                        print(f"[OK] {f} → {txt_path}")
                    except Exception as e:
                        print(f"[ERROR] {f} -> {e}")
