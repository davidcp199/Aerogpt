import requests
from bs4 import BeautifulSoup
import os
import time

# Carpeta donde guardar los PDFs
output_dir = r"C:\Users\David\Documents\Master-Big-Data-Data-Sciencee-e-Inteligencia-Artificial\TFM\data\raw\Airbus_FAST"
os.makedirs(output_dir, exist_ok=True)

# URL de la página de Airbus FAST Magazine
base_url = "https://aircraft.airbus.com/en/fast-magazine-articles"

# Headers para simular un navegador
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# Hacer request a la página
response = requests.get(base_url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Encontrar todos los enlaces a PDFs
pdf_links = []
for a in soup.find_all("a", href=True):
    href = a['href']
    if href.lower().endswith(".pdf") and "fast" in href.lower():
        pdf_links.append(href)

print(f"Encontrados {len(pdf_links)} PDFs.")

# Descargar cada PDF
for link in pdf_links:
    pdf_url = link
    # Algunas URLs son relativas
    if pdf_url.startswith("/"):
        pdf_url = "https://aircraft.airbus.com" + pdf_url

    # Generar nombre de archivo consistente
    file_name = pdf_url.split("/")[-1]
    file_path = os.path.join(output_dir, file_name)

    # Evitar descargar si ya existe
    if os.path.exists(file_path):
        print(f"{file_name} ya existe, saltando...")
        continue

    print(f"Descargando {file_name} ...")
    r = requests.get(pdf_url, headers=headers)
    with open(file_path, "wb") as f:
        f.write(r.content)

    # Espera de 1 segundo entre descargas para no sobrecargar el servidor
    time.sleep(1)

print("Descarga completada.")
