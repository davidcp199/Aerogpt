import pandas as pd
import glob
import os

# 📂 Ruta donde están los archivos SDR
ruta = r"C:\Users\David\Documents\Master-Big-Data-Data-Sciencee-e-Inteligencia-Artificial\TFM\SDRS"

# 🔍 Buscar todos los archivos Excel que empiecen por "20251105_SDR_Export_"
archivos = glob.glob(os.path.join(ruta, "20251105_SDR_Export_*.xlsx"))

# 🧩 Mostrar lista de archivos detectados
print("📂 Archivos SDR encontrados:")
for f in archivos:
    print("  -", f)

print(f"\nTotal detectados: {len(archivos)}\n")

# Lista para los DataFrames
dfs = []

for archivo in archivos:
    print(f"📖 Leyendo {os.path.basename(archivo)} ...")
    try:
        df = pd.read_excel(archivo)
        df.dropna(how="all", inplace=True)
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ Error leyendo {archivo}: {e}")

# 🧮 Combinar todos los archivos
if dfs:
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.drop_duplicates(inplace=True)

    # 💾 Guardar resultado combinado
    salida = os.path.join(ruta, "FAA_SDR_full.csv")
    df_final.to_csv(salida, index=False, encoding="utf-8-sig")

    print(f"\n✅ Archivo final generado: {salida}")
    print(f"📊 Total de registros combinados: {len(df_final)}")
else:
    print("❌ No se encontraron archivos válidos para combinar.")
