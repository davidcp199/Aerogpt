import pandas as pd
import re

archivos = [
    "ASRS_DBOnline_2010_2014.xlsx",
    "ASRS_DBOnline_2014_2018.xlsx",
    "ASRS_DBOnline_2018_2022.xlsx",
    "ASRS_DBOnline_2022_2025.xlsx"
]

def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'<.*?>', '', texto)
    return texto.strip()

dataframes = []

for archivo in archivos:
    print(f"📂 Leyendo {archivo} ...")
    xls = pd.ExcelFile(archivo, engine="openpyxl")
    hoja = xls.sheet_names[0]  # usa la primera hoja
    df = pd.read_excel(xls, sheet_name=hoja, header=1)

    # Detecta columnas con texto narrativo
    cols_texto = [c for c in df.columns if re.search(r'synopsis|narrative|report', str(c), re.I)]
    if not cols_texto:
        print(f"⚠️ No se encontraron columnas de texto en {archivo}")
        continue

    df_texto = df[cols_texto].copy()
    df_texto.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in df_texto.columns]
    df_texto['Fuente'] = archivo

    # Limpieza de texto
    for col in df_texto.columns:
        if col != 'Fuente':
            df_texto[col] = df_texto[col].apply(limpiar_texto)

    dataframes.append(df_texto)

# Combinar todo
df_final = pd.concat(dataframes, ignore_index=True)
df_final.drop_duplicates(inplace=True)

print(f"✅ Registros totales combinados: {len(df_final)}")
print(f"🧾 Columnas incluidas: {list(df_final.columns)}")

df_final.to_csv("ASRS_Combinado_Textos.csv", index=False, encoding='utf-8-sig')
print("💾 Guardado como ASRS_Combinado_Textos.csv")
