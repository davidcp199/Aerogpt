# tools/tool_output_to_df.py
# CAMBIO: convertido a función utilitaria (no @tool) para evitar serialización innecesaria
import pandas as pd
import json

def tool_output_to_df(tool_output):
    """
    Convierte la estructura extraída por extract_cmapss a DataFrame en el formato esperado por predictor.
    tool_output puede ser dict o string JSON.
    """
    if isinstance(tool_output, str):
        try:
            tool_output = json.loads(tool_output)
        except Exception:
            # Intento de limpieza básica antes de reintentar
            s = tool_output.replace("```json", "").replace("```", "")
            tool_output = json.loads(s)

    settings = tool_output.get("configuraciones_operativas", [0,0,0])
    sensors = tool_output.get("mediciones_sensores", {}) or {}
    # Asegurar keys s_1..s_21
    sensor_dict = {f"s_{i}": sensors.get(f"s_{i}", 0) for i in range(1, 22)}

    flat = {
        "setting_1": [settings[0] if len(settings) > 0 else 0],
        "setting_2": [settings[1] if len(settings) > 1 else 0],
        "setting_3": [settings[2] if len(settings) > 2 else 0],
        **{k: [v] for k, v in sensor_dict.items()}
    }
    return pd.DataFrame(flat)
