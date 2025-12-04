# tools/tool_output_to_df.py
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def tool_output_to_df(tool_output):
    """
    Convierte la salida de extract_cmapss_tool (dict o JSON string) a DataFrame con:
    columnas: setting_1, setting_2, setting_3, s_1 ... s_21
    Devuelve un pd.DataFrame con una sola fila.
    Lanza excepciones en caso de formato inválido.
    """
    # Aceptamos dict o string JSON
    if isinstance(tool_output, str):
        try:
            parsed = json.loads(tool_output)
        except Exception as e:
            # Intentar limpiar triples backticks
            s = tool_output.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(s)

    elif isinstance(tool_output, dict):
        parsed = tool_output
    else:
        raise ValueError("tool_output must be a dict or JSON string")

    # Validaciones mínimas
    if not isinstance(parsed, dict):
        raise ValueError("Parsed tool output is not a dict")

    settings = parsed.get("configuraciones_operativas", [0,0,0]) or [0,0,0]
    if not isinstance(settings, (list, tuple)):
        settings = [settings]

    # Asegurar longitud 3
    settings = list(settings)
    while len(settings) < 3:
        settings.append(0)
    settings = settings[:3]

    sensors = parsed.get("mediciones_sensores", {}) or {}
    # Normalizar sensores 1..21
    sensor_dict = {}
    for i in range(1, 22):
        key = f"s_{i}"
        val = sensors.get(key, 0)
        # Intentar convertir a float/int
        try:
            # podría venir "12" o 12.0
            if val is None or val == "":
                numeric = 0
            else:
                numeric = float(val)
                # si es entero en origen, mantener int no es necesario pero convertimos a float para estabilidad
        except Exception:
            logger.debug("No numeric sensor value for %s: %r, defaulting to 0", key, val)
            numeric = 0.0
        sensor_dict[key] = numeric

    flat = {
        "setting_1": [float(settings[0])],
        "setting_2": [float(settings[1])],
        "setting_3": [float(settings[2])],
    }
    # Añadir sensores
    for k, v in sensor_dict.items():
        flat[k] = [v]

    df = pd.DataFrame(flat)
    return df
