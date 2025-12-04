# tools/extract_cmapss.py
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from utils.llm_provider import llm_deterministic
from utils.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Prompt: usamos placeholders claros. Doble {{ }} para incluir JSON literal en la plantilla donde haga falta.
PROMPT_EXTRACT_CMAPSS = ChatPromptTemplate.from_template(
    """
Eres un asistente especializado en extraer datos estructurados para alimentar un modelo de predicción RUL basado en CMAPSS.

TU TAREA:
Extraer únicamente la información explícita mencionada por el usuario sobre el estado actual de un motor aeronáutico.
REGLAS:
- NO inventar valores.
- NO estimar sensores no mencionados.
- NO rellenar medias ni interpolaciones.
- Siempre devolver JSON válido, sin texto adicional ni explicaciones.

DATOS A EXTRAER:
1. unidad (identificador, si no se menciona → 0)
2. tiempo_ciclos (si no se menciona → 0)
3. configuraciones_operativas: lista de tres valores [setting_1, setting_2, setting_3] (si falta alguno → 0)
4. mediciones_sensores: objeto con los 21 sensores "s_1" .. "s_21". Si el usuario menciona alguno, asigna ese valor; si no → 0.
   Si hay actualizaciones, sobreescribe valores previos con los nuevos mencionados por el usuario.

SELECCIÓN DEL MODELO (FD):
- FD001 → condiciones de nivel del mar + solo HPC degradation
- FD002 → condiciones SEIS + solo HPC
- FD003 → nivel del mar + HPC y/o Fan degradation
- FD004 → condiciones SEIS + HPC y/o Fan degradation
Si no hay contexto suficiente → usar "FD001".

RECEPCIÓN DE DATOS PREVIOS:
- PreRUL_DATA (valores actuales): {pre_rul_data_json}

MENSAJE DEL USUARIO:
{message}

SALIDA:
Devuelve exactamente un JSON válido (como texto) con la forma:
{{
  "unidad": <int>,
  "tiempo_ciclos": <int>,
  "configuraciones_operativas": [<num>, <num>, <num>],
  "mediciones_sensores": {{ "s_1": <num>, ..., "s_21": <num> }},
  "modelo_seleccionado": "<FDxxx>"
}}
No añadas explicación ni texto adicional. Si no hay dato, usa 0. 
"""
)


@tool
def extract_cmapss_tool(message: str, pre_rul_data: dict) -> str:
    """
    Tool pura: recibe message y pre_rul_data, llama al LLM y devuelve JSON (string) validado.
    No modifica el 'state' ni hace side-effects.
    """
    try:
        print(f"<<<ENtra {pre_rul_data}")
        # Normalizar pre_rul_data a JSON legible para el prompt
        try:
            pre_rul_data_json = json.dumps(pre_rul_data, ensure_ascii=False)
        except Exception:
            # si hay elementos no serializables, construir de forma segura
            safe = {
                "unidad": int(pre_rul_data.get("unidad", 0)),
                "tiempo_ciclos": int(pre_rul_data.get("tiempo_ciclos", 0)),
                "configuraciones_operativas": list(pre_rul_data.get("configuraciones_operativas", [0, 0, 0])),
                "mediciones_sensores": {f"s_{i}": pre_rul_data.get("mediciones_sensores", {}).get(f"s_{i}", 0) for i in range(1, 22)},
                "modelo_seleccionado": pre_rul_data.get("modelo_seleccionado", "FD001")
            }
            pre_rul_data_json = json.dumps(safe, ensure_ascii=False)

        # Construir y ejecutar la cadena LLM
        chain = PROMPT_EXTRACT_CMAPSS | llm_deterministic
        response = chain.invoke({
            "message": message,
            "pre_rul_data_json": pre_rul_data_json
        })

        raw = response.content.strip()
        # Limpiar marcadores de código
        cleaned = raw.replace("```json", "").replace("```", "").strip()

        # Intentar parsear el JSON devuelto por la LLM
        try:
            parsed = json.loads(cleaned)
        except Exception as ex:
            # intentar extraer el primer bloque JSON dentro del texto
            import re
            m = re.search(r"(\{[\s\S]*\})", cleaned)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception as ex2:
                    logger.exception("No pude parsear JSON extraído por regex: %s", ex2)
                    return json.dumps({"error": "LLM output non-parseable"})
            else:
                logger.exception("LLM no devolvió JSON parseable. Raw: %s", cleaned)
                return json.dumps({"error": "LLM output non-parseable"})

        # Validación básica de la estructura esperada
        if not isinstance(parsed, dict):
            return json.dumps({"error": "LLM output is not a JSON object"})

        # Asegurar que keys existen y tienen el formato mínimo
        parsed.setdefault("unidad", 0)
        parsed.setdefault("tiempo_ciclos", 0)
        parsed.setdefault("configuraciones_operativas", [0, 0, 0])
        parsed.setdefault("mediciones_sensores", {f"s_{i}": 0 for i in range(1, 22)})
        parsed.setdefault("modelo_seleccionado", "FD001")

        # Normalizar sensores a 21 entradas
        sensors = parsed.get("mediciones_sensores", {}) or {}
        normalized_sensors = {f"s_{i}": sensors.get(f"s_{i}", 0) for i in range(1, 22)}
        parsed["mediciones_sensores"] = normalized_sensors

        # Normalizar configuraciones a longitud 3
        settings = parsed.get("configuraciones_operativas", [0, 0, 0])
        try:
            settings = list(settings)
        except Exception:
            settings = [0, 0, 0]
        while len(settings) < 3:
            settings.append(0)
        parsed["configuraciones_operativas"] = settings[:3]

        # Finalmente devolver JSON canónico como string
        return json.dumps(parsed, ensure_ascii=False)

    except Exception as e:
        logger.exception("Error en extract_cmapss_tool: %s", e)
        return json.dumps({"error": "exception_in_tool", "detail": str(e)})


# Registrar (si tu ToolRegistry sigue la misma API)
ToolRegistry.register("extract_cmapss", extract_cmapss_tool)
