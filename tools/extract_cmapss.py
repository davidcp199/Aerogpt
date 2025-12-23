import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from utils.llm_provider import llm_deterministic
from utils.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

PROMPT_EXTRACT_CMAPSS = ChatPromptTemplate.from_template(
    """
Eres un asistente especializado en extraer datos estructurados para alimentar un modelo de predicción RUL basado en CMAPSS.

TU TAREA:
Extraer únicamente la información explícita mencionada por el usuario sobre el estado actual de un motor aeronáutico.
- NO inventar valores.
- NO estimar sensores no mencionados.
- NO rellenar medias ni interpolaciones.
- Devuelve JSON válido, sin texto adicional.

DATOS A EXTRAER:
1. unidad (int, si no se menciona → 0)
2. tiempo_ciclos (int, si no se menciona → 0)
3. configuraciones_operativas: lista de tres valores [setting_1, setting_2, setting_3] (si falta alguno → 0)
4. mediciones_sensores: objeto con los 21 sensores "s_1" .. "s_21" (si el usuario menciona alguno, asigna ese valor; si no → 0)
5. modelo_seleccionado (FD001..FD004, por defecto FD001 si no se indica)

Mensaje del usuario:
{message}

Salida esperada:
Devuelve exactamente un JSON válido con esta forma:
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
def extract_cmapss_tool(message: str) -> str:
    """
    Tool pura: recibe solo el mensaje del usuario y devuelve JSON string con los valores mencionados.
    """
    try:
        chain = PROMPT_EXTRACT_CMAPSS | llm_deterministic
        response = chain.invoke({"message": message})

        raw = response.content.strip()
        # Limpiar posibles ```json
        cleaned = raw.replace("```json", "").replace("```", "").strip()

        # Parsear JSON
        try:
            parsed = json.loads(cleaned)
        except Exception:
            # extraer primer bloque JSON
            import re
            m = re.search(r"(\{[\s\S]*\})", cleaned)
            if m:
                parsed = json.loads(m.group(1))
            else:
                return json.dumps({"error": "LLM output non-parseable"})

        # Validación mínima
        parsed.setdefault("unidad", 0)
        parsed.setdefault("tiempo_ciclos", 0)
        parsed.setdefault("configuraciones_operativas", [0, 0, 0])
        parsed.setdefault("mediciones_sensores", {f"s_{i}": 0 for i in range(1, 22)})
        parsed.setdefault("modelo_seleccionado", "FD001")

        # Normalizar sensores y configuraciones
        parsed["mediciones_sensores"] = {f"s_{i}": parsed["mediciones_sensores"].get(f"s_{i}", 0) for i in range(1, 22)}
        settings = list(parsed["configuraciones_operativas"])
        while len(settings) < 3:
            settings.append(0)
        parsed["configuraciones_operativas"] = settings[:3]

        return json.dumps(parsed, ensure_ascii=False)

    except Exception as e:
        logger.exception("Error en extract_cmapss_tool: %s", e)
        return json.dumps({"error": "exception_in_tool", "detail": str(e)})


# Registrar la tool
ToolRegistry.register("extract_cmapss", extract_cmapss_tool)
