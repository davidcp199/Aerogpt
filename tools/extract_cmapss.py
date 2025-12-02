from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_deterministic
from langchain_core.tools import tool
from utils.tool_registry import ToolRegistry
import logging

logger = logging.getLogger(__name__)

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

1. unidad  
   - Identificador del motor (si no se menciona → 0)

2. tiempo_ciclos  
   - Ciclo operativo actual (si no se menciona → 0)

3. configuraciones_operativas  
   - Tres valores: setting_1, setting_2, setting_3  
   - Si no se menciona alguno → 0

4. mediciones_sensores  
   - Lista EXACTA de 21 sensores (s_1 a s_21)  
   - Si se menciona un sensor, asigna ese valor; si no → 0

SELECCIÓN DEL MODELO (FD):
- FD001 → condiciones de nivel del mar + solo HPC degradation
- FD002 → condiciones SEIS + solo HPC
- FD003 → nivel del mar + HPC y/o Fan degradation
- FD004 → condiciones SEIS + HPC y/o Fan degradation

Si no hay contexto suficiente → usar "FD001"

Complete el siguiente JSON EXACTAMENTE, no añada texto fuera del JSON, ni comas finales inválidas:

{{
  "unidad": 0,
  "tiempo_ciclos": 0,
  "configuraciones_operativas": [0,0,0],
  "mediciones_sensores": {{
        "s_1": 0, "s_2": 0, "s_3": 0, "s_4": 0, "s_5": 0,
        "s_6": 0, "s_7": 0, "s_8": 0, "s_9": 0, "s_10": 0,
        "s_11": 0, "s_12": 0, "s_13": 0, "s_14": 0, "s_15": 0,
        "s_16": 0, "s_17": 0, "s_18": 0, "s_19": 0, "s_20": 0, "s_21": 0
  }},
  "modelo_seleccionado": "FD001"
}}

MENSAJE DEL USUARIO:
{message}
"""
)

@tool
def extract_cmapss_tool(message: str) -> str:
    """Extrae datos CMAPSS desde un texto del usuario usando el LLM."""
    try:
        chain = PROMPT_EXTRACT_CMAPSS | llm_deterministic
        print(">>> LLM que se usará en esta chain:", type(llm_deterministic), getattr(llm_deterministic, "model_name", None))
        
        response = chain.invoke({"message": message})
        tool_output = response.content.strip()
        # Quitar backticks si existen
        tool_output = tool_output.replace("```json", "").replace("```", "").strip()
        print(f"TOOL: {tool_output}")
        return tool_output
    except Exception as e:
        logger.exception("Error en extract_cmapss_tool: %s", e)
        return '{"error": "No se pudo procesar el mensaje"}'

# Registrar en ToolRegistry
ToolRegistry.register("extract_cmapss", extract_cmapss_tool)
