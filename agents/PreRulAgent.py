import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_deterministic
from utils.tool_registry import ToolRegistry
import json

logger = logging.getLogger(__name__)

PROMPT_PRE_RUL = ChatPromptTemplate.from_template(
    """
Eres un asistente que gestiona los datos previos a calcular la RUL de un motor aeronáutico.
Tu tarea es:

1. Revisar la información que ya se ha extraído del usuario sobre el motor:
   - unidad
   - ciclo operativo
   - configuraciones operativas
   - valores de sensores
2. Presentar estos datos al usuario de forma clara.
3. Preguntar si desea calcular el RUL ahora, actualizar algún dato concreto, o hacer otra cosa.
4. Responde siempre en lenguaje natural, profesional y conciso.
5. NO extraigas sensores ni calcules RUL aquí; solo gestiona el flujo y el historial.

Datos actuales del motor:
{pre_rul_data}

Último mensaje del usuario:
{user_message}

Genera una respuesta adecuada indicando los pasos a seguir.
Siempre que presentes los valores presentes tiene que ser con todos estos datos rellenos con lo que corresponde:
  "unidad": 0,
  "tiempo_ciclos": 0,
  "configuraciones_operativas": [0,0,0],
  "mediciones_sensores": {
        "s_1": 0, "s_2": 0, "s_3": 0, "s_4": 0, "s_5": 0,
        "s_6": 0, "s_7": 0, "s_8": 0, "s_9": 0, "s_10": 0,
        "s_11": 0, "s_12": 0, "s_13": 0, "s_14": 0, "s_15": 0,
        "s_16": 0, "s_17": 0, "s_18": 0, "s_19": 0, "s_20": 0, "s_21": 0
  },
  "modelo_seleccionado": "FD001"
}
"""
)

def pre_rul_action(state):
    """
    Gestiona el historial y la presentación de datos previos al cálculo de RUL.
    Actualiza state.pre_rul_data con los valores que el usuario quiera cambiar.
    """
    last_user_msg = state.messages[-1].content if state.messages else ""

    # Inicializar pre_rul_data si no existe
    if not hasattr(state, "pre_rul_data") or state.pre_rul_data is None:
        state.pre_rul_data = {
            "unidad": 0,
            "tiempo_ciclos": 0,
            "configuraciones_operativas": [0, 0, 0],
            "mediciones_sensores": {f"s_{i}": 0 for i in range(1, 22)},
            "modelo_seleccionado": "FD001"
        }

    # --- Actualización de datos con extract_cmapss_tool ---
    try:
        updated_data_raw = ToolRegistry.invoke("extract_cmapss", last_user_msg)
        updated_data = json.loads(updated_data_raw) if isinstance(updated_data_raw, str) else updated_data_raw
    except Exception as e:
        logger.exception("Error procesando actualización de datos: %s", e)
        updated_data = {}

    # Actualizamos solo los valores presentes en updated_data
    if updated_data:
        pre_data = state.pre_rul_data
        # unidad
        if "unidad" in updated_data and updated_data["unidad"] != 0:
            pre_data["unidad"] = updated_data["unidad"]
        # tiempo_ciclos
        if "tiempo_ciclos" in updated_data and updated_data["tiempo_ciclos"] != 0:
            pre_data["tiempo_ciclos"] = updated_data["tiempo_ciclos"]
        # configuraciones operativas
        if "configuraciones_operativas" in updated_data:
            for i, val in enumerate(updated_data["configuraciones_operativas"]):
                if val != 0:
                    pre_data["configuraciones_operativas"][i] = val
        # sensores
        if "mediciones_sensores" in updated_data:
            for sensor, val in updated_data["mediciones_sensores"].items():
                if val != 0:
                    pre_data["mediciones_sensores"][sensor] = val
        # modelo seleccionado
        if "modelo_seleccionado" in updated_data:
            pre_data["modelo_seleccionado"] = updated_data["modelo_seleccionado"]

    # --- Preparar resumen para el LLM ---
    pre_rul_text = (
        f"Unidad: {state.pre_rul_data['unidad']}\n"
        f"Ciclos: {state.pre_rul_data['tiempo_ciclos']}\n"
        f"Configuraciones: {state.pre_rul_data['configuraciones_operativas']}\n"
        f"Sensores presentes: {sum(1 for v in state.pre_rul_data['mediciones_sensores'].values() if v != 0)} de 21\n"
    )

    # --- Llamada a LLM para generar respuesta ---
    try:
        chain = PROMPT_PRE_RUL | llm_deterministic
        response = chain.invoke({
            "pre_rul_data": pre_rul_text,
            "user_message": last_user_msg
        })
        content = response.content.strip()
        state.messages.append(AIMessage(content=content))

        # Decidir siguiente acción
        user_lower = last_user_msg.lower()
        if any(word in user_lower for word in ["sí", "calcular"]):
            state.needs_followup = True
            state.next_agent = "RUL"
        elif any(word in user_lower for word in ["no", "otra cosa"]):
            state.needs_followup = True
            state.next_agent = "General"
        else:
            state.needs_followup = True
            state.next_agent = "PreRUL"  # Continuar gestionando datos

    except Exception as e:
        logger.exception("Error en PreRULAgent: %s", e)
        fallback_msg = "No pude procesar tu solicitud, por favor vuelve a indicarme qué deseas hacer."
        state.messages.append(AIMessage(content=fallback_msg))
        state.needs_followup = True
        state.next_agent = "PreRUL"

    return {"messages": state.messages, "state": state}
