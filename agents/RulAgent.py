# agents/rul_agent.py
import json
import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_creative, paths_config
from utils.tool_registry import ToolRegistry
from tools.tool_output_to_df import tool_output_to_df
from utils.Predictor_RUL import predict_RUL

logger = logging.getLogger(__name__)

PROMPT_RUL_RESPONSE = ChatPromptTemplate.from_template(
    """
Eres un asistente experto en mantenimiento de motores aeronáuticos.
Se te entrega la siguiente información:
- Predicción RUL del motor: {predicted_RUL} ciclos
- Valores de sensores: {sensor_values}
Instrucciones:
1. Evalúa el nivel de desgaste del motor según RUL:
   - RUL > 80: "Desgaste bajo. Continuar operación normal."
   - RUL > 40: "Desgaste moderado. Programar inspección preventiva."
   - RUL > 20: "Desgaste significativo. Evaluar inspección avanzada."
   - RUL > 5 : "Riesgo elevado. Requiere monitorización constante."
   - RUL <=5 : "ALERTA CRÍTICA: Recomendada retirada inmediata del motor."
2. Detecta degradación asociada a sensores clave:
   - Temperatura / Compresor: s_3
   - Presión HPC: s_4
   - Vibraciones: s_7
   - Fan speed / núcleo: s_9
   - Fuel flow: s_14
3. Traduce patrones a modos de fallo probables (usa lenguaje claro y conciso).

Formato de salida: texto en lenguaje natural, profesional y conciso (NO JSON).
"""
)


def extract_cmapss_action(state):
    """
    Acción del agente RUL: toma state.pre_rul_data, lo convierte en DataFrame, predice RUL y genera explicación.
    """
    print(">>>RUL")
    try:
        if not hasattr(state, "pre_rul_data") or state.pre_rul_data is None:
            state.messages.append(AIMessage(content="No hay datos previos. Por favor, actualiza los datos antes de calcular."))
            state.needs_followup = True
            state.next_agent = "PreRUL"
            return state

        # Convertir pre_rul_data a DataFrame
        try:
            df_user = tool_output_to_df(state.pre_rul_data)
        except Exception as e:
            logger.exception("Error convirtiendo pre_rul_data a DataFrame: %s", e)
            state.messages.append(AIMessage(content=f"Error formateando las medidas: {e}"))
            state.needs_followup = True
            state.next_agent = "PreRUL"
            return state

        base_path = paths_config["paths"]["data_directory"]
        fd = state.pre_rul_data.get("modelo_seleccionado", "FD001")
        try:
            pred = predict_RUL(df_user, base_path, fd=fd)
        except Exception as e:
            logger.exception("Error en predict_RUL: %s", e)
            state.messages.append(AIMessage(content=f"Error al predecir RUL: {e}"))
            state.needs_followup = True
            state.next_agent = "PreRUL"
            return state

        predicted_RUL = pred.get("predicted_RUL", None)
        if predicted_RUL is None:
            state.messages.append(AIMessage(content="El modelo no devolvió una predicción válida."))
            state.needs_followup = True
            state.next_agent = "PreRUL"
            return state

        # Preparar sensor_values como dict simple para la explicación
        sensor_values = df_user.to_dict(orient="records")[0]

        # Generar texto explicativo final
        chain = PROMPT_RUL_RESPONSE | llm_creative
        rul_text = chain.invoke({
            "predicted_RUL": predicted_RUL,
            "sensor_values": sensor_values
        }).content.strip()

        # Añadir la salida al historial
        state.messages.append(AIMessage(content=rul_text))

        # Seguimiento: si RUL crítico, cambiar agente next
        try:
            if isinstance(predicted_RUL, (int, float)) and predicted_RUL < 20:
                state.needs_followup = True
                state.next_agent = "Criticidad"
            else:
                state.needs_followup = False
                state.next_agent = None
        except Exception:
            state.needs_followup = False
            state.next_agent = None

        return state

    except Exception as e:
        logger.exception("Error en extract_cmapss_action (RUL): %s", e)
        state.messages.append(AIMessage(content=f"Error interno del agente RUL: {e}"))
        state.needs_followup = True
        state.next_agent = "PreRUL"
        return state
