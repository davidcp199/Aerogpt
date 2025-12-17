# agents/rul_agent.py
import json
import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_creative, paths_config
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

Formato de salida: texto profesional, sin JSON.
"""
)


def rul_action(state):
    """
    Acción del agente RUL:
    - Toma state.pre_rul_data (DataFrame)
    - Calcula RUL
    - Genera explicaciones
    """
    print(">>>RUL")
    state.source = "RUL"

    try:
        # 1. Comprobar que hay datos acumulados
        if state.pre_rul_data is None or len(state.pre_rul_data) == 0:
            state.messages.append(AIMessage(content="No hay datos para calcular el RUL. Añada datos de configuracion y sensores del motor primero."))
            state.needs_followup = True
            state.next_agent = "PreRUL"
            return state

        # 2. El dataframe ya está preparado por PreRUL
        df_user = state.pre_rul_data

        # 3. Obtener ruta y FD actual
        base_path = paths_config["paths"]["data_directory"]
        fd = state.modelo_seleccionado or "FD001"

        # 4. Ejecutar predictor
        try:
            pred = predict_RUL(df_user, base_path, fd=fd)
        except Exception as e:
            logger.exception("Error en predict_RUL: %s", e)
            state.messages.append(AIMessage(content=f"Error al calcular la RUL: {e}"))
            state.needs_followup = True
            state.next_agent = "PreRUL"
            return state

        predicted_RUL = pred.get("predicted_RUL", None)
        if predicted_RUL is None:
            state.messages.append(AIMessage(content="El modelo no devolvió una predicción válida."))
            state.needs_followup = True
            state.next_agent = "PreRUL"
            return state

        # 5. Tomar LA ÚLTIMA FILA (ciclo más reciente)
        sensor_values = df_user.iloc[-1].to_dict()

        # 6. Generar explicación final
        chain = PROMPT_RUL_RESPONSE | llm_creative
        text = chain.invoke({
            "predicted_RUL": predicted_RUL,
            "sensor_values": sensor_values
        }).content.strip()

        state.messages.append(AIMessage(content=text))
        state.update_memory("RUL", text)

        # 7. Si el motor está crítico, ir al agente Criticidad
        if isinstance(predicted_RUL, (int, float)) and predicted_RUL < 20:
            state.needs_followup = True
            state.next_agent = "Criticidad"
        else:
            state.needs_followup = False
            state.next_agent = None

        return state

    except Exception as e:
        logger.exception("Error en rul_action (RUL): %s", e)
        state.messages.append(AIMessage(content=f"Error interno del agente RUL: {e}"))
        state.needs_followup = True
        state.next_agent = "PreRUL"
        return state
