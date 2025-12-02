import json
import logging
from langchain_core.messages import AIMessage
from utils.tool_registry import ToolRegistry
from tools.tool_output_to_df import tool_output_to_df
from utils.llm_provider import paths_config
from utils.llm_provider import llm_creative
from utils.Predictor_RUL import predict_RUL
from langchain_core.prompts import ChatPromptTemplate

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
    2. Detecta degradación asociada a sensores clave del motor (adaptado a señales típicas de CMAPSS):
       - Temperatura / Compresor: s_3
       - Presión HPC: s_4
       - Vibraciones: s_7
       - Fan speed / núcleo: s_9
       - Fuel flow: s_14
       Observa si cada sensor está fuera de su rango normal respecto al promedio y desviación estándar (simular lógica).

    3. Traduce los patrones de degradación a modos de fallo probables:
       - compresor → "Degradación del compresor HPC (High Pressure Compressor)"
       - presión → "Ineficiencia en el sistema de presurización del núcleo"
       - vibración → "Desgaste mecánico – rodamientos o fan rotor"
       - rpm → "Pérdida de empuje – posible daño en fan blades"
       - combustible → "Baja eficiencia térmica – cámara de combustión degradada"

    4. Genera un texto explicativo al usuario incluyendo:
       - Estado de desgaste según RUL
       - Señales de degradación detectadas
       - Modos de fallo probables
       - Lenguaje claro, profesional y conciso

    Formato de salida:
    - Texto en lenguaje natural, no JSON.
    - No incluyas instrucciones ni código, solo el análisis.

    Datos de entrada:
    RUL: {predicted_RUL}
    Sensores: {sensor_values}
    """
)


def extract_cmapss_action(state):
    user_msg = state.messages[-1].content

    # Llamar al llm
    try:
        tool_out = ToolRegistry.invoke("extract_cmapss", user_msg)
    except Exception as e:
        logger.exception("Error invocando extract_cmapss tool: %s", e)
        return {"messages": [AIMessage(content=f"Error extrayendo datos: {e}")]}

    # parsear y validar
    try:
        parsed = json.loads(tool_out) if isinstance(tool_out, str) else tool_out
    except Exception as e:
        try:
            s = str(tool_out).replace("```json", "").replace("```", "")
            parsed = json.loads(s)
        except Exception as ex:
            logger.exception("No se pudo parsear salida de la tool: %s", ex)
            return {"messages": [AIMessage(content="No pude interpretar la información extraída del mensaje.")]}
        
    # Convertir a DataFrame
    try:
        df_user = tool_output_to_df(parsed)
    except Exception as e:
        logger.exception("Error convirtiendo a DataFrame: %s", e)
        return {"messages": [AIMessage(content="Error formateando las medidas enviadas por el usuario.")]}
    
    # Predicción RUL
    base_path = paths_config["paths"]["data_directory"]
    fd = parsed.get("modelo_seleccionado", "FD001")
    try:
        pred = predict_RUL(df_user, base_path, fd=fd)

        # Decisión de seguimiento
        if pred["predicted_RUL"] < 20:
            state.needs_followup = True
            state.next_agent = "Criticidad"
        else:
            state.needs_followup = False
            state.next_agent = None

    except Exception as e:
        logger.exception("Error en predict_RUL: %s", e)
        return {"messages": [AIMessage(content=f"Error al predecir RUL: {e}")]}

    # Generar texto salida
    sensor_values = df_user.to_dict(orient="records")[0]  # dict de sensores
    chain = PROMPT_RUL_RESPONSE | llm_creative
    
    rul_text = chain.invoke({
        "predicted_RUL": pred["predicted_RUL"],
        "sensor_values": sensor_values
    }).content.strip()
    
    #return {"messages": [AIMessage(content=rul_text)]}
    state.messages.append(AIMessage(content=rul_text))
    return state

