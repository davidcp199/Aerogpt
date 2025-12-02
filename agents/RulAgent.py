# agents/RulAgent.py
# CAMBIO: usa ToolRegistry.invoke, utilidades no-tool y validación de errores
import json
import logging
from langchain_core.messages import AIMessage
from utils.tool_registry import ToolRegistry
from tools.tool_output_to_df import tool_output_to_df  # util (no @tool)
from utils.llm_provider import paths_config
from utils.Predictor_RUL import predict_RUL  # asumo signature predict_RUL(df, base_path, fd)

logger = logging.getLogger(__name__)

def extract_cmapss_action(state):
    user_msg = state.messages[-1].content

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
    # Predicción RUL con manejo de errores
    base_path = paths_config["paths"]["data_directory"]
    fd = parsed.get("modelo_seleccionado", "FD001")
    try:
        pred = predict_RUL(df_user, base_path, fd=fd)
    except Exception as e:
        logger.exception("Error en predict_RUL: %s", e)
        return {"messages": [AIMessage(content=f"Error al predecir RUL: {e}")]}

    logger.debug("RUL predicho: %s", pred)
    return {"messages": [AIMessage(content=json.dumps(pred))]}
