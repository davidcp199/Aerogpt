# agents/pre_rul.py
import json
import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_deterministic
from utils.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Prompt principal para decidir acción del usuario
PROMPT_PRE_RUL = ChatPromptTemplate.from_template(
    """
Eres un asistente experto en motores aeronáuticos y en CMAPSS.
Analiza el último mensaje del usuario y decide una sola palabra que indique la acción:
- "Update" → si el usuario quiere actualizar información del motor (sensores, configuraciones, unidad, ciclos).
- "Calculate" → si el usuario quiere calcular la RUL del motor.
- "Exit" → si el usuario quiere finalizar la sesión.
- "Chat" → si el mensaje es distinto a los anteriores.

Último mensaje del usuario:
{user_message}
"""
)

PROMPT_CHAT = ChatPromptTemplate.from_template(
    """
Eres un asistente experto en motores aeronáuticos y en el dataset CMAPSS para predicción de RUL.
Responde de forma clara, concisa y técnica a cualquier pregunta del usuario sobre motores, sensores, degradación o predicción de RUL.

Instrucciones:
- Mantén el contexto de CMAPSS y RUL aunque el usuario pregunte algo general.
- Ofrece explicaciones precisas y consejos de experto si es necesario.
- Nunca inventes valores de sensores ni datos de RUL.
- Al final de tu respuesta, recuerda al usuario las posibles acciones que puede hacer a continuación: "Update" para actualizar datos del motor, "Calculate" para calcular la RUL, o "Exit" para finalizar la sesión.

Mensaje del usuario:
{user_message}
"""
)


def pre_rul_action(state):
    """
    Gestiona la conversación con el usuario:
    - Identifica la acción: Update, Calculate, Exit o Status
    - En caso de Update, llama a extract_cmapss_tool y mergea los datos
    - En caso de Status, imprime el estado actual
    - Retorna estado actualizado
    """
    print("<<<PRERUL")
    try:
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

        # Preguntar al LLM qué acción tomar
        chain = PROMPT_PRE_RUL | llm_deterministic
        response = chain.invoke({"user_message": last_user_msg})
        action = response.content.strip()
        state.messages.append(AIMessage(content=action))

        action_lower = action.lower()
        print("--------")
        print(action_lower)

        if "update" in action_lower:
            # Llamar a la tool con argumentos correctos
            try:
                tool_response = ToolRegistry.invoke(
                    "extract_cmapss",
                    message=last_user_msg, 
                    pre_rul_data=dict(state.pre_rul_data)
                )
            except TypeError:
                # fallback
                tool_response = ToolRegistry.invoke(
                    "extract_cmapss",
                    message=last_user_msg,
                    pre_rul_data=state.pre_rul_data
                )


            # Parsear respuesta
            if isinstance(tool_response, dict):
                parsed = tool_response
            else:
                try:
                    parsed = json.loads(tool_response)
                except Exception as e:
                    logger.exception("No pude parsear la salida de la tool: %s", e)
                    state.messages.append(AIMessage(
                        content="Error: no pude interpretar la información extraída."))
                    state.needs_followup = False
                    state.next_agent = "PreRUL"
                    return state

            # Revisar errores de la tool
            if parsed.get("error"):
                state.messages.append(AIMessage(
                    content=f"Error en extractor: {parsed.get('error')}"))
                state.needs_followup = False
                state.next_agent = "PreRUL"
                return state

            # Merge seguro de datos
            state.pre_rul_data["unidad"] = int(parsed.get("unidad", state.pre_rul_data.get("unidad", 0)))
            state.pre_rul_data["tiempo_ciclos"] = int(parsed.get("tiempo_ciclos", state.pre_rul_data.get("tiempo_ciclos", 0)))

            # Configuraciones operativas
            new_settings = parsed.get("configuraciones_operativas", [0, 0, 0])
            if isinstance(new_settings, (list, tuple)):
                s = list(new_settings)[:3]
                while len(s) < 3:
                    s.append(0)
                state.pre_rul_data["configuraciones_operativas"] = s

            # Sensores: sobreescribir solo los mencionados
            new_sensors = parsed.get("mediciones_sensores", {})
            if isinstance(new_sensors, dict):
                for i in range(1, 22):
                    key = f"s_{i}"
                    if key in new_sensors:
                        state.pre_rul_data["mediciones_sensores"][key] = new_sensors.get(key, 0)

            # Modelo seleccionado
            model = parsed.get("modelo_seleccionado")
            if model:
                state.pre_rul_data["modelo_seleccionado"] = model

            state.messages.append(AIMessage(content="Datos actualizados correctamente. Diga Calcular RUL si quiere hacerlo con estos datos"))
            state.needs_followup = False
            state.next_agent = "PreRUL"
            return state

        elif "calculate" in action_lower:
            # Revisar si hay datos completos antes de calcular
            state.messages.append(AIMessage(content="Iniciando cálculo de RUL."))
            state.needs_followup = True
            state.next_agent = "RUL"
            return state

        elif "exit" in action_lower:
            state.messages.append(AIMessage(content="Finalizando sesión."))
            state.needs_followup = False
            state.next_agent = "END"
            return state

        elif "status" in action_lower:
            print("Estado actual del motor:")
            print(json.dumps(state.pre_rul_data, indent=2, ensure_ascii=False))
            state.messages.append(AIMessage(content="Se ha mostrado el estado actual del motor en consola."))
            state.needs_followup = False
            state.next_agent = "PreRUL"
            return state
        
        elif "chat" in action_lower:
            # Usar PROMPT_CHAT para generar respuesta experta
            chat_chain = PROMPT_CHAT | llm_deterministic
            chat_response = chat_chain.invoke({"user_message": last_user_msg})
            state.messages.append(AIMessage(content=chat_response.content))
            state.needs_followup = False
            state.next_agent = "PreRUL"
            return state

        else:
            # Clarify / repetir
            state.messages.append(AIMessage(
                content="No entendí la acción. Por favor responde 'Update', 'Calculate', 'Status' o 'Exit'."))
            state.needs_followup = False
            state.next_agent = "PreRUL"
            return state

    except Exception as e:
        logger.exception("Error en pre_rul_action: %s", e)
        state.messages.append(AIMessage(
            content="No pude procesar tu solicitud. Por favor indícame qué deseas hacer."))
        state.needs_followup = False
        state.next_agent = "PreRUL"
        return state
