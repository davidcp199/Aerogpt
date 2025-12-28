import json
import logging
import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_deterministic
from utils.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# --- Prompt para decidir acción ---
PROMPT_PRE_RUL = ChatPromptTemplate.from_template(
    """
Eres un asistente experto en motores aeronáuticos y en el dataset CMAPSS para predicción de RUL.

Tu tarea es analizar el último mensaje del usuario y decidir **una sola acción prioritaria** de acuerdo con las siguientes reglas:

1. **Update** → Si el usuario menciona **nuevos valores de sensores, configuraciones, unidad, ciclos o motor**, incluso si también menciona calcular el RUL.  
2. **Calculate** → Solo si el usuario quiere calcular el RUL explícitamente **y no hay nuevos datos a registrar**.  
3. **Status** → Si el usuario quiere ver el historial de mediciones acumuladas o los datos del motor hasta ahora.
4. **Reset** → Si quiere reiniciar los datos o indicar un nuevo motor.
5. **Exit** → Si el usuario quiere finalizar la sesión.
6. **Chat** → Para mensajes generales o preguntas técnicas sobre motores, degradación o RUL que no impliquen ninguna acción de registro o cálculo.

**Importante**:
- Prioridad: si hay datos de sensor mencionados, aunque ponga calcular el RUL, se debe devolver "Update" o "Reset", el que corresponda, Nunca "Calculate" para estos casos.
- Devuelve SOLO una palabra EXACTA: "Update", "Calculate", "Status", "Reset", "Exit", "Chat".
- No agregues explicaciones ni texto adicional.

Último mensaje del usuario:
{user_message}

"""
)


# --- Prompt para chat ---
PROMPT_CHAT = ChatPromptTemplate.from_template(
"""
Eres un asistente experto en motores aeronáuticos y en el dataset CMAPSS para predicción de RUL (Remaining Useful Life).

Instrucciones generales:
- Responde de forma clara, concisa y técnica a cualquier pregunta sobre motores, sensores, degradación o predicción de RUL.
- Nunca inventes valores de sensores ni datos de RUL.
- Mantén el contexto de CMAPSS y RUL aunque la pregunta sea general.
- Al final de tu respuesta, recuerda al usuario las posibles acciones que puede hacer a continuación: 
  "Update" para actualizar datos, "Calculate" para calcular la RUL, o "Exit" para finalizar la sesión.

Información sobre la predicción de RUL:
- El RUL se calcula a partir de una **secuencia de datos históricos fila por fila**.
- Cada fila contiene:
    * 3 configuraciones operativas del motor.
    * 21 mediciones de sensores etiquetadas como s_1, s_2, ..., s_21.
    * Identificador de unidad y tiempo de ciclos de operación.
- Para obtener predicciones precisas, se recomienda tener **al menos 30 filas acumuladas** de datos por motor.
- Si hay menos de 30 filas, se puede generar un historial sintético, pero esto reduce la precisión de la predicción.

Mensaje del usuario:
{user_message}
"""
)


def pre_rul_action(state):
    """
    Gestiona la conversación con el usuario:
    - Identifica acción: Update, Calculate, Exit o Chat
    - Update: extrae nuevos valores y añade al DataFrame
    - Calculate: va al cálculo de RUL si hay datos
    """
    print("<<<PRERUL")
    state.source = "PreRUL"
    
    try:
        last_user_msg = state.messages[-1].content if state.messages else ""

        # Inicializar DataFrame si no existe
        if state.pre_rul_data is None:
            state.pre_rul_data = pd.DataFrame()
            print("Inicializando DataFrame vacío para medidas CMAPSS...")

        # LLM Accion
        chain = PROMPT_PRE_RUL | llm_deterministic
        response = chain.invoke({
         "user_message": last_user_msg
        })
        action = response.content.strip()

        action_lower = action.lower()
        print("--------")
        print(action_lower)

        if "update" in action_lower:
            tool_response = ToolRegistry.invoke("extract_cmapss", message=last_user_msg)
            parsed = json.loads(tool_response)

            if parsed.get("error"):
                state.messages.append(AIMessage(content=f"Error: {parsed['error']}"))
                state.needs_followup = False
                return state

            # Crear fila con valores
            fila = {
                "unidad": parsed.get("unidad", 0),
                "tiempo_ciclos": parsed.get("tiempo_ciclos", 0),
                "setting_1": parsed.get("configuraciones_operativas", [0,0,0])[0],
                "setting_2": parsed.get("configuraciones_operativas", [0,0,0])[1],
                "setting_3": parsed.get("configuraciones_operativas", [0,0,0])[2],
            }

            for i in range(1, 22):
                fila[f"s_{i}"] = parsed.get("mediciones_sensores", {}).get(f"s_{i}", 0)

            # Añadir al DataFrame
            state.pre_rul_data = pd.concat([state.pre_rul_data, pd.DataFrame([fila])], ignore_index=True)

            # Actualizar modelo seleccionado
            state.modelo_seleccionado = parsed.get("modelo_seleccionado", state.modelo_seleccionado)

            state.messages.append(AIMessage(content=f"Nueva medición registrada. ({len(state.pre_rul_data)} filas acumuladas). ¿Quiere Calcular RUL ahora?"))
            state.update_memory("PreRUL", f"Nueva medición registrada ({len(state.pre_rul_data)} filas)")
            state.needs_followup = False
            state.next_agent = None
            return state

        elif "calculate" in action_lower:
            if state.pre_rul_data is None or len(state.pre_rul_data) == 0:
                state.messages.append(AIMessage(content="No hay datos suficientes para calcular el RUL. Primero use 'Update'."))
                return state

            state.messages.append(AIMessage(content="Calculando RUL con histórico actual..."))
            
            state.needs_followup = True
            state.next_agent = "RUL"
            return state

        elif "status" in action_lower:
            if state.pre_rul_data is None or state.pre_rul_data.empty:
                state.messages.append(AIMessage(content="No hay datos registrados todavía."))
            else:
                df_str = state.pre_rul_data.to_string(index=False)
                state.messages.append(AIMessage(content=f"Estado actual de mediciones:\n{df_str}"))
            state.needs_followup = False
            state.next_agent = None
            return state

        elif "reset" in action_lower:
            # Extraer info nueva si hay
            if "nuevo motor" in last_user_msg.lower() or "nuevo" in last_user_msg.lower() or "update" in last_user_msg.lower():
                tool_response = ToolRegistry.invoke("extract_cmapss", message=last_user_msg)
                parsed = json.loads(tool_response)
                fila = {
                    "unidad": parsed.get("unidad", 0),
                    "tiempo_ciclos": parsed.get("tiempo_ciclos", 0),
                    "setting_1": parsed.get("configuraciones_operativas", [0,0,0])[0],
                    "setting_2": parsed.get("configuraciones_operativas", [0,0,0])[1],
                    "setting_3": parsed.get("configuraciones_operativas", [0,0,0])[2],
                }
                for i in range(1, 22):
                    fila[f"s_{i}"] = parsed.get("mediciones_sensores", {}).get(f"s_{i}", 0)

                state.pre_rul_data = pd.DataFrame([fila])
                state.modelo_seleccionado = parsed.get("modelo_seleccionado", "FD001")
                state.messages.append(AIMessage(content="Datos reseteados y nueva medición registrada."))
            else:
                # solo reset
                state.pre_rul_data = pd.DataFrame()
                state.modelo_seleccionado = "FD001"
                state.messages.append(AIMessage(content="Datos reseteados a cero."))
                state.update_memory("PreRUL", "Datos reseteados a cero")
            state.needs_followup = False
            state.next_agent = None
            return state

        elif "chat" in action_lower:
            chat_chain = PROMPT_CHAT | llm_deterministic
            chat_response = chat_chain.invoke({"user_message": last_user_msg})
            state.messages.append(AIMessage(content=chat_response.content))
            state.update_memory("PreRUL", chat_response.content)
            state.needs_followup = False
            state.next_agent = None
            return state

        elif "exit" in action_lower:
            state.messages.append(AIMessage(content="Cerrando sesión."))
            state.update_memory("PreRUL", "Cerrando sesión")
            state.needs_followup = False
            state.next_agent = None
            return state

        else:
            state.messages.append(AIMessage(content="No entendí la acción. Responde 'Update', 'Calculate', 'Status' o 'Exit'."))
            state.needs_followup = False
            state.next_agent = None
            return state

    except Exception as e:
        logger.exception("Error en pre_rul_action: %s", e)
        state.messages.append(AIMessage(content="No pude procesar tu solicitud. Por favor indícame qué deseas hacer."))
        state.needs_followup = False
        state.next_agent = None
        return state