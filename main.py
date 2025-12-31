import os
import sys
import warnings
import logging

from utils.config_loader import load_all_configs
from utils.logger import setup_logger
from utils.llm_provider import paths_config, settings_config
from agents.GraphBuilder import GraphBuilder
from langchain_core.messages import HumanMessage, AIMessage
from agents.State import AgentState
from tools import extract_cmapss

warnings.filterwarnings("ignore")

# ==============================================
# Configuración paths y logger
# ==============================================
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

model_cfg, paths_cfg, settings_cfg = load_all_configs(ROOT)
logger = setup_logger(
    "AeroGPT",
    level=settings_cfg["settings"].get("logging_level", "DEBUG")
)
logger.info("Iniciando AeroGPT")

# ==============================================
# Construcción del grafo y estado inicial
# ==============================================
graph = GraphBuilder().build()
state = AgentState(messages=[])

# ==============================================
# Memoria histórica avanzada por agente
# ==============================================
if not hasattr(state, "history_by_agent"):
    state.history_by_agent = {
        "Regulacion": [],
        "Criticidad": [],
        "Reparacion": [],
        "Tecnico": [],
        "RUL": [],
        "PreRUL": [],
        "General": [],
        "Final": []
    }

# ==============================================
# Reset selectivo de estado por iteración
# ==============================================
def reset_state_iteration(state: AgentState):
    """
    Resetea solo los atributos de agentes por iteración
    para evitar contaminación de la pregunta anterior.
    No se toca pre_rul_data para que los datos de RUL persistan.
    """
    agent_attrs = [
        "regulation",
        "criticidad",
        "reparacion",
        "tecnico",
        "dispatch_allowed",
        "needs_followup",
        "next_agent",
        "source"
    ]

    for attr in agent_attrs:
        if attr in ["dispatch_allowed", "needs_followup"]:
            setattr(state, attr, False)
        elif attr == "next_agent":
            setattr(state, attr, None)
        else:
            setattr(state, attr, None)

# ==============================================
# Guardar resultados en historial por agente
# ==============================================

FIELD_AGENT_MAP = {
    "regulation": "Regulacion",
    "criticidad": "Criticidad",
    "reparacion": "Reparacion",
    "pre_rul_data": "PreRUL",
    "rul": "RUL",
    "tecnico": "Tecnico",
    "final_response": "Final",
    "general_notes": "General"
}

def save_history(state: AgentState):
    for field, agent in FIELD_AGENT_MAP.items():
        val = getattr(state, field, None)
        if val is None:
            continue

        history = state.history_by_agent.setdefault(agent, [])

        entry = {field: val}
        if entry not in history:
            history.append(entry)



# ==============================================
# Impresión SOLO de la respuesta final
# ==============================================
def print_final_response(state: AgentState):
    """
    Imprime únicamente la respuesta consolidada del FinalAgent.
    """
    ia_msgs = [m for m in state.messages if isinstance(m, AIMessage)]
    if ia_msgs:
        print("\n=== RESPUESTA ===\n")
        print(ia_msgs[-1].content)

# ==============================================
# Detección de nuevo caso
# ==============================================
def is_new_case(user_text: str) -> bool:
    triggers = [
        "nuevo motor",
        "nuevo avión",
        "otro motor",
        "otro avión",
        "caso nuevo",
        "equipo distinto",
        "empezar de cero",
        "motor diferente",
        "avión diferente"
    ]
    text = user_text.lower()
    return any(t in text for t in triggers)


# ==============================================
# Bucle principal
# ==============================================
def main_loop():
    global state
    try:
        while True:
            user_input = input("Pregunta del usuario ('stop' para salir): ")
            if user_input.lower() == "stop":
                break

            # Comandos de depuración
            if user_input.lower() == "show history":
                print("\n=== HISTORIAL POR AGENTE ===")
                for agent, entries in state.history_by_agent.items():
                    print(f"\n--- {agent} ---")
                    if entries:
                        for i, entry in enumerate(entries):
                            print(f"{i+1}: {entry}")
                    else:
                        print("Sin entradas")
                continue

            if user_input.lower() == "show conversation":
                print("\n=== RESUMEN DE CONVERSACIÓN ===")
                print(state.conversation_summary or "Sin resumen aún")
                continue

            # --- RESET DE CONTEXTO POR NUEVO CASO ---
            if is_new_case(user_input):
                print(">>> Nuevo caso detectado. Limpiando estado técnico.")
                logger.info("Nuevo caso detectado. Limpiando estado técnico.")

                state.rul = None
                state.criticidad = None
                state.reparacion = None
                state.regulation = None

            # Reset de estado por iteración
            reset_state_iteration(state)

            # Añadir mensaje humano
            state.messages.append(HumanMessage(content=user_input))

            # Ejecutar grafo
            result = graph.invoke(state)

            # Normalizar AgentState
            if isinstance(result, AgentState):
                state = result
            elif isinstance(result, dict):
                try:
                    state = AgentState(**result)
                except Exception as e:
                    logger.exception(
                        "No se pudo reconstruir AgentState desde dict: %s", e
                    )
                    continue
            else:
                logger.warning(
                    "El grafo devolvió un tipo inesperado, usando state previo"
                )

            # Guardar historial
            save_history(state)

            # Imprimir SOLO respuesta final
            print_final_response(state)

            # Limpiar mensajes AI (mantener solo humanos)
            state.messages = [
                m for m in state.messages if isinstance(m, HumanMessage)
            ]

        print("\nAeroGPT terminado.")

    except KeyboardInterrupt:
        print("\nSaliendo...")
        logger.info("Interrupción por teclado, cerrando.")
    except Exception as e:
        logger.exception("Error inesperado en main.py: %s", e)

# ==============================================
# Entrada
# ==============================================
if __name__ == "__main__":
    main_loop()
