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
import tools.extract_cmapss

warnings.filterwarnings("ignore")

# ==============================================
# Configuración paths y logger
# ==============================================
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

model_cfg, paths_cfg, settings_cfg = load_all_configs(ROOT)
logger = setup_logger("AeroGPT", level=settings_cfg["settings"].get("logging_level", "DEBUG"))
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
        "General": []
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
    agent_attrs = ["regulation", "criticidad", "reparacion",
                   "dispatch_allowed", "needs_followup", "next_agent", "source"]
    
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
def save_history(state: AgentState):
    agent = state.source
    if agent:
        if agent not in state.history_by_agent:
            state.history_by_agent[agent] = []
        snapshot = {}
        for attr in ["regulation", "criticidad", "reparacion", "pre_rul_data"]:
            val = getattr(state, attr, None)
            if val is not None:
                snapshot[attr] = val
        if snapshot:
            state.history_by_agent[agent].append(snapshot)


# ==============================================
# Funciones de impresión
# ==============================================
def print_agent_results(state: AgentState):
    """
    Imprime resultados de la iteración actual y los mensajes AI.
    """
    if state.regulation:
        print("\n=== REGULACIÓN ===")
        print(f"Aplicabilidad: {state.regulation.get('applicability')}")
        print(f"Aeronave afectada: {state.regulation.get('aircraft_applicability')}")
        print(f"Dispatch relevante: {state.regulation.get('dispatch_relevance')}")
        for reg in state.regulation.get("regulations", []):
            print(f" - {reg['authority']} {reg['reference']}: {reg['constraint']}")
        print(f"Limitaciones operacionales: {state.regulation.get('operational_limitations')}")
        print(f"Riesgo de cumplimiento: {state.regulation.get('compliance_risk')}")
        print(f"Resumen: {state.regulation.get('summary')}\n")

    if state.criticidad:
        print("\n=== CRITICIDAD ===")
        print(f"Sistema afectado: {state.criticidad.get('affected_system')}")
        print(f"Fase de vuelo: {state.criticidad.get('flight_phase')}")
        print(f"Severidad: {state.criticidad.get('severity')}")
        print(f"Riesgo operacional: {state.criticidad.get('operational_risk')}")
        print(f"Referencias: {state.criticidad.get('references')}")
        print(f"Recomendaciones: {state.criticidad.get('recommendations')}")
        print(f"Dispatch permitido: {state.dispatch_allowed}\n")

    if state.reparacion:
        print("\n=== REPARACIÓN ===")
        print(f"Sistema afectado: {state.reparacion.get('system_affected')}")
        print(f"Fase de vuelo: {state.reparacion.get('flight_phase')}")
        print(f"Severidad: {state.reparacion.get('severity')}")
        print("Acciones recomendadas:")
        for action in state.reparacion.get('recommended_actions', []):
            print(f" - {action}")
        print("Referencias:")
        for ref in state.reparacion.get('references', []):
            print(f" - {ref}")
        notes = state.reparacion.get('notes')
        if notes:
            print(f"Notas: {notes}\n")

    # Mensajes AI
    ia_msgs = [m for m in state.messages if isinstance(m, AIMessage)]
    if ia_msgs:
        print("\n=== Mensajes AI ===")
        for msg in ia_msgs:
            print(msg.content)
    # Limpiar solo mensajes AI de la iteración actual
    state.messages = [m for m in state.messages if isinstance(m, HumanMessage)]

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

            # Comandos especiales de depuración
            if user_input.lower() == "show history":
                print("\n=== HISTORIAL POR AGENTE ===")
                for agent, entries in state.history_by_agent.items():
                    print(f"\n--- {agent} ---")
                    if entries:
                        for i, entry in enumerate(entries):
                            print(f"{i+1}: {entry}")
                    else:
                        print("Sin entradas")
                continue  # no enviar al grafo

            if user_input.lower() == "show conversation":
                print("\n=== RESUMEN DE CONVERSACIÓN ===")
                print(state.conversation_summary or "Sin resumen aún")
                continue  # no enviar al grafo

            reset_state_iteration(state)
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
                    logger.exception("No se pudo reconstruir AgentState desde dict: %s", e)
                    continue
            else:
                logger.warning("El grafo devolvió un tipo inesperado, usando state previo")

            # Guardar en historial
            save_history(state)

            # Imprimir resultados
            print_agent_results(state)

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
