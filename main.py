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

warnings.filterwarnings("ignore")

# ==============================================
# Configuración paths y logger
# ==============================================
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Cargar configs
model_cfg, paths_cfg, settings_cfg = load_all_configs(ROOT)

# Logger
logger = setup_logger("AeroGPT", level=settings_cfg["settings"].get("logging_level", "DEBUG"))
logger.info("Iniciando AeroGPT")

# ==============================================
# Construcción del grafo y estado inicial
# ==============================================
graph = GraphBuilder().build()

state = AgentState(
    messages=[],
    decision=None,
    next_agent=None,
    needs_followup=False,
    pre_rul_data=None
)

# ==============================================
# Funciones de impresión
# ==============================================
def print_regulation(state: AgentState):
    if hasattr(state, "regulation") and state.regulation:
        print("\n=== REGULACIÓN ===")
        print(f"Aplicabilidad: {state.regulation.get('applicability')}")
        print(f"Aeronave afectada: {state.regulation.get('aircraft_applicability')}")
        print(f"Dispatch relevante: {state.regulation.get('dispatch_relevance')}")
        print("Regulaciones citadas:")
        for reg in state.regulation.get("regulations", []):
            print(f" - {reg['authority']} {reg['reference']}: {reg['constraint']}")
        print(f"Limitaciones operacionales: {state.regulation.get('operational_limitations')}")
        print(f"Riesgo de cumplimiento: {state.regulation.get('compliance_risk')}")
        print(f"Resumen: {state.regulation.get('summary')}\n")


def print_criticidad(state: AgentState):
    if hasattr(state, "criticidad") and state.criticidad:
        print("\n=== CRITICIDAD ===")
        print(f"Sistema afectado: {state.criticidad.get('affected_system')}")
        print(f"Fase de vuelo: {state.criticidad.get('flight_phase')}")
        print(f"Severidad: {state.criticidad.get('severity')}")
        print(f"Riesgo operacional: {state.criticidad.get('operational_risk')}")
        print(f"Referencias: {state.criticidad.get('references')}")
        print(f"Recomendaciones: {state.criticidad.get('recommendations')}")
        print(f"Dispatch permitido: {state.dispatch_allowed}\n")


def print_ai_messages(state: AgentState):
    ia_msgs = [m for m in state.messages if isinstance(m, AIMessage)]
    for msg in ia_msgs:
        print(msg.content)
    # Limpiar mensajes de IA antiguos
    state.messages = [m for m in state.messages if isinstance(m, HumanMessage)]

# ==============================================
# Bucle principal
# ==============================================
def main_loop():
    global state  # <-- Declaración al inicio
    try:
        while True:
            user_input = input("Pregunta del usuario ('stop' para salir): ")
            if user_input.lower() == "stop":
                break

            state.messages.append(HumanMessage(content=user_input))

            # Ejecutar grafo
            result = graph.invoke(state)

            logger.debug(f"DEBUG result type: {type(result)}")

            # Normalizar: reconstruir AgentState si devuelve dict
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

            # Imprimir todo de forma organizada
            print_regulation(state)
            print_criticidad(state)
            print_ai_messages(state)

        print("AeroGPT terminado.")

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
