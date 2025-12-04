# main.py
import os
import sys
import warnings
from utils.config_loader import load_all_configs
from utils.logger import setup_logger
from utils.llm_provider import paths_config, settings_config  # inicializa LLMs por import
from agents.GraphBuilder import GraphBuilder
from langchain_core.messages import HumanMessage, AIMessage
from agents.State import AgentState

warnings.filterwarnings("ignore")

# Añadir ROOT al path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Cargar configs
model_cfg, paths_cfg, settings_cfg = load_all_configs(ROOT)

# Logger
logger = setup_logger("AeroGPT", level=settings_cfg["settings"].get("logging_level", "DEBUG"))
logger.info("Iniciando AeroGPT")

# Construir grafo e iniciar loop
graph = GraphBuilder().build()

# Inicializar estado global
state = AgentState(
    messages=[],
    decision=None,
    next_agent=None,
    needs_followup=False,
    pre_rul_data=None
)

try:
    while True:
        user_input = input("Pregunta del usuario ('stop' para salir): ")
        if user_input.lower() == "stop":
            break

        # Añadir mensaje del usuario al state
        state.messages.append(HumanMessage(content=user_input))

        # Ejecutar grafo
        result = graph.invoke(state)

        # DEBUG: tipo de result
        logger.debug(f"DEBUG result type: {type(result)}")
        #logger.debug(f"DEBUG result content: {result}")

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

        # Leer solo los mensajes de IA nuevos
        ia_msgs = [m for m in state.messages if isinstance(m, AIMessage)]
        for msg in ia_msgs:
            print(msg.content)

        # Limpiar mensajes de IA antiguos para no imprimirlos de nuevo
        state.messages = [m for m in state.messages if isinstance(m, HumanMessage)]

    print("AeroGPT terminado.")

except KeyboardInterrupt:
    print("\nSaliendo...")
    logger.info("Interrupción por teclado, cerrando.")
except Exception as e:
    logger.exception("Error inesperado en main.py: %s", e)
