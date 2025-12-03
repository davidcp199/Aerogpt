# main.py
# CAMBIO: incializa logger, LLMs centralizados y grafos con agentes limpios
import os
import sys
from utils.config_loader import load_all_configs
from utils.logger import setup_logger
from utils.llm_provider import paths_config, settings_config  # inicializa LLMs por import
from agents.GraphBuilder import GraphBuilder
from langchain_core.messages import HumanMessage

import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Cargar configs (ya cacheadas por config_loader)
model_cfg, paths_cfg, settings_cfg = load_all_configs(ROOT)

# Logger
logger = setup_logger("AeroGPT", level=settings_cfg["settings"].get("logging_level", "DEBUG"))
logger.info("Iniciando AeroGPT")

# Construir grafo e iniciar loop
graph = GraphBuilder().build()

try:
    # Inicializar el estado global
    state = {
        "messages": [],
        "next_agent": None,
        "needs_followup": False
    }

    while True:
        user_input = input("Pregunta del usuario ('stop' para salir): ")

        if user_input.lower() == "stop":
            break

        # Guardar el mensaje del usuario
        state["messages"].append(HumanMessage(content=user_input))

        # Ejecutar el grafo con TODO el estado acumulado
        result = graph.invoke(state)

        # Guardar el nuevo estado actualizado
        state = result

        # Mostrar la última respuesta de la IA
        ia_msgs = [m for m in state["messages"] if m.__class__.__name__ == "AIMessage"]

        if ia_msgs:
            print("\n>>> RESPUESTA:")
            print(ia_msgs[-1].content)
        else:
            print("\n(No hubo respuesta de IA)")

    print("AeroGPT terminado.")

except KeyboardInterrupt:
    print("\nSaliendo...")
    logger.info("Interrupción por teclado, cerrando.")
