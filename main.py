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
    user_input = input("\nPregunta del usuario ('stop' para salir): ")
    while user_input != "stop":
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

        # Impresión robusta del resultado (el formato depende del nodo final)
        try:
            last_msg = result["messages"][-1]
            print("\n>>> RESPUESTA:")
            print(last_msg.content)
        except Exception:
            # Si StateGraph devuelve otra estructura
            print("\n>>> Resultado bruto:")
            print(result)

        user_input = input("\nPregunta del usuario ('stop' para salir): ")
except KeyboardInterrupt:
    print("\nSaliendo...")
    logger.info("Interrupción por teclado, cerrando.")
