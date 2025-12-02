# agents/SupervisorAgent.py
# CAMBIO: usa llm centralizado y devuelve messages explícitos
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_deterministic
import logging

logger = logging.getLogger(__name__)

PROMPT_SUPERVISOR = ChatPromptTemplate.from_template(
    """
Eres un supervisor cuyo objetivo es decidir si un mensaje del usuario está relacionado con:

*Extracción de datos sobre motores aeronáuticos (CMAPSS)*  
*Predicción de RUL*  
*Sensores aeronáuticos FI, HPC, LPT, etc.*

Reglas:
- Si el mensaje del usuario **está relacionado con CMAPSS, motores, sensores o RUL**, responde **solo** con:
    extract
- Si el mensaje **NO está relacionado**, responde **solo** con:
    none
- No agregues explicaciones, frases adicionales ni símbolos.

Mensaje del usuario: {user_message}

Tu respuesta: (extract/none)
"""
)

def supervisor_action(state):
    chain = PROMPT_SUPERVISOR | llm_deterministic
    response = chain.invoke({"user_message": state.messages[-1].content})
    decision = response.content.strip()
    if decision not in ("extract", "none"):
        # CAMBIO: normalización por seguridad
        logger.warning("Supervisor devolvió valor inesperado '%s' — forzando 'none'", decision)
        decision = "none"

    logger.debug("Supervisor decisión: %s", decision)
    # CAMBIO: devolvemos mensajes explícitos vacíos para evitar comportamiento ambiguo
    return {"decision": decision, "messages": []}
