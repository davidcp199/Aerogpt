# agents/ReparacionAgent.py
import logging
from langchain_core.messages import AIMessage
#from utils.llm_provider import llm_creativo

logger = logging.getLogger(__name__)

def reparacion_action(state):
    user_msg = state.messages[-1].content
    prompt = f"""
Usuario describe un fallo: {user_msg}

Genera instrucciones de reparación o mantenimiento correctivo:
- Basado en FAST, ACs, EASA y ASRS.
- Incluye pasos de troubleshooting, precauciones y referencias.
- Devuelve texto claro y accionable.
"""
    # response = llm_creativo.invoke({"message": prompt})
    # return {"messages": [AIMessage(content=response.content)]}
    print(">>> Reparacion")
    return {"messages": [AIMessage(content="Reparacion")]}


# if accion_requiere_normativa:
#     state.needs_followup = True
#     state.next_agent = "Regulacion"
