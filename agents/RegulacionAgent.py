# agents/RegulacionAgent.py
import logging
from langchain_core.messages import AIMessage
#from utils.llm_provider import llm_creativo

logger = logging.getLogger(__name__)

def regulacion_action(state):
    user_msg = state.messages[-1].content
    prompt = f"""
Usuario consulta normativa: {user_msg}

Proporciona normativa aplicable (FAA ACs, EASA CS, ACs):
- Menciona requisitos legales, certificación y regulaciones.
- Texto claro, resumido y referenciado.
"""
    # response = llm_creativo.invoke({"message": prompt})
    # return {"messages": [AIMessage(content=response.content)]}
    print(">>> CRITICIDAD")
    return {"messages": [AIMessage(content="Criticidad")]}
