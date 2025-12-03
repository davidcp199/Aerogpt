import logging
from langchain_core.messages import AIMessage
#from utils.llm_provider import llm_creativo

logger = logging.getLogger(__name__)

def criticidad_action(state):
    user_msg = state.messages[-1].content
    prompt = f"""
Evalúa el nivel de criticidad de la situación o fallo descrito por el usuario:

Usuario: {user_msg}

Considera: ASRS, FAA SDR y riesgos de seguridad.
Devuelve un texto con nivel de severidad y recomendaciones de seguridad.
"""
    # response = llm_creativo.invoke({"message": prompt})
    # return {"messages": [AIMessage(content=response.content)]}
    state.needs_followup = False
    state.next_agent = None
    print(">>> CRITICIDAD")
    #return {"messages": [AIMessage(content=output_text)], "state": state}
    return {"messages": state.messages, "state": state}
    #return {"messages": [AIMessage(content="Criticidad")]}

# if riesgo_severidad > 3:
#     state.needs_followup = True
#     state.next_agent = "Reparacion"
