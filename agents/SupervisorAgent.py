from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_deterministic
import logging

logger = logging.getLogger(__name__)

PROMPT_SUPERVISOR = ChatPromptTemplate.from_template(
"""
Eres un supervisor que decide qué agente debe responder según el mensaje del usuario.
Tienes cinco posibles decisiones: RUL, Criticidad, Reparacion, Regulacion, none.

Agentes disponibles:

1. **RUL (Vida útil del motor / Predicción de RUL)**  
   - Consultas sobre la vida útil restante de un motor aeronáutico.  
   - Cuando el usuario menciona valores de sensores, configuraciones operativas, ciclos de operación o FD001/FD002/FD003/FD004.  
   - Cuando quiere calcular RUL, prever degradación, o analizar el estado actual de un motor usando datos CMAPSS.  

2. **Criticidad (Riesgos / Safety)**  
   - Consultas sobre riesgo de fallo, gravedad de fallos, seguridad operacional.  
   - Uso de bases de datos ASRS, FAA SDR o referencias a incidentes.  

3. **Reparacion (Mantenimiento / Troubleshooting)**  
   - Consultas sobre cómo reparar un motor o componente, instrucciones de mantenimiento, procedimientos correctivos.  
   - Basado en FAST, ACs, EASA o ASRS.  

4. **Regulacion (Legal / Certificación)**  
   - Consultas sobre normativas, certificaciones, reglamentación aeronáutica.  
   - Referencias a FAA ACs, EASA CS, regulaciones legales.  

5. **none**  
   - Si el mensaje del usuario no aplica a ninguno de los casos anteriores.  

**Reglas generales**:
- Debes elegir **exactamente uno** de los cinco agentes: RUL, Criticidad, Reparacion, Regulacion o none.  
- No añadas explicaciones ni comentarios adicionales.  
- No inventes categorías.  

Mensaje del usuario: {user_message}

Solo devuelve **una palabra**, sin explicaciones: 

Opciones válidas:
- RUL
- Criticidad
- Reparacion
- Regulacion
- none

"""
)

from langchain_core.messages import AIMessage

def supervisor_action(state):
    """
    Decide qué agente ejecutar según el mensaje del usuario.
    """


    chain = PROMPT_SUPERVISOR | llm_deterministic
    response = chain.invoke({"user_message": state.messages[-1].content})
    decision = response.content.strip()
    # if decision not in ("extract", "none"):
    #     logger.warning("Supervisor devolvió valor inesperado '%s' — forzando 'none'", decision)
    #     decision = "none"



    print(">>> SUPERVISOR DECIDE:", decision)
    logger.debug("Supervisor decisión: %s", decision)


    return {"messages": [], "state": state}

