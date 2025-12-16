from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_deterministic
import logging

logger = logging.getLogger(__name__)

PROMPT_SUPERVISOR = ChatPromptTemplate.from_template(
"""
Eres un supervisor que decide qué agente debe responder según el mensaje del usuario.
Tienes seis posibles decisiones: PreRUL, Criticidad, Reparacion, Regulacion, Tecnico, General.

Agentes disponibles:

1. **PreRUL (Vida útil del motor / Predicción de RUL)**  
   - Consultas sobre la vida útil restante de un motor aeronáutico.  
   - Cuando el usuario menciona valores de sensores, configuraciones operativas, ciclos de operación o FD001/FD002/FD003/FD004.  
   - Cuando quiere calcular o menciona RUL, prever degradación, o analizar el estado actual de un motor usando datos CMAPSS.
   - Cuando menciona actualizar datos del motor, sensores o configuraciones o quiere saber su estado.

2. Criticidad (Riesgos / Safety)
- Consultas EXPLÍCITAS sobre seguridad operacional, análisis de riesgo, severidad o probabilidad de fallo.
- Referencias directas a incidentes, accidentes, ASRS, FAA SDR, ECCAIRS o eventos de seguridad.
- Evaluaciones de safety ya contextualizadas (p. ej., “riesgo inaceptable”, “impacto en la seguridad del vuelo”).
- NO usar Criticidad si el usuario está pidiendo primero explicación técnica del sistema.


3. **Reparacion (Mantenimiento / Troubleshooting)**  
   - Consultas sobre cómo reparar un motor o componente, instrucciones de mantenimiento, procedimientos correctivos.  
   - Basado en FAST, ACs, EASA o ASRS.  

4. **Regulacion (Legal / Certificación)**  
   - Consultas sobre normativas, certificaciones, reglamentación aeronáutica.  
   - Referencias a FAA ACs, EASA CS, regulaciones legales.

5. Tecnico (Ingeniería aeronáutica / Documentación técnica / Sistemas / Datos)
- PRIORIDAD ALTA:
  Si la pregunta menciona ECAM, EICAS, un sistema concreto (BLEED, FUEL, ELEC, HYD, PRESS),
  condiciones de disparo, consecuencias técnicas, lógica de funcionamiento o degradaciones,
  DEBE asignarse primero a Tecnico, incluso si la pregunta menciona consecuencias o efectos operativos.
   - Consultas que requieran análisis técnico profundo y especializado.
   - Preguntas sobre Airbus, Boeing, lógica de sistemas, señales, sensores, control, arquitectura, configuraciones, performances.
   - Preguntas sobre ECAM/EICAS, lógica de alertas, condiciones de disparo, modos de operación, ATA Chapters.
   - Uso de documentación técnica: AMM, FIM, MEL, CDL, MPD, IPC, SRM, WDM, TSM, ACM+, manuales eléctricos y estructurales.
   - Análisis de sistemas electrónicos, eléctricos, hidráulicos, neumáticos, combustible, bleed, presurización, aviónica, comunicaciones.
   - Consultas técnicas relacionadas con:
       * Datos técnicos Airbus y presentaciones de ingeniería.  
       * Gestión y mejora de datos técnicos: airnavX, plataformas digitales, actualización de datos en tiempo real.  
       * Documentación técnica en general: diagnóstico, troubleshooting, procedimientos avanzados.  
       * Sistemas electrónicos y electromagnéticos (ELISE, ASERIS).  
       * Modelado, simulación y análisis de sistemas complejos.  
       * Gestión de partes, configuración, documentación digital (ACTUAL).  
       * Formación técnica avanzada (ENAC, software, herramientas).  
       * Innovación tecnológica, electromagnetismo y desarrollos avanzados.
   - En general, cualquier pregunta que use lenguaje técnico aeronáutico o que requiera interpretación de documentación técnica.

6. **General**  
   - Preguntas generales o superficiales sobre aviación.  
   - Consultas que no requieran ingeniería, normativa, mantenimiento, safety o RUL.
   - Preguntas contextuales simples o de cultura aeronáutica básica.

**Reglas generales**:
- Debes elegir exactamente uno de los seis agentes: PreRUL, Criticidad, Reparacion, Regulacion, Tecnico o General.  
- No añadas explicaciones ni comentarios adicionales.  
- No inventes categorías.

Último mensaje de IA (si existe): {last_ai_message}
Mensaje del usuario: {user_message}

- Si una pregunta puede responderse mediante análisis técnico del sistema antes de evaluar riesgos,
  selecciona Tecnico y NO Criticidad.
- Criticidad solo debe usarse cuando el enfoque principal sea safety y no la ingeniería del sistema.

Solo devuelve una palabra, sin explicaciones:

Opciones válidas:
- PreRUL
- Criticidad
- Reparacion
- Regulacion
- Tecnico
- General
"""
)



from langchain_core.messages import AIMessage

def supervisor_action(state):
    """
    Decide qué agente ejecutar según el mensaje del usuario.
    Devuelve siempre un dict con 'messages' y 'state'.
    """
    # Si ya hay un agente definido por el flujo anterior, lo usamos
    if hasattr(state, "next_agent") and state.next_agent:
      agent = state.next_agent
      state.next_agent = None
      return state


    # Si no hay next_agent, consultar LLM para decidir
    user_msg = state.messages[-1].content
    last_ai_msg = None
    for m in reversed(state.messages):
      if isinstance(m, AIMessage):
         last_ai_msg = m.content
         break

    try:
        chain = PROMPT_SUPERVISOR | llm_deterministic
        response = chain.invoke({
         "user_message": user_msg,
         "last_ai_message": last_ai_msg or ""
        })

        agent = response.content.strip()

        # Validar que sea uno de los agentes conocidos
        VALID_AGENTS = ["PreRUL", "Criticidad", "Reparacion", "Regulacion", "Tecnico", "General"]

        if agent not in VALID_AGENTS:
         logger.warning("Agente inválido devuelto por supervisor: %s", agent)
         agent = "General"
         state.source = "supervisor"


        # Guardar el agente elegido en el state para que GraphBuilder lo use
        state.next_agent = agent
        state.source = "supervisor"

    except Exception as e:
        logger.exception("Error en supervisor LLM: %s", e)
        state.next_agent = "none"
        agent = "none"
   
    state.decision = agent  # guarda la decisión interna
    return state
