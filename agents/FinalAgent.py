import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from utils.llm_provider import llm_creative
from agents.State import AgentState

logger = logging.getLogger(__name__)

FINAL_AGENT_PROMPT = ChatPromptTemplate.from_template("""
Eres un asistente experto en aviación, mantenimiento y normativa aeronáutica.

Tu tarea:
1. Responder a la pregunta del usuario de manera clara, profesional y concisa.
2. Integrar información relevante de los agentes disponibles:
   - Regulación
   - Criticidad
   - Reparación
   - RUL
   - Técnico
3. Solo incluir información que sea **relevante para la pregunta del usuario**.
4. Diferencia claramente entre:
   - Observaciones técnicas o de estado actual
   - Recomendaciones prácticas o acciones preventivas
5. No incluyas información irrelevante o que no aplique al contexto de la pregunta.
6. Si no hay datos de reparación o criticidad relevantes, omítelos.
7. Siempre mantén un tono profesional, en español y accesible para tripulación y personal técnico.
8. Siempre que haya informacion de RUL, inclúyela si es relevante para la pregunta del usuario, idicando el RUL calculado al principio siempre.
9. No inventes información ni hagas suposiciones no basadas en los datos proporcionados, los únicos datos que puedes usar es la información de los agentes y tu la sintetizas.

Pregunta del usuario:
{user_question}

Información disponible de agentes:
Regulación: {regulation_info}
Criticidad: {criticality_info}
Reparación: {repair_info}
RUL: {rul_info}

Genera la respuesta final en lenguaje natural, estructurada y coherente.
""")


def final_action(state):
    """
    Acción del agente Final:
    - Integra información de otros agentes
    - Responde a la pregunta del usuario
    """
    print(">>>FINAL")
    logger.info(">>> FINAL AGENT")
    state.source = "Final"

    try:
        user_question = state.messages[-1].content

        regulation_info = getattr(state, "regulation", "No disponible")
        criticality_info = getattr(state, "criticidad", "No disponible")
        repair_info = getattr(state, "reparacion", "No disponible")
        rul_state = getattr(state, "rul", None)
        if isinstance(rul_state, dict):
            rul_info = f"RUL estimado: {rul_state.get('predicted_RUL', 'No disponible')} ciclos. {rul_state.get('text', '')}"
        else:
            rul_info = rul_state or "No disponible"


        chain = FINAL_AGENT_PROMPT | llm_creative
        response = chain.invoke({
            "user_question": user_question,
            "regulation_info": regulation_info,
            "criticality_info": criticality_info,
            "repair_info": repair_info,
            "rul_info": rul_info
        })

        print("==============================Entrando en FinalAgent, respuesta generada.=========================================")
        print(f"regulation_info: {regulation_info}")
        print(f"criticality_info: {criticality_info}")
        print(f"repair_info: {repair_info}")
        print(f"rul_info: {rul_info}")
        print("==============================SALIENDO en FinalAgent, respuesta generada.=========================================")


        state.messages.append(AIMessage(content=response.content.strip()))
        state.needs_followup = False
        state.next_agent = None
        return state


    except Exception as e:
        logger.exception("Error interno en FinalAgent: %s", e)
        state.messages.append(AIMessage(content=f"Error generando respuesta final: {e}"))
        state.needs_followup = False
        state.next_agent = None
        return state
