from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_creative
from langchain_core.messages import AIMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

# Variable global para controlar cuántas entradas recientes por agente incluir
AGENT_HISTORY_LIMIT = None  # None = todas, o un número entero para limitar

PROMPT_GENERAL = ChatPromptTemplate.from_template(
"""
Eres un asistente experto en aviación y motores aeronáuticos.

Tu objetivo:
1. Responder preguntas **generales** sobre aviación, aeronáutica, motores, procedimientos, reglamentación o conceptos explicados previamente.
2. Mantener coherencia con la conversación anterior, usando el historial de la conversación y los resultados recientes de los agentes especializados.
3. Siempre responde en lenguaje natural, profesional y conciso.
4. No derivar a ningún agente, no calcules RUL ni interpretes sensores.
5. No respondas nada fuera de tu ámbito de experto en aviación, si esto ocurre, di que no puedes ayudar y di las cosas que sí puedes hacer.
6. Si no tienes suficiente información, pide más detalles al usuario.
7. Puedes hacer resumenes de las conversaciones previas y resultados de los agentes
   además de poder tambien dar información específica de algún punto de la conversación previa.
8. Si los agentes especializados indican explícitamente que NO hay criticidad,
NO debes reinterpretar la situación como problemática.

Resumen de la conversación:
{conversation_summary}

Resultados recientes de los agentes (si existen):
{agents_history}

Último mensaje del usuario:
{user_message}

Devuelve únicamente el texto de respuesta.
"""
)

def general_action(state):
    """
    GeneralAgent ahora responde preguntas generales considerando:
    - conversation_summary completo
    - outputs recientes de los agentes, configurable mediante AGENT_HISTORY_LIMIT
    """
    state.source = "General"
    state.needs_followup = True
    state.next_agent = "Final"
    state.emit("\n---> GENERAL AGENT", level="debug")
    try:
        # Usar conversation_summary si existe, sino construir desde mensajes
        conversation_summary = getattr(state, "conversation_summary", "")
        if not conversation_summary:
            conversation_summary = "\n".join([
                f"Humano: {m.content}" if isinstance(m, HumanMessage) else f"IA: {m.content}"
                for m in state.messages
            ])

        # Construir resumen de outputs recientes de los agentes
        agents_history_list = []
        for agent, entries in state.history_by_agent.items():
            if entries:
                if AGENT_HISTORY_LIMIT is None:
                    selected_entries = entries
                else:
                    selected_entries = entries[-AGENT_HISTORY_LIMIT:]
                for entry in selected_entries:
                    agents_history_list.append(f"{agent}: {entry}")
        agents_history = "\n".join(agents_history_list) if agents_history_list else "Sin información reciente de agentes."

        user_msg = state.messages[-1].content

        chain = PROMPT_GENERAL | llm_creative
        response = chain.invoke({
            "conversation_summary": conversation_summary,
            "agents_history": agents_history,
            "user_message": user_msg
        })

        #print(f"entrando en GeneralAgent LLM con user_msg {user_msg}, agents_history: {agents_history}, conversation_summary: {conversation_summary}")

        content = response.content.strip()
        state.messages.append(AIMessage(content=content))
        state.emit(content, level="user")
        return state

    except Exception as e:
        logger.exception("Error en GeneralAgent: %s", e)
        fallback_msg = "Lo siento, no pude procesar tu solicitud correctamente."
        state.messages.append(AIMessage(content=fallback_msg))
        state.needs_followup = False
        state.next_agent = None
        return state
