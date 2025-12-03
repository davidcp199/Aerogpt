from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_creative
from langchain_core.messages import AIMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

PROMPT_GENERAL = ChatPromptTemplate.from_template(
"""
Eres un asistente experto en aviación y motores aeronáuticos.

Tu objetivo:
1. Responder preguntas **generales** sobre aviación, aeronáutica, motores, procedimientos, reglamentación o conceptos explicados previamente.
2. Mantener coherencia con la conversación anterior, usando el historial de mensajes.
3. Siempre responde en lenguaje natural, profesional y conciso.
4. No derivar a ningún agente, no calcules RUL ni interpretes sensores.

Historial de la conversación: 
{conversation_history}

Último mensaje del usuario: 
{user_message}

Devuelve únicamente el texto de respuesta.
"""
)

def general_action(state):
    """
    GeneralAgent ahora solo responde preguntas generales.
    No devuelve ningún agente.
    Retorna dict con 'messages' y 'state'.
    """
    try:
        # Construir historial completo
        conversation_history = "\n".join([
            f"Humano: {m.content}" if isinstance(m, HumanMessage) else f"IA: {m.content}"
            for m in state.messages
        ])

        user_msg = state.messages[-1].content

        # Llamada al LLM
        chain = PROMPT_GENERAL | llm_creative
        response = chain.invoke({
            "conversation_history": conversation_history,
            "user_message": user_msg
        })

        content = response.content.strip()

        # Añadir la respuesta al historial
        state.messages.append(AIMessage(content=content))

        # Solo devuelve el historial actualizado
        return {"messages": state.messages, "state": state}

    except Exception as e:
        logger.exception("Error en GeneralAgent: %s", e)
        fallback_msg = "Lo siento, no pude procesar tu solicitud correctamente."
        state.messages.append(AIMessage(content=fallback_msg))
        return {"messages": state.messages, "state": state}
