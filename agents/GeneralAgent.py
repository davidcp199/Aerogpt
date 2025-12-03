# agents/GeneralAgent.py
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_creative
from langchain_core.messages import AIMessage, HumanMessage
import logging
import re

logger = logging.getLogger(__name__)

PROMPT_GENERAL = ChatPromptTemplate.from_template(
"""
Eres un asistente experto en aviación y motores aeronáuticos, con capacidad de mantener contexto de la conversación.

Tu objetivo:
1. Responder preguntas generales del usuario sobre aviación, aeronáutica, motores, procedimientos, reglamentación o conceptos explicados previamente.
2. Mantener coherencia con la conversación anterior, usando el historial de mensajes.
3. Detectar si la pregunta requiere un agente especializado (RUL, Criticidad, Reparacion, Regulacion):
   - Si se trata de vida útil del motor o predicción de RUL → sugiere "RUL"
   - Si se trata de riesgos, seguridad o criticidad operacional → sugiere "Criticidad"
   - Si se trata de mantenimiento, reparación o troubleshooting → sugiere "Reparacion"
   - Si se trata de normativas, certificaciones o reglamentación → sugiere "Regulacion"
4. Si no se requiere derivación, responde directamente al usuario de forma clara, profesional y concisa.
5. Siempre responde en **lenguaje natural**, sin JSON ni instrucciones técnicas.
6. No inventes información; si no sabes, responde indicando que no tienes la información suficiente.
7. Si se trata de vida útil del motor o predicción de RUL, pero no se menciona minimo el estado de dos sensores, se le comunica esto al usuario pidiendo mas informacion en vez de derivar a RUL.

Historial de la conversación: 
{conversation_history}

Último mensaje del usuario: 
{user_message}

Instrucciones de salida:
- Si debes derivar a un agente especializado, solo devuelve **el nombre del agente**: RUL, Criticidad, Reparacion, Regulacion
- Si respondes directamente, devuelve el texto de respuesta.
- No mezcles ambas cosas.
"""
)

def general_action(state):
    """
    Analiza la pregunta general y decide si responder directamente o derivar a otro agente.
    Retorna dict con 'messages' y 'state'.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            print(f"Humano: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"IA: {msg.content}")
    try:
        # Construir historial completo
        conversation_history = "\n".join([
            f"Humano: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
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

        # Validar si se trata de derivación a otro agente
        if content in ["RUL", "Criticidad", "Reparacion", "Regulacion"]:
            state.next_agent = content
            print(f">>> GENERAL DERIVA A: {content}")
            return {"messages": state.messages, "state": state}

        # Sino, respuesta directa
        state.messages.append(AIMessage(content=content))
        return {"messages": state.messages, "state": state}

    except Exception as e:
        logger.exception("Error en GeneralAgent: %s", e)
        # Retorno fallback
        fallback_msg = "Lo siento, no pude procesar tu solicitud correctamente."
        state.messages.append(AIMessage(content=fallback_msg))
        return {"messages": state.messages, "state": state}