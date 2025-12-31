import logging
import json
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from utils.llm_provider import llm_creative, paths_config
from langchain_openai import OpenAIEmbeddings
from agents.State import AgentState

logger = logging.getLogger(__name__)

# ============================================================
# PROMPTS
# ============================================================

REGULATION_SCHEMA_DESCRIPTION = """
Devuelve EXCLUSIVAMENTE un objeto JSON con la siguiente estructura:

{{
  "applicability": ["CS-25", "FAR 25"],
  "aircraft_applicability": true | false,
  "dispatch_relevance": true | false,
  "regulations": [
    {{
      "authority": "EASA | FAA",
      "reference": "CS 25.1309",
      "topic": "System safety",
      "constraint": "Descripción clara del requisito normativo"
    }}
  ],
  "operational_limitations": [
    "Limitación operacional relevante"
  ],
  "compliance_risk": "LOW | MEDIUM | HIGH",
  "summary": "Resumen técnico en lenguaje profesional"
}}

NO añadas texto fuera del JSON.
"""


PROMPT_REGULACION = ChatPromptTemplate.from_template(
    f"""
Eres un asistente experto en regulación aeronáutica (FAA & EASA).

Tu función:
- Analizar la pregunta del usuario.
- Usar EXCLUSIVAMENTE el contexto proporcionado de normativa FAA/EASA.
- Citar la normativa cuando sea posible (AC, CFR, CS-XX, Part-M, Part-145…).
- Ser claro, preciso y profesional.
- Responder SIEMPRE en español.
- Genera la respuesta en formato JSON según este esquema:

{REGULATION_SCHEMA_DESCRIPTION}

CONTEXTO NORMATIVO:
{{context}}

PREGUNTA DEL USUARIO:
{{question}}

Contexto adicional de conversación:
{{conversation_summary}}

RESPUESTA:
"""
)


DECIDE_COMPLIANCE_IMPACT = ChatPromptTemplate.from_template(
"""
Eres un experto en normativa aeronáutica y safety operacional.

Tu tarea es clasificar si el análisis normativo implica CRITICIDAD OPERACIONAL REAL.

Clasificación:

CRITICO:
- Existe un INCUMPLIMIENTO explícito de la normativa
- Se describe una operación fuera de límites certificados
- Hay impacto directo o inmediato en la seguridad del vuelo
- Se menciona una condición MEL / CDL / Dispatch NO conforme
- Se describe una situación real, actual o inminente

SIN_CRITICIDAD:
- La consulta es puramente informativa, descriptiva o interpretativa
- Se explican límites normativos sin indicar que se han superado
- No hay operación real ni escenario de incumplimiento
- Es una pregunta teórica, formativa o de referencia
- No se menciona ningún evento, fallo o violación normativa

IMPORTANTE:
- Nunca clasifiques como CRITICO una consulta que solo pregunta del tipo "cuál es la regulación" o "qué dice la normativa".
- La existencia de límites de seguridad NO implica criticidad por sí sola.

REGLA CLAVE:
Si la consulta es informativa, descriptiva o académica, y NO describe una operación real,
evento, fallo, desviación o incumplimiento, DEBES:

- Establecer "aircraft_applicability": false
- Establecer "dispatch_relevance": false
- Establecer "compliance_risk": "LOW"
- Indicar claramente en el summary que NO existe impacto operacional ni necesidad de acción.

PROHIBIDO inferir incumplimientos, escenarios operacionales o violaciones normativas
si el usuario no los describe explícitamente.


Análisis normativo:
{analysis}

Responde SOLO con una palabra, sin explicaciones:

CRITICO
SIN_CRITICIDAD
"""
)


# ============================================================
# AGENT ACTION
# ============================================================

def regulation_action(state: AgentState) -> AgentState:
    """
    Agente de regulación FAA/EASA:
    - Recupera chunks desde el vectorstore de normativa.
    - Genera análisis normativo con llm_creative.
    - Añade respuesta al state.regulation.
    - Decide siguiente agente (Criticidad) si es crítico.
    """

    # print(">>> REGULACION")
    logger.info(">>> REGULACION")
    state.source = "Regulacion"
    state.emit("\n---> REGULACION AGENT", level="debug")

    try:
        # Obtener pregunta del usuario
        question = state.messages[-1].content
        if not question:
            state.messages.append(
                AIMessage(content="No se ha proporcionado ninguna pregunta para analizar regulación.")
            )
            state.emit("\nNo se ha proporcionado ninguna pregunta para analizar regulación.", level="user")
            return state

        # Cargar vectorstore regulatorio
        try:
            vec_path = paths_config["paths"]["regulatory_vector_store"]
            embeddings = OpenAIEmbeddings()
            store = FAISS.load_local(vec_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.exception("Error cargando vectorstore regulatorio: %s", e)
            state.messages.append(AIMessage(content="Error cargando normativa FAA/EASA."))
            state.emit("\nError cargando normativa FAA/EASA.", level="user")
            state.needs_followup = False
            state.next_agent = None
            return state

        # Recuperar documentos con MMR
        try:
            retrieved_docs = store.max_marginal_relevance_search(
                query=question,
                k=5,
                fetch_k=20,
                lambda_mult=0.7
            )
            context = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])
        except Exception as e:
            logger.exception("Error en similarity_search (regulación): %s", e)
            state.messages.append(AIMessage(content="Error recuperando información de normativa FAA/EASA."))
            state.emit("\nError recuperando información de normativa FAA/EASA.", level="user")
            state.needs_followup = False
            state.next_agent = None
            return state

        # Generar respuesta normativa
        try:
            chain = PROMPT_REGULACION | llm_creative
            raw_response = chain.invoke({
                "context": context,
                "question": question,
                "conversation_summary": getattr(state, "conversation_summary", "No hay histórico reciente.")
            })
            # Asegurarse de que raw_response es string
            if hasattr(raw_response, "content"):
                raw_response_str = raw_response.content
            else:
                raw_response_str = str(raw_response)

            regulation_data = json.loads(raw_response_str)
        except Exception as e:
            logger.exception("Error generando análisis normativo: %s", e)
            state.messages.append(AIMessage(content="Error generando análisis normativo."))
            state.emit("\nError generando análisis normativo.", level="user")
            state.needs_followup = False
            state.next_agent = None
            return state

        # Guardar respuesta en el estado
        state.regulation = regulation_data
        state.messages.append(AIMessage(content=f"Análisis normativo generado:\n{raw_response_str}"))
        state.emit(f"\nAnálisis normativo generado:\n{raw_response_str}", level="debug")
        

        # Decidir si es crítico o no
        try:
            decision = (DECIDE_COMPLIANCE_IMPACT | llm_creative).invoke(
                {"analysis": raw_response_str}
            )
            if hasattr(decision, "content"):
                decision_str = decision.content.strip().upper()
            else:
                decision_str = str(decision).strip().upper()

            if "CRITICO" in decision_str:
                state.needs_followup = True
                state.next_agent = "Criticidad"
            else:
                state.needs_followup = True
                state.next_agent = "Final"

        except Exception as e:
            logger.exception("Error clasificando criticidad normativa: %s", e)
            state.emit("\nError clasificando criticidad normativa.", level="user")
            state.needs_followup = False
            state.next_agent = None

        return state

    except Exception as e:
        logger.exception("Error interno en regulation_action: %s", e)
        state.messages.append(AIMessage(content=f"Error interno en agente de regulación: {e}"))
        state.emit(f"\nError interno en agente de regulación: {e}", level="user")
        state.needs_followup = False
        state.next_agent = None
        return state
