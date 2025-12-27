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
Clasifica el siguiente análisis normativo:

- CRITICO → si existe incumplimiento normativo con impacto potencial en seguridad operacional
- NO_CRITICO → si es solo informativo o interpretativo

Análisis:
{analysis}

Responde SOLO con: CRITICO o NO_CRITICO
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

    print(">>> REGULACION")
    logger.info(">>> REGULACION")
    state.source = "Regulacion"

    try:
        # Obtener pregunta del usuario
        question = state.messages[-1].content
        if not question:
            state.messages.append(
                AIMessage(content="No se ha proporcionado ninguna pregunta para analizar regulación.")
            )
            state.needs_followup = False
            state.next_agent = None
            return state

        # Cargar vectorstore regulatorio
        try:
            vec_path = paths_config["paths"]["regulatory_vector_store"]
            embeddings = OpenAIEmbeddings()
            store = FAISS.load_local(vec_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.exception("Error cargando vectorstore regulatorio: %s", e)
            state.messages.append(AIMessage(content="Error cargando normativa FAA/EASA."))
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
            state.needs_followup = False
            state.next_agent = None
            return state

        # Generar respuesta normativa
        try:
            chain = PROMPT_REGULACION | llm_creative
            raw_response = chain.invoke({
                "context": context,
                "question": question,
                "conversation_summary": state.conversation_summary
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
            state.needs_followup = False
            state.next_agent = None
            return state

        # Guardar respuesta en el estado
        state.regulation = regulation_data
        

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
                state.needs_followup = False
                state.next_agent = None

        except Exception as e:
            logger.exception("Error clasificando criticidad normativa: %s", e)
            state.needs_followup = False
            state.next_agent = None

        return state

    except Exception as e:
        logger.exception("Error interno en regulation_action: %s", e)
        state.messages.append(AIMessage(content=f"Error interno en agente de regulación: {e}"))
        state.needs_followup = False
        state.next_agent = None
        return state
