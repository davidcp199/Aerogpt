import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from utils.llm_provider import llm_creative, paths_config
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

PROMPT_REGULACION = ChatPromptTemplate.from_template(
    """
Eres un asistente experto en regulación aeronáutica (FAA & EASA).

Tu función:
- Analizar la pregunta del usuario.
- Usar EXCLUSIVAMENTE el contexto proporcionado de normativa FAA/EASA.
- Citar la normativa cuando sea posible (AC, CFR, CS-XX, Part-M, Part-145…).
- Ser claro, preciso y profesional.
- Responder SIEMPRE en español.

CONTEXTO NORMATIVO:
{context}

PREGUNTA DEL USUARIO:
{question}

Contexto adicional de conversación:
{conversation_summary}

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

def regulation_action(state):
    """
    Agente de regulación FAA/EASA:
    - Recupera chunks desde el vectorstore de normativa.
    - Genera análisis normativo con llm_creative.
    - Añade respuesta al state.
    - Decide siguiente agente.
    """

    print(">>>REGULACION")
    logger.info(">>> REGULACION")
    state.source = "Regulacion"


    try:
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

        # Chunks
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

        chain = PROMPT_REGULACION | llm_creative

        try:
            response_text = chain.invoke({
                "context": context,
                "question": question,
                "conversation_summary": state.conversation_summary
            }).content.strip()
        except Exception as e:
            logger.exception("Error generando respuesta normativa: %s", e)
            state.messages.append(AIMessage(content="Error al generar la interpretación normativa."))
            state.needs_followup = False
            state.next_agent = None
            return state


        state.messages.append(AIMessage(content=response_text))
        state.update_memory("Regulacion", response_text)


        # Decidir si es crítico o no
        decision = (DECIDE_COMPLIANCE_IMPACT | llm_creative).invoke(
            {"analysis": response_text}
        ).content.strip()

        decision = decision.strip().upper()

        if "CRITICO" in decision:
            state.needs_followup = True
            state.next_agent = "Criticidad"
        else:
            state.needs_followup = False
            state.next_agent = None
        return state


    except Exception as e:
        logger.exception("Error interno en regulation_action: %s", e)
        state.messages.append(AIMessage(content=f"Error interno en agente de regulación: {e}"))
        state.needs_followup = False
        state.next_agent = None
        return state
