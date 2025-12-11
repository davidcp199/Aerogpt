# agents/regulation_agent.py
import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from utils.llm_provider import llm_creative, paths_config
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ============================================================
#                     PROMPT DEL AGENTE
# ============================================================
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

RESPUESTA:
"""
)

# ============================================================
#                     ACCIÓN DEL AGENTE
# ============================================================
def regulation_action(state):
    """
    Agente de regulación FAA/EASA:
    - Recupera chunks desde el vectorstore de normativa.
    - Genera análisis normativo con llm_creative.
    - Añade respuesta al state.
    - Decide siguiente agente.
    """

    print(">>>REGULACION")

    try:
        question = state.messages[-1].content
        if not question:
            state.messages.append(
                AIMessage(content="No se ha proporcionado ninguna pregunta para analizar regulación.")
            )
            state.needs_followup = False
            state.next_agent = None
            return state

        # ----------------------------------------------------
        # 1) Cargar vectorstore regulatorio
        # ----------------------------------------------------
        # 1) Cargar vectorstore regulatorio
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

        # ----------------------------------------------------
        # 2) Recuperar contexto (chunks)
        # ----------------------------------------------------
        try:
            retrieved_docs = store.similarity_search(question, k=5)
            context = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])
        except Exception as e:
            logger.exception("Error en similarity_search (regulación): %s", e)
            state.messages.append(AIMessage(content="Error recuperando información de normativa FAA/EASA."))
            state.needs_followup = False
            state.next_agent = None
            return state

        # ----------------------------------------------------
        # 3) Ejecutar LLM con el prompt oficial
        # ----------------------------------------------------
        chain = PROMPT_REGULACION | llm_creative

        try:
            response_text = chain.invoke({
                "context": context,
                "question": question
            }).content.strip()
        except Exception as e:
            logger.exception("Error generando respuesta normativa: %s", e)
            state.messages.append(AIMessage(content="Error al generar la interpretación normativa."))
            state.needs_followup = False
            state.next_agent = None
            return state

        # ----------------------------------------------------
        # 4) Añadir respuesta al estado
        # ----------------------------------------------------
        state.messages.append(AIMessage(content=response_text))

        # ----------------------------------------------------
        # 5) Lógica de encaminamiento
        # ----------------------------------------------------
        # Si menciona incumplimiento → enviar a Criticidad
        if "no conforme" in response_text.lower() or "incumplimiento" in response_text.lower():
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
