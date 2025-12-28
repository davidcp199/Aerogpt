import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from utils.llm_provider import llm_creative, llm_deterministic,  paths_config
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


PROMPT_TECNICO = ChatPromptTemplate.from_template(
    """
Eres un asistente experto en ingeniería y mantenimiento aeronáutico.

Tu función:
- Analizar la pregunta técnica del usuario.
- Usar EXCLUSIVAMENTE el contexto técnico proporcionado.
- Referenciar cuando sea posible documentos técnicos como: AMM, FIM, MEL, CDL, IPC, SRM, WDM, TSM, MPD, Aircraft Maintenance Manuals.
- Ser preciso, claro y profesional.
- Responder SIEMPRE en español.

REGLA CRÍTICA:
Si el contexto NO describe un fallo real, síntoma anómalo, mensaje ECAM/EICAS,
limitación MEL, o desviación de parámetros operativos, tu respuesta debe ser
EXCLUSIVAMENTE explicativa o informativa.

Si no hay indicios de fallo, deja explícitamente indicado:
"No se identifica ninguna condición de fallo ni necesidad de acción correctiva."

IMPORTANTE:
Una explicación técnica o descriptiva NO es crítica si no hay fallo real descrito.


NO debes:
- Diagnosticar fallos
- Proponer reparaciones
- Sugerir acciones correctivas
- Asumir sistemas afectados


CONTEXTO TÉCNICO:
{context}

PREGUNTA DEL USUARIO:
{question}

Contexto adicional de conversación:
{conversation_summary}

RESPUESTA:
"""
)

DECIDE_CRITICALITY = ChatPromptTemplate.from_template(
"""
Clasifica el siguiente análisis técnico como:
- CRITICO → si implica riesgo de seguridad o impacto operacional significativo
- NO_CRITICO → si es solo informativo o técnico

Análisis:
{analysis}

Responde SOLO con: CRITICO o NO_CRITICO
"""
)


def technical_action(state):
    """
    Agente técnico aeronáutico:
    - Recupera chunks desde el vectorstore técnico.
    - Genera análisis técnico basado en el contexto.
    - Añade respuesta al state.
    - Decide siguiente agente (si procede).
    """

    print(">>>TECNICO")
    logger.info(">>> TECNICO")
    state.source = "Tecnico"


    try:
        question = state.messages[-1].content
        if not question:
            state.messages.append(
                AIMessage(content="No se ha proporcionado ninguna pregunta técnica para analizar.")
            )
            state.needs_followup = False
            state.next_agent = None
            return state

        # Cargar vectorstore técnico
        try:
            vec_path = paths_config["paths"]["technical_vector_store"]
            embeddings = OpenAIEmbeddings()
            store = FAISS.load_local(vec_path, embeddings, allow_dangerous_deserialization=True)

        except Exception as e:
            logger.exception("Error cargando vectorstore técnico: %s", e)
            state.messages.append(AIMessage(content="Error cargando documentación técnica aeronáutica."))
            state.needs_followup = False
            state.next_agent = None
            return state

        # Chunks
        try:
            retrieved_docs = store.similarity_search(question, k=5)
            context = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])
        except Exception as e:
            logger.exception("Error en similarity_search (técnico): %s", e)
            state.messages.append(AIMessage(content="Error recuperando información técnica aeronáutica."))
            state.needs_followup = False
            state.next_agent = None
            return state


        chain = PROMPT_TECNICO | llm_creative

        try:
            response_text = chain.invoke({
                "context": context,
                "question": question,
                "conversation_summary": state.conversation_summary
            }).content.strip()

        except Exception as e:
            logger.exception("Error generando respuesta técnica: %s", e)
            state.messages.append(AIMessage(content="Error al generar el análisis técnico."))
            state.needs_followup = False
            state.next_agent = None
            return state

        state.messages.append(AIMessage(content=response_text))
        state.update_memory("Tecnico", response_text)
        

        # Decide si se envia a agente Criticidad
        decision = (DECIDE_CRITICALITY | llm_deterministic).invoke(
            {"analysis": response_text}
        ).content.strip()
        decision = decision.strip().upper()

        if "CRITICO" in decision:
            print(">>> Derivando a CRITICIDAD desde TÉCNICO")
            state.needs_followup = True
            state.next_agent = "Criticidad"
        else:
            state.needs_followup = False
            state.next_agent = None
        return state


    except Exception as e:
        logger.exception("Error interno en technical_action: %s", e)
        state.messages.append(AIMessage(content=f"Error interno en agente técnico: {e}"))
        state.needs_followup = False
        state.next_agent = None
        return state
