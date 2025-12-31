# ============================================================
# CriticidadAgent.py
# ============================================================

from pathlib import Path
from typing import List
import json
import logging

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage

from utils.llm_provider import llm_deterministic, paths_config
from agents.State import AgentState

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(paths_config["paths"]["base"])

VECTORSTORES = {
    "ASRS": BASE_DIR / "data" / "vectorStores" / "asrs_store",
    "SDR": BASE_DIR / "data" / "vectorStores" / "sdr_store",
    "REGULATORY": BASE_DIR / "data" / "vectorStores" / "regulatory_store",
    "TECHNICAL": BASE_DIR / "data" / "vectorStores" / "technical_store",
}

EMBEDDINGS = OpenAIEmbeddings()
K = 5

# ============================================================
# VECTORSTORE LOADERS
# ============================================================

def load_store(path: Path) -> FAISS | None:
    if not path.exists():
        return None
    try:
        return FAISS.load_local(
            path,
            EMBEDDINGS,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.exception("Error cargando FAISS store: %s", e)
        return None

STORES = {
    name: store
    for name, path in VECTORSTORES.items()
    if (store := load_store(path)) is not None
}

# ============================================================
# RETRIEVAL + FILTRADO POR METADATA
# ============================================================

def retrieve_context(question: str, system: str = None, flight_phase: str = None, far_part: str = None) -> List[Document]:
    if not isinstance(question, str):
        raise TypeError("La pregunta debe ser una cadena")

    retrieved: List[Document] = []

    for store_name, store in STORES.items():
        docs = store.max_marginal_relevance_search(
            query=question,
            k=K,
            fetch_k=K*4,
            lambda_mult=0.7
        )
        # Filtrado por metadata si aplica
        if system or flight_phase or far_part:
            docs = [
                d for d in docs
                if (not system or d.metadata.get("system") == system)
                and (not flight_phase or d.metadata.get("flight_phase") == flight_phase)
                and (not far_part or d.metadata.get("far_part") == far_part)
            ]
        retrieved.extend(docs)

    return retrieved

def build_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "UNKNOWN")
        blocks.append(f"[{source}]\n{d.page_content}")
    return "\n\n".join(blocks)

# ============================================================
# PROMPT EN ESPAÑOL
# ============================================================

CRITICIDAD_PROMPT = """
Eres un ingeniero experto en seguridad y confiabilidad aeronáutica.

Usando la evidencia proporcionada (reportes operacionales, SDR/ASRS,
documentación técnica y normativa), evalúa la criticidad operacional
de la condición descrita.

Debes:
- Identificar el sistema afectado y la fase de vuelo
- Evaluar la severidad: LOW / MEDIUM / HIGH / CRITICAL
- Explicar riesgo operacional y propagación de fallos
- Referenciar evidencia histórica cuando aplique
- Evitar especulación fuera de los datos proporcionados

Pregunta:
{question}

Evidencia:
{context}

Información adicional:
- Histórico reciente de mensajes: {history}
- Estado de RUL: {rul_info}
- Información de Regulación: {regulation_info}
- Información Técnica: {tecnico_info}

Proporciona un análisis estructurado en JSON con las siguientes claves:
- affected_system
- flight_phase
- severity
- operational_risk
- references
- recommendations
"""

# ============================================================
# AGENTE CRITICIDAD
# ============================================================

def criticidad_action(state: AgentState) -> AgentState:
    # print(">>> Ejecutando acción CRITICIDAD")
    state.emit("\n---> CRITICIDAD AGENT", level="debug")
    logger.info(">>> CRITICIDAD AGENT")
    state.source = "Criticidad"

    try:
        question = state.messages[-1].content
        if not question:
            state.messages.append(
                AIMessage(content="No se ha proporcionado ninguna pregunta técnica para analizar.")
            )
            state.emit("\nNo se ha proporcionado ninguna pregunta técnica para analizar.", level="user")
            state.needs_followup = False
            state.next_agent = None
            return state

        # Recuperar contexto
        docs = retrieve_context(question)
        context = build_context(docs)

        # Histórico reciente
        history = "\n".join(
            [f"{m.content}" for m in state.messages[-5:] if isinstance(m, AIMessage)]
        ) or "No hay histórico reciente."

        # Información de RUL
        rul_info = ""
        if isinstance(state.rul, dict):
            rul_info = f"RUL estimado: {state.rul.get('predicted_RUL', 'No disponible')} ciclos. {state.rul.get('text', '')}"
        elif isinstance(state.rul, str):
            rul_info = state.rul
        else:
            rul_info = "No hay información de RUL."

        # Información de Regulación y Técnico
        regulation_info = getattr(state, "regulation", "No disponible")
        tecnico_info = getattr(state, "tecnico", "No disponible")

        # Construir prompt
        prompt = CRITICIDAD_PROMPT.format(
            question=question,
            context=context,
            history=history,
            rul_info=rul_info,
            regulation_info=regulation_info,
            tecnico_info=tecnico_info
        )

        # Llamar al LLM determinista
        response = llm_deterministic.invoke(prompt)
        criticidad_data = response.content

        # Parsear JSON
        try:
            criticidad_json = json.loads(criticidad_data)
        except Exception as e:
            state.messages.append(AIMessage(content=f"Error parseando JSON de criticidad: {e}"))
            state.emit(f"\nError parseando JSON de criticidad: {e}", level="user")    
            state.needs_followup = False
            state.next_agent = None
            return state

        # Guardar directamente en el estado
        state.criticidad = criticidad_json
        state.emit(f"\nAnálisis de criticidad generado: {json.dumps(criticidad_json, ensure_ascii=False)}", level="debug")

        # Asignar dispatch_allowed según severidad
        severity = criticidad_json.get("severity", "").upper()
        state.dispatch_allowed = severity in ["LOW", "MEDIUM"]

        # Actualizar memoria
        state.update_memory("Criticidad", json.dumps(criticidad_json, ensure_ascii=False))

        # Decidir siguiente agente
        if severity in ["HIGH", "CRITICAL"]:
            # print(">>> Criticidad alta detectada, se requiere seguimiento.")
            state.emit("\nCriticidad alta detectada, se requiere seguimiento.", level="debug")
            state.needs_followup = True
            state.next_agent = "Reparacion"
        else:
            state.needs_followup = True
            state.next_agent = "Final"

        # print(f"Análisis criticidad generado. Severidad: {severity}. Dispatch permitido: {state.dispatch_allowed}")
        state.emit(f"\nAnálisis criticidad generado. Severidad: {severity}. Dispatch permitido: {state.dispatch_allowed}", level="debug")  

        return state

    except Exception as e:
        import traceback
        traceback.print_exc()
        state.messages.append(AIMessage(content=f"Error interno en criticidad_action: {e}"))
        state.emit(f"\nError interno en criticidad_action: {e}", level="user")
        state.needs_followup = False
        state.next_agent = None
        return state
