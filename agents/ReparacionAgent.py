import json
import logging
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage

from utils.llm_provider import llm_creative, paths_config
from agents.State import AgentState

logger = logging.getLogger(__name__)

# ============================================================
# Configuración paths y vectorstores
# ============================================================
BASE_DIR = Path(paths_config["paths"]["base"])

VECTORSTORES = {
    "ASRS": BASE_DIR / "data" / "vectorStores" / "asrs_store",
    "SDR": BASE_DIR / "data" / "vectorStores" / "sdr_store",
    "REGULATORY": BASE_DIR / "data" / "vectorStores" / "regulatory_store",
    "TECHNICAL": BASE_DIR / "data" / "vectorStores" / "technical_store"
}

EMBEDDINGS = OpenAIEmbeddings()
K = 5

# ============================================================
# Cargar vectorstores
# ============================================================
STORES: Dict[str, FAISS] = {}

def load_store(path: Path) -> FAISS | None:
    if not path.exists():
        logger.warning(f"Vectorstore {path} no existe.")
        return None
    try:
        store = FAISS.load_local(path, EMBEDDINGS, allow_dangerous_deserialization=True)
        logger.debug(f"Vectorstore cargado: {path}")
        return store
    except Exception as e:
        logger.warning(f"No se pudo cargar el vectorstore {path}: {e}")
        return None

for name, path in VECTORSTORES.items():
    store_instance = load_store(path)
    if store_instance:
        STORES[name] = store_instance

# ============================================================
# Funciones de recuperación de contexto
# ============================================================
def retrieve_context(system: str, flight_phase: str, severity: str) -> List[Document]:
    query = f"Sistema: {system}, Fase de vuelo: {flight_phase}, Severidad: {severity}"
    retrieved: List[Document] = []
    for store_name, store in STORES.items():
        if store:
            try:
                docs = store.max_marginal_relevance_search(query=query, k=K, fetch_k=20, lambda_mult=0.7)
                retrieved.extend(docs)
            except Exception as e:
                logger.warning(f"Error buscando en store {store_name}: {e}")
    return retrieved

def build_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "UNKNOWN")
        blocks.append(f"[{source}]\n{d.page_content}")
    return "\n\n".join(blocks)

# ============================================================
# Prompt de Reparación
# ============================================================
REPARACION_PROMPT = """
Eres un ingeniero de mantenimiento aeronáutico experto.

Usando la información proporcionada (informes operacionales, reportes de incidentes, 
documentación técnica, normativa y procedimientos de Airbus/FAA/EASA), genera recomendaciones 
de reparación o mitigación para la condición detectada.

Debes entregar EXCLUSIVAMENTE un objeto JSON con esta estructura:

{{
  "system_affected": "{system}",
  "flight_phase": "{flight_phase}",
  "severity": "{severity}",
  "recommended_actions": [
    "Acción correctiva 1",
    "Acción correctiva 2",
    "..."
  ],
  "references": [
    "Documento o base de datos que soporta la acción",
    "..."
  ],
  "notes": "Comentarios técnicos adicionales"
}}

NO incluyas texto fuera del JSON.
Información disponible:
{context}
"""

# ============================================================
# Acción del agente
# ============================================================
def reparacion_action(state: AgentState) -> AgentState:
    """
    Genera recomendaciones de reparación o mitigación basadas en criticidad.
    """
    print(">>> Ejecutando acción REPARACION")
    logger.info(">>> REPARACION")

    if not state.criticidad:
        logger.error("No se detectó criticidad en el estado. Se requiere criticidad antes de reparacion.")
        state.messages.append(AIMessage(content="No hay información de criticidad disponible para generar recomendaciones de reparación."))
        state.needs_followup = False
        state.next_agent = None
        return state

    try:
        system = state.criticidad.get("affected_system", "Desconocido")
        flight_phase = state.criticidad.get("flight_phase", "Desconocida")
        severity = state.criticidad.get("severity", "MEDIUM")

        docs = retrieve_context(system, flight_phase, severity)
        context = build_context(docs)

        prompt = REPARACION_PROMPT.format(
            system=system,
            flight_phase=flight_phase,
            severity=severity,
            context=context
        )

        response = llm_creative.invoke(prompt)
        raw_text = response.content.strip()
        
        # parsear JSON
        try:
            reparacion_data = json.loads(raw_text)
        except Exception as e:
            logger.error(f"Error parseando JSON de ReparacionAgent: {e}")
            state.messages.append(AIMessage(content=f"Error generando recomendaciones de reparación: {e}"))
            state.needs_followup = False
            state.next_agent = None
            return state

        # Guardar en state
        state.reparacion = reparacion_data
        state.messages.append(AIMessage(content=f"Recomendaciones de reparación generadas para el sistema {system}."))
        

        # Seguimiento
        state.needs_followup = False
        state.next_agent = None

        return state

    except Exception as e:
        logger.exception(f"Error interno en reparacion_action: {e}")
        state.messages.append(AIMessage(content=f"Error interno en agente de reparación: {e}"))
        state.needs_followup = False
        state.next_agent = None
        return state
