import json
import logging
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

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
# Recuperación de contexto
# ============================================================
def retrieve_context(system: str, flight_phase: str, severity: str, user_query: str) -> List[Document]:
    query = f"{user_query} | Sistema: {system} | Fase: {flight_phase} | Severidad: {severity}"
    retrieved: List[Document] = []

    for store_name, store in STORES.items():
        try:
            docs = store.max_marginal_relevance_search(
                query=query,
                k=K,
                fetch_k=20,
                lambda_mult=0.7
            )
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
# Prompt Reparación
# ============================================================
REPARACION_PROMPT = ChatPromptTemplate.from_template(
"""
Eres un ingeniero de mantenimiento aeronáutico experto.

Debes generar recomendaciones de reparación, inspección o mitigación técnica
basándote en la información disponible.

Contexto posible:
- Puede existir un análisis previo de criticidad
- O puede tratarse de una consulta directa de troubleshooting sin análisis de safety

Si no hay información suficiente:
- Proporciona acciones genéricas de diagnóstico (AMM / FIM / TSM)
- Indica claramente supuestos y limitaciones

El JSON debe ser válido según el estándar RFC 8259.
No uses saltos de línea dentro de strings.
No incluyas texto fuera del JSON.

Devuelve EXCLUSIVAMENTE un objeto JSON con esta estructura:

{{
  "system_affected": "{system}",
  "flight_phase": "{flight_phase}",
  "severity": "{severity}",
  "recommended_actions": [
    "Acción correctiva o diagnóstica 1",
    "Acción correctiva o diagnóstica 2"
  ],
  "references": [
    "AMM XX-XX-XX",
    "FIM XX-XX",
    "Fuente técnica"
  ],
  "notes": "Supuestos realizados y observaciones técnicas"
}}

REGLAS CRÍTICAS:
Está PROHIBIDO asumir sistemas, ATA chapters o modos de fallo no mencionados explícitamente
en el contexto o en el análisis previo.

Si no se identifica ningún fallo, degradación, aviso técnico, ni condición anómala,
o si el contexto indica operación normal (por ejemplo RUL alto, parámetros dentro de rango),
DEBES devolver el siguiente JSON indicando que NO procede reparación correctiva.

En ese caso, el JSON DEBE ser exactamente:

{{
  "system_affected": "N/A",
  "flight_phase": "N/A",
  "severity": "NONE",
  "recommended_actions": [
    "No se requiere reparación correctiva.",
    "Continuar operación normal según programa de mantenimiento.",
    "Mantener monitoreo de tendencias e inspecciones preventivas (MPD)."
  ],
  "references": [
    "MPD",
    "Manual de mantenimiento del motor"
  ],
  "notes": "No se han identificado fallos ni condiciones anómalas que requieran intervención."
}}


Información disponible:
{context}
"""
)

# ============================================================
# Acción del agente
# ============================================================
def reparacion_action(state: AgentState) -> AgentState:
    """
    Genera recomendaciones de reparación:
    - Desde criticidad (flujo safety)
    - O directamente como troubleshooting técnico
    """
    print(">>> Ejecutando acción REPARACION")
    logger.info(">>> REPARACION")
    state.source = "Reparacion"

    # --------------------------------------------------------
    # Determinar origen
    # --------------------------------------------------------
    if state.criticidad:
        system = state.criticidad.get("affected_system", "Desconocido")
        flight_phase = state.criticidad.get("flight_phase", "Desconocida")
        severity = state.criticidad.get("severity", "MEDIUM")
    else:
        logger.info("Reparacion sin criticidad previa (modo troubleshooting)")
        system = "No especificado"
        flight_phase = "UNKNOWN"
        severity = "MEDIUM"

    user_query = state.messages[-1].content

    try:
        docs = retrieve_context(system, flight_phase, severity, user_query)
        context = build_context(docs)

        chain = REPARACION_PROMPT | llm_creative
        response = chain.invoke({
            "system": system,
            "flight_phase": flight_phase,
            "severity": severity,
            "context": context
        })

        raw_text = response.content.strip()

        # ----------------------------------------------------
        # Limpieza defensiva del output
        # ----------------------------------------------------
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            raw_text = raw_text.replace("json", "", 1).strip()

        # ----------------------------------------------------
        # Parseo JSON
        # ----------------------------------------------------
        try:
            reparacion_data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error("JSON inválido devuelto por LLM:\n%s", raw_text)
            state.messages.append(
                AIMessage(
                    content=(
                        "He generado recomendaciones técnicas, pero el formato "
                        "estructurado no pudo validarse correctamente. "
                        "¿Desea que lo reformule?"
                    )
                )
            )
            state.needs_followup = False
            state.next_agent = None
            return state

        # ----------------------------------------------------
        # Guardar resultado
        # ----------------------------------------------------
        state.reparacion = reparacion_data
        state.messages.append(
            AIMessage(content=f"Recomendaciones de reparación generadas:\n{json.dumps(reparacion_data, indent=2)}")
        )

        state.needs_followup = True
        state.next_agent = "Final"
        return state

    except Exception as e:
        logger.exception("Error interno en ReparacionAgent: %s", e)
        state.messages.append(
            AIMessage(content="Error interno al generar recomendaciones de reparación.")
        )
        state.needs_followup = False
        state.next_agent = None
        return state
