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
# Prompt Reparación actualizado
# ============================================================
REPARACION_PROMPT = ChatPromptTemplate.from_template(
"""
Eres un ingeniero de mantenimiento aeronáutico experto.

Genera recomendaciones de reparación, inspección o mitigación técnica
basándote en toda la información disponible:

- Análisis previo de criticidad
- Información de RUL
- Información regulatoria
- Información técnica
- Histórico reciente de mensajes del usuario

Si la criticidad es CRITICAL o RUL cercano a fin de vida, debes generar
acciones correctivas inmediatas y no devolver "NO PROCEDEN REPARACIÓN CORRECTIVA".

Si no hay información suficiente:
- Proporciona acciones genéricas de diagnóstico (AMM / FIM / TSM)
- Indica supuestos y limitaciones

El JSON debe ser válido según RFC 8259.
No uses saltos de línea dentro de strings.
Devuelve exclusivamente un objeto JSON con la estructura:

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

Información disponible:
- Contexto de documentos: {{context}}
- Análisis Criticidad: {{criticidad_info}}
- RUL: {{rul_info}}
- Regulación: {{regulation_info}}
- Información Técnica: {{tecnico_info}}
- Histórico reciente: {{history}}
"""
)

# ============================================================
# Acción del agente
# ============================================================
def reparacion_action(state: AgentState) -> AgentState:
    """
    Genera recomendaciones de reparación basadas en:
    Criticidad, RUL, Regulación, Información técnica y contexto histórico
    """
    print(">>> Ejecutando acción REPARACION")
    logger.info(">>> REPARACION")
    state.source = "Reparacion"

    try:
        # --------------------------------------------------------
        # Extraer contexto principal
        # --------------------------------------------------------
        system = state.criticidad.get("affected_system", "Desconocido") if state.criticidad else "No especificado"
        flight_phase = state.criticidad.get("flight_phase", "Desconocida") if state.criticidad else "UNKNOWN"
        severity = state.criticidad.get("severity", "MEDIUM").upper() if state.criticidad else "MEDIUM"

        user_query = state.messages[-1].content

        # --------------------------------------------------------
        # Construir contexto integrado
        # --------------------------------------------------------
        context_blocks = []

        if state.criticidad:
            context_blocks.append(f"[CRITICIDAD]\n{json.dumps(state.criticidad, ensure_ascii=False, indent=2)}")
        if state.rul:
            context_blocks.append(f"[RUL]\n{state.rul}")
        if getattr(state, "regulation", None):
            context_blocks.append(f"[REGULACION]\n{json.dumps(state.regulation, ensure_ascii=False, indent=2)}")
        if getattr(state, "tecnico", None):
            context_blocks.append(f"[TECNICO]\n{json.dumps(state.tecnico, ensure_ascii=False, indent=2)}")
        # Histórico reciente de mensajes
        history = "\n".join([m.content for m in state.messages[-5:]])  # últimos 5 mensajes
        context_blocks.append(f"[HISTORICO]\n{history}")

        context_str = "\n\n".join(context_blocks)

        # --------------------------------------------------------
        # Recuperar documentos vectorstores
        # --------------------------------------------------------
        docs = []
        for store_name, store in STORES.items():
            try:
                retrieved_docs = store.max_marginal_relevance_search(
                    query=f"{user_query} | Sistema: {system} | Fase: {flight_phase} | Severidad: {severity}",
                    k=K,
                    fetch_k=20,
                    lambda_mult=0.7
                )
                docs.extend(retrieved_docs)
            except Exception as e:
                logger.warning(f"Error buscando en store {store_name}: {e}")

        for d in docs:
            source = d.metadata.get("source", "UNKNOWN")
            context_str += f"\n\n[{source}]\n{d.page_content}"

        # --------------------------------------------------------
        # Invocar LLM
        # --------------------------------------------------------
        chain = REPARACION_PROMPT | llm_creative
        response = chain.invoke({
            "system": system,
            "flight_phase": flight_phase,
            "severity": severity,
            "context": context_str,
            "criticidad_info": json.dumps(state.criticidad, ensure_ascii=False) if state.criticidad else "",
            "rul_info": state.rul or "",
            "regulation_info": json.dumps(state.regulation, ensure_ascii=False) if getattr(state, "regulation", None) else "",
            "tecnico_info": json.dumps(state.tecnico, ensure_ascii=False) if getattr(state, "tecnico", None) else "",
            "history": history
        })

        raw_text = response.content.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").replace("json", "", 1).strip()

        # Parseo JSON defensivo
        try:
            reparacion_data = json.loads(raw_text)
        except json.JSONDecodeError:
            state.messages.append(
                AIMessage(content="He generado recomendaciones técnicas, pero el formato JSON no pudo validarse. Solicite reformulación.")
            )
            state.needs_followup = False
            state.next_agent = None
            return state

        # Guardar resultado
        state.reparacion = reparacion_data
        state.messages.append(
            AIMessage(content=f"Recomendaciones de reparación generadas:\n{json.dumps(reparacion_data, indent=2)}")
        )

        state.needs_followup = True
        state.next_agent = "Final"
        return state

    except Exception as e:
        logger.exception("Error interno en ReparacionAgent: %s", e)
        state.messages.append(AIMessage(content="Error interno al generar recomendaciones de reparación."))
        state.needs_followup = False
        state.next_agent = None
        return state
