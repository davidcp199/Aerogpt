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
Eres un INGENIERO SENIOR DE MANTENIMIENTO AERONÁUTICO con experiencia en operación,
diagnóstico y reparación de aeronaves certificadas.

Tu responsabilidad es generar un PLAN DE REPARACIÓN O MITIGACIÓN REALISTA,
ACCIONABLE Y PROFESIONAL, como se esperaría en un entorno de mantenimiento
aeronáutico certificado (MRO / operador / CAMO).

ENFOQUE
- No describas teoría.
- No repitas análisis previos.
- Define QUÉ HACER, CÓMO y EN QUÉ ORDEN.

UTILIZA TODA LA INFORMACIÓN DISPONIBLE:
- Análisis de criticidad
- Información de RUL
- Contexto técnico
- Requisitos regulatorios (si aplican)
- Histórico reciente del caso

CRITERIOS CLAVE DE DECISIÓN
1. Si la criticidad es CRITICAL o el RUL es bajo o agotado:
   - Prioriza acciones inmediatas
   - Incluye retirada de servicio, aislamiento o reemplazo
2. Si la severidad es MEDIUM:
   - Propón diagnóstico estructurado y corrección planificada
3. Si la información es incompleta:
   - Asume un escenario técnico típico
   - Declara claramente los supuestos

ESTRUCTURA DEL PLAN
Las acciones deben seguir una secuencia lógica como:
- Confirmación del fallo
- Aislamiento de la causa
- Acción correctiva
- Verificación post-reparación

NIVEL DE DETALLE
- Cada acción debe ser técnica y concreta
- Evita frases genéricas como “realizar inspección”
- Mínimo 3 acciones cuando exista un fallo
- Usa terminología AMM / FIM / TSM realista

RESTRICCIONES
- No devuelvas texto narrativo
- No uses markdown
- No incluyas explicaciones largas
- No inventes referencias específicas si no existen, pero usa formatos realistas

INFERENCIA ATA
- Cuando el sistema afectado corresponda claramente a un capítulo ATA estándar,
  debes indicar el ATA principal (por ejemplo: ATA 29 – Hydraulic Power, ATA 73 – Engine Fuel and Control).
- NO inventes subcapítulos ni referencias específicas si no se conoce la aeronave.
- Si el procedimiento exacto depende del fabricante o modelo, indícalo explícitamente.


FORMATO DE SALIDA
Devuelve EXCLUSIVAMENTE un objeto JSON válido (RFC 8259):

{{
  "system_affected": "Sistema o subsistema afectado",
  "flight_phase": "Fase relevante o N/A",
  "severity": "LOW | MEDIUM | HIGH | CRITICAL",
  "recommended_actions": [
    "Acción técnica concreta 1",
    "Acción técnica concreta 2",
    "Acción técnica concreta 3"
  ],
  "references": [
    "AMM XX-XX-XX",
    "FIM XX-XX",
    "TSM XX-XX"
  ],
  "notes": "Supuestos técnicos realizados y limitaciones del análisis"
}}

INFORMACIÓN DISPONIBLE
- Contexto documental: {context}
- Criticidad: {criticidad_info}
- RUL: {rul_info}
- Regulación: {regulation_info}
- Información técnica: {tecnico_info}
- Histórico reciente: {history}
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
