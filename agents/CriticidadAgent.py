from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage

from utils.llm_provider import llm_deterministic, paths_config
from agents.State import AgentState


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
    except Exception:
        return None


STORES = {
    name: store
    for name, path in VECTORSTORES.items()
    if (store := load_store(path)) is not None
}


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(question: str) -> List[Document]:
    if not isinstance(question, str):
        raise TypeError("Question must be a string")

    retrieved: List[Document] = []

    for store in STORES.values():
        retrieved.extend(store.similarity_search(question, k=K))

    return retrieved


def build_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "UNKNOWN")
        blocks.append(f"[{source}]\n{d.page_content}")
    return "\n\n".join(blocks)


# ============================================================
# PROMPT
# ============================================================

CRITICIDAD_PROMPT = """
You are an aviation safety and reliability engineer.

Using the provided evidence (operational reports, service difficulty data,
technical documentation and regulations), assess the operational criticality
of the described condition.

You must:
- Identify the affected system and phase of flight
- Assess severity (LOW / MEDIUM / HIGH / CRITICAL)
- Explain operational risk and failure propagation
- Reference historical evidence when applicable
- Avoid speculation beyond provided data

Question:
{question}

Evidence:
{context}

Provide a structured professional assessment.
"""


# ============================================================
# LANGGRAPH NODE
# ============================================================

def criticidad_action(state: AgentState) -> AgentState:
    print(">>> Ejecutando acción CRITICIDAD")
    question = state.messages[-1].content
    if not question:
        state.messages.append(
            AIMessage(content="No se ha proporcionado ninguna pregunta técnica para analizar.")
        )

    if not isinstance(question, str):
        raise ValueError("AgentState.question must be a string")

    docs = retrieve_context(question)
    context = build_context(docs)

    prompt = CRITICIDAD_PROMPT.format(
        question=question,
        context=context
    )

    response = llm_deterministic.invoke(prompt)

    return state.model_copy(
        update={
            "criticidad": response.content,
            "criticidad_sources": [
                d.metadata.get("source", "UNKNOWN") for d in docs
            ]
        }
    )
