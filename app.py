import streamlit as st
from langchain_core.messages import HumanMessage

# ===== IMPORTS DEL PROYECTO =====
from main import (
    graph,
    AgentState,
    is_new_case,
    reset_state_iteration,
    save_history
)

# ==============================================
# Configuración UI
# ==============================================
st.set_page_config(
    page_title="AeroGPT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("✈️ AeroGPT")
st.caption("Asistente inteligente para análisis aeronáutico")

# ==============================================
# Sidebar: configuración
# ==============================================
with st.sidebar:
    st.subheader("Configuración")

    show_debug = st.checkbox(
        "🔧 Mostrar decisiones internas",
        value=True
    )

    if st.button("🧹 Limpiar conversación"):
        for key in ["chat", "state"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ==============================================
# Inicialización de estado persistente
# ==============================================
if "state" not in st.session_state:
    st.session_state.state = AgentState(messages=[])
    st.session_state.state.history_by_agent = {
        "Regulacion": [],
        "Criticidad": [],
        "Reparacion": [],
        "Tecnico": [],
        "RUL": [],
        "PreRUL": [],
        "General": [],
        "Final": []
    }

if "chat" not in st.session_state:
    st.session_state.chat = []

state = st.session_state.state

# ==============================================
# Función de procesamiento principal
# ==============================================
def process_input(user_input: str, show_debug: bool):
    global state

    # --- NUEVO CASO ---
    if is_new_case(user_input):
        state.emit("\n---> Nuevo caso detectado. Limpiando estado.", level="debug")
        state.rul = None
        state.criticidad = None
        state.reparacion = None
        state.regulation = None

    # --- RESET DE BUFFERS ---
    state.output_buffer = []
    state.debug_buffer = []

    # --- RESET DE ESTADO POR ITERACIÓN ---
    reset_state_iteration(state)

    # --- MENSAJE HUMANO ---
    state.messages.append(HumanMessage(content=user_input))

    # --- EJECUTAR GRAFO ---
    result = graph.invoke(state)

    if isinstance(result, AgentState):
        state = result
    elif isinstance(result, dict):
        state = AgentState(**result)
    else:
        raise TypeError(f"Resultado inesperado del grafo: {type(result)}")

    # --- HISTORIAL ---
    save_history(state)

    # --- LIMPIAR MENSAJES IA ---
    state.messages = [
        m for m in state.messages if isinstance(m, HumanMessage)
    ]

    # --- CONSTRUIR RESPUESTA ---
    response_parts = []

    # 1) DEBUG (DECISIONES INTERNAS)
    if show_debug and state.debug_buffer:
        debug_text = "\n".join(state.debug_buffer)
        response_parts.append(
            f"""
<div style="
    color:#6c757d;
    font-size:0.9em;
    background-color:#f8f9fa;
    padding:0.75em;
    border-radius:6px;
    border-left:4px solid #adb5bd;
">
<strong>Decisiones internas</strong><br><br>
<pre style="
    margin:0;
    color:#6c757d;
    background-color:transparent;
">{debug_text}</pre>
</div>
"""
        )

    # 2) SALIDA USUARIO (FinalAgent)
    if state.output_buffer:
        response_parts.append("\n".join(state.output_buffer))
    else:
        response_parts.append("⚠️ El proceso terminó sin salida.")

    final_response = "\n\n".join(response_parts)

    st.session_state.state = state
    return final_response

# ==============================================
# Input del usuario
# ==============================================
user_input = st.chat_input("Escribe tu consulta")

if user_input:
    st.session_state.chat.append(("user", user_input))

    with st.spinner("Pensando…"):
        response = process_input(user_input, show_debug)

    st.session_state.chat.append(("assistant", response))

# ==============================================
# Render del chat
# ==============================================
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg, unsafe_allow_html=True)

# ==============================================
# Sidebar: historial por agente
# ==============================================
with st.sidebar:
    if st.checkbox("📜 Mostrar historial por agente"):
        for agent, entries in state.history_by_agent.items():
            st.markdown(f"### {agent}")
            if entries:
                for i, entry in enumerate(entries):
                    st.json({f"{i+1}": entry})
            else:
                st.caption("Sin entradas")
