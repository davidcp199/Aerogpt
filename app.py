import streamlit as st
from langchain_core.messages import HumanMessage

# ===== IMPORTS DEL PROYECTO =====
from main import (
    graph,
    AgentState,
    is_new_case,
    reset_state_iteration,
    save_history,
    reset_full_case
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
# Inicialización de session_state (UNA SOLA VEZ)
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

if "show_history_by_agent" not in st.session_state:
    st.session_state.show_history_by_agent = False

if "show_debug" not in st.session_state:
    st.session_state.show_debug = True

state = st.session_state.state

# ==============================================
# Sidebar – CONFIGURACIÓN (solo widgets)
# ==============================================
with st.sidebar:
    st.subheader("Configuración")

    st.checkbox(
        "🔧 Mostrar decisiones internas",
        key="show_debug"
    )

    st.checkbox(
        "📜 Mostrar historial por agente",
        key="show_history_by_agent"
    )

    if st.button("🧹 Limpiar conversación"):
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
        st.session_state.chat = []
        st.rerun()



# ==============================================
# Función principal de procesamiento
# ==============================================
def process_input(user_input: str) -> str:
    state = st.session_state.state

    # --- RESET BUFFERS ---
    state.output_buffer = []
    state.debug_buffer = []

    # --- NUEVO CASO ---
    if is_new_case(user_input):
        reset_full_case(state)
        state.emit("\n🔄 Nuevo caso detectado. Estado técnico limpiado.", level="debug")

    # --- RESET ITERACIÓN ---
    reset_state_iteration(state)

    # --- MENSAJE HUMANO ---
    state.messages.append(HumanMessage(content=user_input))

    # --- EJECUTAR GRAFO ---
    result = graph.invoke(state)

    if isinstance(result, AgentState):
        state = result
    elif isinstance(result, dict):
        for k, v in result.items():
            setattr(state, k, v)
    else:
        raise TypeError(f"Resultado inesperado: {type(result)}")

    # --- GUARDAR HISTORIAL ---
    save_history(state)

    # --- LIMPIAR IA (mantener humanos) ---
    state.messages = [
        m for m in state.messages if isinstance(m, HumanMessage)
    ]

    # --- CONSTRUIR RESPUESTA ---
    parts = []

    if st.session_state.show_debug and state.debug_buffer:
        parts.append(
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
<pre style="margin:0;">{chr(10).join(state.debug_buffer)}</pre>
</div>
"""
        )

    if state.output_buffer:
        parts.append("\n".join(state.output_buffer))
    else:
        parts.append("⚠️ El proceso terminó sin salida.")

    st.session_state.state = state
    return "\n\n".join(parts)

# ==============================================
# Render del chat
# ==============================================
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg, unsafe_allow_html=True)

# ==============================================
# Input del usuario (SIN rerun)
# ==============================================
user_input = st.chat_input("Escribe tu consulta")

if user_input:
    # Usuario
    st.session_state.chat.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Asistente
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            "<div style='color:#6c757d; font-style:italic;'>Pensando…</div>",
            unsafe_allow_html=True
        )

        response = process_input(user_input)
        placeholder.markdown(response, unsafe_allow_html=True)

        # ============================================
        # MOSTRAR SOLO TABLA SI EL PRERUL DECIDIÓ STATUS
        # ============================================
        state = st.session_state.state
        if state.pre_rul_data is not None and not state.pre_rul_data.empty:
            # Revisar si la última acción fue 'status' en debug_buffer
            if any("PreRUL Accion decidida: Status" in line for line in state.debug_buffer):
                fd_actual = state.modelo_seleccionado or "FD001"

                # Convertir DataFrame a tabla HTML con scroll horizontal
                df_html = state.pre_rul_data.to_html(
                    index=False,
                    border=1,
                    justify="center",
                    classes="motor-table"
                )

                html_content = f"""
                <div>
                    <b style="color:white;">(FD: {fd_actual})</b><br><br>
                    <div style="overflow-x:auto; max-width:100%;">
                        {df_html}
                    </div>
                    <style>
                        table.motor-table {{
                            border-collapse: collapse;
                            width: 100%;
                            font-family: Arial, sans-serif;
                            font-size: 0.9em;
                            color: white;  /* Texto de la tabla */
                            background-color: #1e1e1e; /* Fondo de la tabla */
                        }}
                        table.motor-table th, table.motor-table td {{
                            border: 1px solid #555; /* Bordes más claros para fondo oscuro */
                            padding: 5px;
                            text-align: center;
                        }}
                        table.motor-table th {{
                            background-color: #333; /* Encabezado más oscuro */
                            color: #fff; /* Texto encabezado */
                        }}
                        table.motor-table tr:nth-child(even) {{
                            background-color: #2a2a2a; /* Filas pares */
                        }}
                        table.motor-table tr:nth-child(odd) {{
                            background-color: #1e1e1e; /* Filas impares */
                        }}
                    </style>
                </div>
                """


                st.markdown(html_content, unsafe_allow_html=True)

    st.session_state.chat.append(("assistant", response))


# ==============================================
# Sidebar – HISTORIAL POR AGENTE (render)
# ==============================================
if st.session_state.show_history_by_agent:
    with st.sidebar:
        st.divider()
        st.subheader("Historial por agente")

        for agent, entries in state.history_by_agent.items():
            st.markdown(f"### {agent}")
            if entries:
                for i, entry in enumerate(entries):
                    st.json({f"{i+1}": entry})
            else:
                st.caption("Sin entradas")
