import pandas as pd
from agents.State import AgentState


# ==============================================
# Detección de nuevo caso
# ==============================================

def is_new_case(user_text: str) -> bool:
    triggers = [
        "nuevo motor",
        "nuevo avión",
        "otro motor",
        "otro avión",
        "caso nuevo",
        "equipo distinto",
        "empezar de cero",
        "motor diferente",
        "avión diferente"
    ]
    text = user_text.lower()
    return any(t in text for t in triggers)


# ==============================================
# Reset completo del estado
# ==============================================

def reset_full_case(state: AgentState):
    state.history_by_agent = {
        "Regulacion": [],
        "Criticidad": [],
        "Reparacion": [],
        "Tecnico": [],
        "RUL": [],
        "PreRUL": [],
        "General": [],
        "Final": []
    }

    state.conversation_summary = ""
    state.messages = []

    state.regulation = None
    state.criticidad = None
    state.dispatch_allowed = None
    state.reparacion = None
    state.rul = None
    state.pre_rul_data = None
    state.modelo_seleccionado = "FD001"
    state.tecnico = None
    state.final_response = None
    state.general_notes = None

    state.output_buffer = []
    state.debug_buffer = []


# ==============================================
# Reset selectivo por iteración
# ==============================================

def reset_state_iteration(state: AgentState):
    agent_attrs = [
        "regulation",
        "criticidad",
        "reparacion",
        "tecnico",
        "dispatch_allowed",
        "needs_followup",
        "next_agent",
        "source"
    ]

    for attr in agent_attrs:
        if attr in ["dispatch_allowed", "needs_followup"]:
            setattr(state, attr, False)
        else:
            setattr(state, attr, None)


# ==============================================
# Guardar historial por agente
# ==============================================

FIELD_AGENT_MAP = {
    "regulation": "Regulacion",
    "criticidad": "Criticidad",
    "reparacion": "Reparacion",
    "pre_rul_data": "PreRUL",
    "rul": "RUL",
    "tecnico": "Tecnico",
    "final_response": "Final",
    "general_notes": "General"
}

def save_history(state: AgentState):
    for field, agent in FIELD_AGENT_MAP.items():
        val = getattr(state, field, None)
        if val is None:
            continue

        if isinstance(val, pd.DataFrame):
            val_to_save = val.to_dict(orient="records")
        else:
            val_to_save = val

        history = state.history_by_agent.setdefault(agent, [])
        entry = {field: val_to_save}

        if entry not in history:
            history.append(entry)
