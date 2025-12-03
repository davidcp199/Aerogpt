from langgraph.graph import StateGraph, START, END
from agents.State import AgentState
from agents.SupervisorAgent import supervisor_action
from agents.RulAgent import extract_cmapss_action
from agents.CriticidadAgent import criticidad_action
from agents.ReparacionAgent import reparacion_action
from agents.RegulacionAgent import regulacion_action

class GraphBuilder:
    def __init__(self):
        pass

    def supervisor_decision(self, state: AgentState):
        """
        Devuelve el nodo que debe ejecutarse según la decisión del supervisor.
        Si la decisión no es un nodo válido, devuelve END.
        """
        print(f"EJ DEC {state.next_agent}")
        if state.next_agent in ["RUL", "Criticidad", "Reparacion", "Regulacion"]:
            print(f"SUP DEC {state.next_agent}")
            return state.next_agent
        return END


    def followup_rul(self, state: AgentState):
        if state.needs_followup and state.next_agent == "Criticidad":
            next_agent = state.next_agent
            state.needs_followup = False
            state.next_agent = None
            print(">>> FOLLOWUP a:", next_agent)
            return next_agent
        return END

    def agent_end(self, state: AgentState):
        """Marca el final del flujo de un agente."""
        return END

    def build(self):
        graph = StateGraph(AgentState)

        # Nodos
        graph.add_node("Supervisor", supervisor_action)
        graph.add_node("RUL", extract_cmapss_action)
        graph.add_node("Criticidad", criticidad_action)
        graph.add_node("Reparacion", reparacion_action)
        graph.add_node("Regulacion", regulacion_action)

        # START → Supervisor
        graph.add_edge(START, "Supervisor")

        # Branch condicional Supervisor
        graph.add_conditional_edges("Supervisor", self.supervisor_decision)

        # Followup: RUL solo puede llamar a Criticidad
        graph.add_conditional_edges("RUL", self.followup_rul)
        graph.add_edge("RUL", END)  # termina si no hay followup

        # Criticidad, Reparacion y Regulacion → END
        graph.add_edge("Criticidad", END)
        graph.add_edge("Reparacion", END)
        graph.add_edge("Regulacion", END)

        return graph.compile()