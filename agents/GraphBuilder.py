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

    # Decide qué agente ejecutar según la decisión del supervisor
    def supervisor_decision(self, state: AgentState):
        if state.decision == "RUL":
            return "RULAgent"
        elif state.decision == "Criticidad":
            return "CriticidadAgent"
        elif state.decision == "Reparacion":
            return "ReparacionAgent"
        elif state.decision == "Regulacion":
            return "RegulacionAgent"
        return END

    # Todos los agentes terminan en END
    def agent_end(self, state: AgentState):
        return END

    def build(self):
        graph = StateGraph(AgentState)

        # Nodos
        graph.add_node("Supervisor", supervisor_action)
        graph.add_node("RULAgent", extract_cmapss_action)
        graph.add_node("CriticidadAgent", criticidad_action)
        graph.add_node("ReparacionAgent", reparacion_action)
        graph.add_node("RegulacionAgent", regulacion_action)

        # Edges
        graph.add_edge(START, "Supervisor")
        graph.add_conditional_edges("Supervisor", self.supervisor_decision)

        for agent in ["RULAgent", "CriticidadAgent", "ReparacionAgent", "RegulacionAgent"]:
            graph.add_conditional_edges(agent, self.agent_end)

        return graph.compile()