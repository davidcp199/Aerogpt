from langgraph.graph import StateGraph, START, END
from agents.State import AgentState
from agents.SupervisorAgent import supervisor_action
from agents.RulAgent import extract_cmapss_action
from agents.CriticidadAgent import criticidad_action
from agents.ReparacionAgent import reparacion_action
from agents.RegulacionAgent import regulacion_action
from agents.GeneralAgent import general_action
from agents.PreRulAgent import pre_rul_action

class GraphBuilder:
    def __init__(self):
        pass

    def supervisor_decision(self, state: AgentState):
        """Decide el siguiente agente basado en la decisión del Supervisor."""
        print(f">>> Decision {state.decision}")
        if state.decision in ["PreRUL", "Criticidad", "Reparacion", "Regulacion", "General"]:
            return state.decision
        return END

    def followup_decision(self, state: AgentState):
        """Si un agente indica que se necesita otro agente, se invoca."""
        if state.needs_followup and state.next_agent:
            return state.next_agent
        return END

    def build(self):
        graph = StateGraph(AgentState)

        # nodos
        graph.add_node("Supervisor", supervisor_action)
        graph.add_node("PreRUL", pre_rul_action)
        graph.add_node("RUL", extract_cmapss_action)
        graph.add_node("Criticidad", criticidad_action)
        graph.add_node("Reparacion", reparacion_action)
        graph.add_node("Regulacion", regulacion_action)
        graph.add_node("General", general_action)

        # edges
        graph.add_edge(START, "Supervisor")
        graph.add_conditional_edges("Supervisor", self.supervisor_decision)
        graph.add_conditional_edges("PreRUL", self.followup_decision)
        #graph.add_conditional_edges("General", self.followup_decision)
        graph.add_conditional_edges("RUL", self.followup_decision)
        graph.add_edge("Criticidad", END)
        
        # agentes individuales pueden disparar followup
        # for agent_name in ["RUL", "Criticidad", "Reparacion", "Regulacion"]:
        #     graph.add_conditional_edges(agent_name, self.followup_decision)
        #     graph.add_conditional_edges(agent_name, self.agent_end)

        return graph.compile()
