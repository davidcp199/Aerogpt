# agents/GraphBuilder.py
from langgraph.graph import StateGraph, START, END
from agents.State import AgentState
from agents.SupervisorAgent import supervisor_action
from agents.RulAgent import extract_cmapss_action

class GraphBuilder:
    def __init__(self):
        pass

    def supervisor_decision(self, state: AgentState):
        if state.decision == "extract":
            return "Extractor"
        return END

    def extractor_end(self, state: AgentState):
        return END

    def build(self):
        graph = StateGraph(AgentState)

        graph.add_node("Supervisor", supervisor_action)
        graph.add_node("Extractor", extract_cmapss_action)

        graph.add_edge(START, "Supervisor")
        graph.add_conditional_edges("Supervisor", self.supervisor_decision)
        graph.add_conditional_edges("Extractor", self.extractor_end)

        return graph.compile()
