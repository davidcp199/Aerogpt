from pydantic import BaseModel
from typing import Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, List, Dict, Any

class AgentState(BaseModel):
    messages: List[BaseMessage]
    pre_rul_data: Optional[Dict[str, Any]] = None
    decision: Optional[str] = None         # decisión interna del nodo
    next_agent: Optional[str] = None       # siguiente nodo en el grafo
    needs_followup: bool = False

    def update_pre_rul_data(self, updates: Dict[str, Any]):
        if self.pre_rul_data is None:
            self.pre_rul_data = {
                "unidad": 0,
                "tiempo_ciclos": 0,
                "configuraciones_operativas": [0,0,0],
                "mediciones_sensores": {f"s_{i}": 0 for i in range(1, 22)},
                "modelo_seleccionado": "FD001"
            }
        for k, v in updates.items():
            if v is not None:
                if isinstance(v, dict) and k in self.pre_rul_data:
                    self.pre_rul_data[k].update(v)
                elif isinstance(v, list) and k in self.pre_rul_data:
                    for i, val in enumerate(v):
                        if val != 0:
                            self.pre_rul_data[k][i] = val
                else:
                    self.pre_rul_data[k] = v
