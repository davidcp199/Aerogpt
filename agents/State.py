from pydantic import BaseModel
from typing import Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, List

class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]  # historial de mensajes y outputs
    decision: Optional[Literal["RUL", "Criticidad", "Reparacion", "Regulacion", "General"]] = None
    needs_followup: Optional[bool] = False  # indica si se requiere llamar a otro agente
    next_agent: Optional[Literal["RUL", "Criticidad", "Reparacion", "Regulacion", "General"]] = None  # agente a invocar después
