from pydantic import BaseModel
from typing import Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated

class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    decision: Optional[Literal["RUL", "Criticidad", "Reparacion", "Regulacion", "none"]] = None
