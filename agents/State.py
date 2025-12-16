from pydantic import BaseModel, field_validator
import pandas as pd
from typing import Optional, List, Dict
from langchain_core.messages import BaseMessage

class AgentState(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }

    messages: List[BaseMessage]
    pre_rul_data: Optional[pd.DataFrame] = None
    modelo_seleccionado: Optional[str] = "FD001"
    decision: Optional[str] = None
    next_agent: Optional[str] = None
    needs_followup: bool = False
    source: Optional[str] = None
    conversation_summary: str = ""
    last_agent_output: Dict[str, str] = {}

    @field_validator("pre_rul_data", mode="before")
    def convert_to_df(cls, v):
        if isinstance(v, dict):
            return pd.DataFrame([v])  # lo convierte automáticamente
        return v
