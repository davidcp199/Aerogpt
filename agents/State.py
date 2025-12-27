from pydantic import BaseModel, field_validator, Field
import pandas as pd
from typing import Optional, List, Dict, Any
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
    last_ai_message: Optional[str] = None
    conversation_summary: str = ""
    last_agent_output: Dict[str, str] = Field(default_factory=dict)

    regulation: Optional[dict] = None
    criticidad: Optional[dict] = None
    criticidad_sources: Optional[List[str]] = None
    dispatch_allowed: Optional[bool] = None

    reparacion: Optional[dict] = None

    history_by_agent: Dict[str, List[dict]] = {
        "Regulacion": [],
        "Criticidad": [],
        "Reparacion": [],
        "Tecnico": [],
        "RUL": [],
        "PreRUL": [],
        "General": []
    }

    @field_validator("pre_rul_data", mode="before")
    def convert_to_df(cls, v):
        if isinstance(v, dict):
            return pd.DataFrame([v])
        return v
    
    def update_memory(self, agent_name: str, output: str):
        """
        Actualiza la memoria del estado después de que un agente responde.
        """
        # Guardar salida del agente
        self.last_agent_output[agent_name] = output

        # Actualizar resumen de conversación
        self.conversation_summary += f"\n{agent_name}: {output}"

        # Limitar tamaño
        max_len = 3000
        if len(self.conversation_summary) > max_len:
            self.conversation_summary = self.conversation_summary[-max_len:]
