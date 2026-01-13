from email.mime import text
from pydantic import BaseModel, field_validator, Field
import pandas as pd
from typing import Optional, List, Dict, Any, Literal
from langchain_core.messages import BaseMessage

class AgentState(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }

    messages: List[BaseMessage]

    decision: Optional[str] = None
    next_agent: Optional[str] = None
    needs_followup: bool = False
    source: Optional[str] = None
    conversation_summary: Optional[str] = None
    
    output_buffer: List[str] = Field(default_factory=list)
    debug_buffer: List[str] = Field(default_factory=list)

    regulation: Optional[dict] = None
    
    criticidad: Optional[dict] = None
    dispatch_allowed: Optional[bool] = None

    reparacion: Optional[dict] = None

    rul: Optional[dict] = None
    pre_rul_data: Optional[pd.DataFrame] = None
    modelo_seleccionado: Optional[str] = "FD001"

    tecnico: Optional[str] = None
    final_response: Optional[str] = None
    general_notes: Optional[str] = None

    history_by_agent: Dict[str, List[dict]] = {
        "Regulacion": [],
        "Criticidad": [],
        "Reparacion": [],
        "Tecnico": [],
        "RUL": [],
        "PreRUL": [],
        "General": [],
        "Final": []
    }

    @field_validator("pre_rul_data", mode="before")
    def convert_to_df(cls, v):
        if isinstance(v, dict):
            return pd.DataFrame([v])
        return v
    
    def update_memory(self, output: str):
        """
        Actualiza el resumen de conversación solo con la salida final.
        Limita el tamaño a 3000 caracteres.
        """
        # Concatenar salida
        if self.conversation_summary:
            self.conversation_summary += "\n" + output
        else:
            self.conversation_summary = output

        # Limitar tamaño
        max_len = 3000
        if len(self.conversation_summary) > max_len:
            self.conversation_summary = self.conversation_summary[-max_len:]


    # Capturar las salidas de los agentes    
    def emit(self, text: str, level: Literal["user","debug"]="user"):
        """
        level="user" -> para UI
        level="debug" -> para decisiones internas
        """
        if level == "user":
            self.output_buffer.append(text)
        else:
            self.debug_buffer.append(text)
