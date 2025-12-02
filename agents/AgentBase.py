class AgentBase:
    def __init__(self, model_name: str = None, temperature: float = None):
        self.model_name = model_name
        self.temperature = temperature
