from langchain_openai import ChatOpenAI
from utils.config_loader import load_all_configs
import os
from functools import lru_cache

@lru_cache()
def create_llms(repo_root: str = None):
    model_cfg, paths_cfg, settings_cfg = load_all_configs(repo_root)

    os.environ["OPENAI_API_KEY"] = settings_cfg["settings"].get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))

    det = model_cfg["models"]["deterministic"]
    cre = model_cfg["models"]["creative"]

    llm_deterministic = ChatOpenAI(
        model=det["name"],
        temperature=det["temperature"],
        max_tokens=det.get("max_tokens")
    )
    llm_creative = ChatOpenAI(
        model=cre["name"],
        temperature=cre["temperature"],
        max_tokens=cre.get("max_tokens")
    )

    return {"deterministic": llm_deterministic, "creative": llm_creative, "paths": paths_cfg, "settings": settings_cfg}


_llms_cache = create_llms()

llm_deterministic = _llms_cache["deterministic"]
llm_creative = _llms_cache["creative"]
paths_config = _llms_cache["paths"]
settings_config = _llms_cache["settings"]

def get_llm(profile: str = "deterministic"):
    return llm_creative if profile == "creative" else llm_deterministic
