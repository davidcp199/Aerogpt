# utils/config_loader.py
import yaml
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
import os


@lru_cache()  # cachear la carga para evitar repetir IO
def load_yaml(path: str):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs(repo_root: str = None):
    """
    Devuelve (model_cfg, paths_cfg, settings_cfg).
    Todas las rutas en paths_cfg['paths'] son objetos Path absolutos.
    repo_root: si se pasa, se usa para construir rutas relativas; si no, se usa cwd.
    """
    if repo_root is None:
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)

    # Cargar .env desde la raíz del proyecto
    env_path = repo_root / ".env"
    load_dotenv(env_path)

    config_dir = repo_root / "config"

    model_cfg = load_yaml(config_dir / "model_config.yaml")
    paths_cfg = load_yaml(config_dir / "paths.yaml")
    settings_cfg = load_yaml(config_dir / "settings.yaml")

    # Construir rutas absolutas desde base
    base_path = (repo_root / paths_cfg["paths"].get("base", ".")).resolve()
    paths_cfg["paths"]["base"] = base_path

    for key, value in paths_cfg["paths"].items():
        if key != "base":
            # Convertir a Path absoluto
            paths_cfg["paths"][key] = (Path(str(value)).expanduser() 
                                       if Path(str(value)).is_absolute() 
                                       else (base_path / str(value)).resolve())

    # Garantizar API key
    settings_cfg["settings"]["openai_api_key"] = os.getenv(
        "OPENAI_API_KEY",
        settings_cfg["settings"].get("openai_api_key")
    )

    return model_cfg, paths_cfg, settings_cfg
