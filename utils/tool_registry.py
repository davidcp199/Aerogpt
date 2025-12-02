from typing import Callable, Dict, Any

class ToolRegistry:
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, tool_fn: Callable):
        cls._registry[name] = tool_fn

    @classmethod
    def get(cls, name: str):
        return cls._registry.get(name)

    @classmethod
    def all(cls):
        return dict(cls._registry)

    @classmethod
    def invoke(cls, name: str, *args, **kwargs) -> Any:
        tool = cls.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not registrada.")

        if hasattr(tool, "invoke") and callable(getattr(tool, "invoke")):
            return tool.invoke(dict(zip(["message"] if args else [], args), **kwargs))
        if hasattr(tool, "run") and callable(getattr(tool, "run")):
            return tool.run(*args, **kwargs)
        if callable(tool):
            return tool(*args, **kwargs)

        raise TypeError(f"Tool '{name}' no es invocable.")
