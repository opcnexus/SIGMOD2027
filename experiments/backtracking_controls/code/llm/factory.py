"""LLM backend factory: selects mock or api according to config.llm.backend."""
from .api import APILLM
from .base import BaseLLM
from .mock import MockOracleLLM


def get_llm(cfg):
    b = cfg.get("llm", {}).get("backend", "mock")
    if b == "api":
        return APILLM(temperature=cfg.get("llm", {}).get("temperature", 0.0))
    return MockOracleLLM(
        p=cfg.get("llm", {}).get("mock_edge_accuracy", 0.9),
        seed=cfg.get("seed", 0),
    )


__all__ = ["BaseLLM", "MockOracleLLM", "APILLM", "get_llm"]
