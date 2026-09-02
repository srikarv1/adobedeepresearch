from adr.core.types import PilotDecision
from adr.orchestrate.apply import apply_action, apply_decision
from adr.orchestrate.base import Orchestrator
from adr.orchestrate.fixed import FixedOrchestrator
from adr.orchestrate.learned import LearnedOrchestrator
from adr.orchestrate.prompted import PromptedOrchestrator, parse_orchestration

__all__ = [
    "Orchestrator",
    "PilotDecision",
    "FixedOrchestrator",
    "PromptedOrchestrator",
    "LearnedOrchestrator",
    "apply_action",
    "apply_decision",
    "parse_orchestration",
]


def build_orchestrator(name: str, config: dict | None = None) -> Orchestrator:
    key = (name or "fixed").strip().lower()
    cfg = config or {}
    if key in {"fixed", "none", ""}:
        return FixedOrchestrator(cfg)
    if key in {"prompted", "pilot", "flash", "parallel"}:
        return PromptedOrchestrator(cfg)
    if key in {"learned", "policy"}:
        return LearnedOrchestrator(cfg)
    raise ValueError(f"Unknown orchestrator {name!r}")
