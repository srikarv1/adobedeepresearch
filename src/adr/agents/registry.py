from __future__ import annotations

from pathlib import Path

import yaml

from adr.agents.base import ResearchAgent
from adr.agents.fixture import build as build_fixture
from adr.agents.intern import build_deep_research, build_learned, build_pilot

_BUILDERS = {
    "fixture": build_fixture,
    "deep_research": build_deep_research,
    "pilot": build_pilot,
    "learned": build_learned,
}


def available_agents() -> list[str]:
    return sorted(_BUILDERS)


def build_agent(name: str, config: dict | str | Path | None = None) -> ResearchAgent:
    key = name.strip().lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown agent {name!r}. Registered: {available_agents()}")
    cfg = _load_config(config)
    return _BUILDERS[key](cfg)


def _load_config(config: dict | str | Path | None) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Agent config must be a mapping: {path}")
    return loaded
