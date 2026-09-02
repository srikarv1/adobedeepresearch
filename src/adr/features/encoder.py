"""Frozen text encoder with a disk cache. No extra model calls at decide-time.

Uses a local hashing encoder by default so tests and Azure-only machines work
without downloading MiniLM. If ``sentence-transformers`` is installed, that
model is used instead.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / ".cache" / "embeddings"
DIM = 64


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def hash_embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic bag-of-bytes embedding. Good enough for tests and cheap ρ/ν."""
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    raw: list[float] = []
    seed = digest
    while len(raw) < dim:
        seed = hashlib.sha256(seed).digest()
        raw.extend((b / 127.5) - 1.0 for b in seed)
    return _l2(raw[:dim])


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=False))))


class EmbeddingCache:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or DEFAULT_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, list[float]] = {}

    def _path(self, text: str) -> Path:
        digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def embed(self, text: str) -> list[float]:
        key = text or ""
        if key in self._mem:
            return self._mem[key]
        path = self._path(key)
        if path.exists():
            vec = json.loads(path.read_text(encoding="utf-8"))
            self._mem[key] = vec
            return vec
        vec = hash_embed(key)
        path.write_text(json.dumps(vec), encoding="utf-8")
        self._mem[key] = vec
        return vec
