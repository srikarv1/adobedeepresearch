"""Disk cache for search hits so sweep/dev runs do not re-hit Gym/Tavily."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adr.tools.search import SearchBackend, SearchHit

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / ".cache" / "search"


class CachedSearch:
    """SearchBackend wrapper. Fetch is not cached (usually empty on Gym)."""

    def __init__(self, inner: SearchBackend, *, cache_dir: Path | None = None) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "cached")
        self.cache_dir = Path(cache_dir or DEFAULT_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str, k: int) -> Path:
        digest = hashlib.sha1(f"{self.name}|{k}|{query}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        path = self._key(query, k)
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            return [SearchHit.model_validate(row) for row in rows]
        hits = await self._inner.search(query, k=k)
        path.write_text(
            json.dumps([hit.model_dump() for hit in hits], ensure_ascii=False),
            encoding="utf-8",
        )
        return hits

    async def fetch(self, url: str) -> str:
        return await self._inner.fetch(url)
