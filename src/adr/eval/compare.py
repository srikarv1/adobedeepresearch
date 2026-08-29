from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_summaries(left: Path, right: Path) -> dict[str, Any]:
    a = json.loads(Path(left).read_text(encoding="utf-8"))
    b = json.loads(Path(right).read_text(encoding="utf-8"))
    keys = sorted(
        {
            k
            for k in set(a) | set(b)
            if k.startswith("mean_") and isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float))
        }
    )
    deltas = {k: round(float(b[k]) - float(a[k]), 4) for k in keys}
    return {
        "left": str(left),
        "right": str(right),
        "n_left": a.get("n_queries"),
        "n_right": b.get("n_queries"),
        "deltas_right_minus_left": deltas,
        "left_means": {k: a[k] for k in keys},
        "right_means": {k: b[k] for k in keys},
    }
