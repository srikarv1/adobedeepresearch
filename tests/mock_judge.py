"""A tiny OpenAI-compatible server that stands in for a judge model.

Lets the official DeepResearchGym scripts run end to end with no API key and no
network, so the wiring and the score aggregation can be checked offline. Replies
are fixed per criterion, which makes the expected aggregate exactly predictable.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Chosen so the normalized quality score is a round number:
# (7+6+5+8+4+6) / (6*10) * 100 = 60.0
CRITERION_RATINGS = {
    "Clarity": 7,
    "Depth": 6,
    "Balance": 5,
    "Breadth": 8,
    "Support": 4,
    "Insightfulness": 6,
}
EXPECTED_QUALITY = 60.0

KPR_LABEL = "Supported"
EXPECTED_SUPPORT_RATE = 100.0


def _criterion_from_prompt(prompt: str) -> str | None:
    marker = "single criterion:"
    if marker not in prompt:
        return None
    tail = prompt.split(marker, 1)[1]
    for name in CRITERION_RATINGS:
        if tail.lstrip().startswith(name):
            return name
    return None


def _reply_for(schema_name: str, prompt: str) -> dict[str, Any]:
    if schema_name == "KeyPointRecall":
        return {"label": KPR_LABEL, "justification": "mock judge"}
    criterion = _criterion_from_prompt(prompt)
    rating = CRITERION_RATINGS.get(criterion or "", 5)
    # Mirrors the one hard rule in the real Support rubric, so a demo can show
    # the pipeline discriminating between a cited and an uncited report.
    if criterion == "Support" and "http" not in prompt:
        rating = 0
    return {"rating": rating, "justification": f"mock judge for {criterion}"}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # keep test output clean
        return

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")

        prompt = "\n".join(
            str(m.get("content") or "") for m in body.get("messages") or []
        )
        schema_name = (
            ((body.get("response_format") or {}).get("json_schema") or {}).get("name") or ""
        )
        content = json.dumps(_reply_for(schema_name, prompt))

        self.server.request_count += 1  # type: ignore[attr-defined]
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model") or "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "refusal": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class MockJudgeServer:
    """Context manager exposing ``base_url`` for OPENAI_BASE_URL."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.request_count = 0  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def request_count(self) -> int:
        return self._server.request_count  # type: ignore[attr-defined]

    def __enter__(self) -> MockJudgeServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
