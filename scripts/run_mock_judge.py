"""Local OpenAI-compatible judge for dry-run scoring. Not a real evaluator."""

from __future__ import annotations

import argparse
import time

from tests.mock_judge import MockJudgeServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    with MockJudgeServer(host=args.host, port=args.port) as server:
        print(f"mock judge listening on {server.base_url}", flush=True)
        print("export OPENAI_API_KEY=mock-key", flush=True)
        print(f"export OPENAI_BASE_URL={server.base_url}", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
