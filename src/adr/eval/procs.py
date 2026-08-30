"""Running the upstream judge scripts as subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

TAIL = 2_000


def run_script(
    args: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Invoke ``python -u <args>`` and capture a summary of the outcome."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    cmd = [sys.executable, "-u", *[str(a) for a in args]]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "returncode": None, "timeout": True, "ok": False}
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout_tail": completed.stdout[-TAIL:],
        "stderr_tail": completed.stderr[-TAIL:],
    }


def stage_relative_dir(staging_root: Path, relative: str, target: Path) -> Path:
    """Expose ``target`` at ``staging_root/relative`` via a symlink.

    ``eval_kpr_async.py`` reads key points from a path relative to its working
    directory, so we build a working directory that satisfies it instead of
    editing the upstream checkout.
    """
    link = staging_root / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            return link
    link.symlink_to(target, target_is_directory=True)
    return link
