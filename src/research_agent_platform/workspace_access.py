"""Access markers used by stale-session workspace cleanup."""

from __future__ import annotations

import os
import time
from pathlib import Path


MARKER_NAME = ".last_access"


def touch_workspace_access(root: str | Path) -> None:
    path = Path(root)
    if not path.is_dir():
        return
    marker = path / MARKER_NAME
    try:
        marker.touch(exist_ok=True)
        os.utime(marker, (time.time(), time.time()))
    except OSError:
        # Access tracking must never make a research request fail.
        return
