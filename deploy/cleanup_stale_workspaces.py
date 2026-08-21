"""Remove session workspaces that have not been accessed for a configured TTL.

Access is tracked by a small ``.last_access`` marker written by the running
APIs.  For legacy workspaces without a marker, the newest file/directory mtime
is used as a conservative fallback.  The script accepts one or more explicit
workspace roots and never traverses outside those roots.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


MARKER = ".last_access"


def workspace_last_access(path: Path) -> float:
    marker = path / MARKER
    try:
        return float(marker.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        newest = path.stat().st_mtime
        for child in path.rglob("*"):
            try:
                newest = max(newest, child.stat().st_mtime)
            except OSError:
                continue
        return newest


def cleanup(root: Path, *, ttl_hours: float, now: float | None = None, dry_run: bool = False) -> dict[str, object]:
    root = root.resolve()
    current = time.time() if now is None else float(now)
    cutoff = current - max(0.0, float(ttl_hours)) * 3600.0
    removed: list[str] = []
    retained: list[str] = []
    if not root.is_dir():
        return {"root": str(root), "cutoff": cutoff, "removed": removed, "retained": retained, "missing": True}
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        last_access = workspace_last_access(candidate)
        relative = candidate.relative_to(root).as_posix()
        if last_access < cutoff:
            removed.append(relative)
            if not dry_run:
                shutil.rmtree(candidate)
        else:
            retained.append(relative)
    return {
        "root": str(root),
        "cutoff": cutoff,
        "ttl_hours": float(ttl_hours),
        "dry_run": dry_run,
        "removed": removed,
        "retained": retained,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, help="Explicit session workspace root; repeatable.")
    parser.add_argument("--ttl-hours", type=float, default=168.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = [cleanup(Path(root), ttl_hours=args.ttl_hours, dry_run=args.dry_run) for root in args.root]
    print(json.dumps({"roots": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
