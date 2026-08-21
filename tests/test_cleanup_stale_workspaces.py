from __future__ import annotations

import os
import time
from pathlib import Path

from deploy.cleanup_stale_workspaces import cleanup


def test_cleanup_removes_only_workspaces_older_than_ttl(tmp_path: Path) -> None:
    old = tmp_path / "old"
    fresh = tmp_path / "fresh"
    old.mkdir(); fresh.mkdir()
    now = 2_000_000_000.0
    (old / ".last_access").write_text(str(now - 8 * 24 * 3600), encoding="ascii")
    (fresh / ".last_access").write_text(str(now - 2 * 24 * 3600), encoding="ascii")

    report = cleanup(tmp_path, ttl_hours=168, now=now)

    assert report["removed"] == ["old"]
    assert report["retained"] == ["fresh"]
    assert not old.exists()
    assert fresh.exists()


def test_cleanup_dry_run_does_not_delete(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    old_mtime = time.time() - 10 * 24 * 3600
    os.utime(old, (old_mtime, old_mtime))

    report = cleanup(tmp_path, ttl_hours=168, now=time.time(), dry_run=True)

    assert report["removed"] == ["old"]
    assert old.exists()
