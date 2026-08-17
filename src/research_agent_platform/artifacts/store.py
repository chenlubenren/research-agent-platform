from __future__ import annotations

from pathlib import Path

from ..models import ArtifactKind, ArtifactRecord


WORKSPACE_DIRS = (
    "bib",
    "plan",
    "idea",
    "code",
    "figures",
    "paper",
    "presentation",
    "rebuttal",
    "wiki",
    "Content",
    "logs",
)

LOCAL_WORKSPACE_USER = "local"


class ArtifactStore:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def task_root(
        self,
        task_id: str,
        *,
        user_id: str = "local",
        session_id: str | None = None,
    ) -> Path:
        if not session_id:
            raise ValueError("session_id is required; task-specific workspace roots are not allowed")
        return self.session_root(user_id=user_id, session_id=session_id)

    def session_root(self, *, user_id: str = "local", session_id: str) -> Path:
        path = self.root / LOCAL_WORKSPACE_USER / self._slug(session_id)
        path.mkdir(parents=True, exist_ok=True)
        self.ensure_workspace(path)
        return path

    def ensure_workspace(self, task_root: Path) -> None:
        for name in WORKSPACE_DIRS:
            (task_root / name).mkdir(parents=True, exist_ok=True)

    def write_text(
        self,
        task_id: str,
        relative_path: str,
        content: str,
        *,
        kind: ArtifactKind,
        description: str,
        task_root: str | Path | None = None,
    ) -> ArtifactRecord:
        root = self._resolve_root(task_id, task_root)
        target = self._target_path(root, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        clean_relative_path = relative_path.replace("\\", "/")
        artifact = ArtifactRecord(
            name=target.name,
            kind=kind,
            relative_path=clean_relative_path,
            absolute_path=str(target.resolve()),
            url_path=self.url_for(root, clean_relative_path),
            description=description,
        )
        self._write_manifest_for_root(root)
        return artifact

    def write_bytes(
        self,
        task_id: str,
        relative_path: str,
        content: bytes,
        *,
        kind: ArtifactKind,
        description: str,
        task_root: str | Path | None = None,
    ) -> ArtifactRecord:
        root = self._resolve_root(task_id, task_root)
        target = self._target_path(root, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        clean_relative_path = relative_path.replace("\\", "/")
        artifact = ArtifactRecord(
            name=target.name,
            kind=kind,
            relative_path=clean_relative_path,
            absolute_path=str(target.resolve()),
            url_path=self.url_for(root, clean_relative_path),
            description=description,
        )
        self._write_manifest_for_root(root)
        return artifact

    def record_existing(
        self,
        task_id: str,
        relative_path: str,
        *,
        kind: ArtifactKind,
        description: str,
        task_root: str | Path | None = None,
    ) -> ArtifactRecord:
        root = self._resolve_root(task_id, task_root)
        target = self._target_path(root, relative_path)
        if not target.is_file():
            raise FileNotFoundError(target)
        clean_relative_path = relative_path.replace("\\", "/")
        artifact = ArtifactRecord(
            name=target.name,
            kind=kind,
            relative_path=clean_relative_path,
            absolute_path=str(target.resolve()),
            url_path=self.url_for(root, clean_relative_path),
            description=description,
        )
        self._write_manifest_for_root(root)
        return artifact

    def _write_manifest_for_root(self, task_root: Path) -> None:
        manifest = task_root / "Content" / "MANIFEST.md"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        entries: list[str] = ["# Artifact Manifest", ""]
        for path in sorted(p for p in task_root.rglob("*") if p.is_file() and p != manifest):
            entries.append(f"- `{path.relative_to(task_root).as_posix()}`")
        manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")

    def url_for(self, task_root: str | Path, relative_path: str) -> str:
        root = Path(task_root).resolve()
        try:
            root_relative = root.relative_to(self.root.resolve()).as_posix()
            return f"/workspace-files/{root_relative}/{relative_path.replace('\\', '/')}"
        except ValueError:
            task_id = root.name
            return f"/workspace-files/{task_id}/{relative_path.replace('\\', '/')}"

    def _resolve_root(self, task_id: str, task_root: str | Path | None) -> Path:
        if task_root:
            root = Path(task_root)
            root.mkdir(parents=True, exist_ok=True)
            return root
        raise ValueError("task_root is required; artifacts must use a session workspace")

    def _target_path(self, root: Path, relative_path: str) -> Path:
        normalized = relative_path.replace("\\", "/").lstrip("/")
        parts = Path(normalized).parts
        if not parts or parts[0] not in WORKSPACE_DIRS:
            raise ValueError(f"Artifact path must start with a standard workspace directory: {relative_path}")
        target = (root / Path(*parts)).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes the session workspace: {relative_path}") from exc
        return target

    def _slug(self, value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
        return cleaned or "local"
