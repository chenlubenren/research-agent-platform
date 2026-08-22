from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urljoin

import httpx

from ..artifacts.store import WORKSPACE_DIRS


CLOUD_STATE_PATH = PurePosixPath("Content/CLOUD_SYNC.json")
SHARE_LINK_ENDPOINT_UNAVAILABLE = {404, 405, 501}


@dataclass
class SeafileSyncResult:
    status: str
    remote_path: str
    repo_id: str = ""
    share_url: str = ""
    preview_url: str = ""
    download_url: str = ""
    synced_files: int = 0
    uploaded_files: int = 0
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class SeafileWorkspaceSync:
    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        api_token: str = "",
        username: str = "",
        password: str = "",
        repo_id: str = "",
        repo_name: str = "Research Agent",
        remote_root: str = "research-agent",
        create_share_links: bool = True,
        share_password: str = "",
        timeout_seconds: float = 120.0,
        sync_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token.strip()
        self.username = username.strip()
        self.password = password
        self.repo_id = repo_id.strip()
        self.repo_name = repo_name.strip() or "Research Agent"
        self.remote_root = remote_root.strip("/ ") or "research-agent"
        self.create_share_links = create_share_links
        self.share_password = share_password
        self.timeout_seconds = timeout_seconds
        self.sync_retries = max(1, sync_retries)
        self.transport = transport
        self._locks: dict[str, asyncio.Lock] = {}

    def configuration_status(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "configured": bool(self.base_url and (self.api_token or (self.username and self.password))),
                "auth_mode": "token" if self.api_token else "username_password" if self.username and self.password else "missing",
                "hint": "设置 CLOUD_SYNC_ENABLED=true 后重启服务即可启用 Seafile 增量同步。",
            }
        if not self.base_url:
            return {
                "enabled": True,
                "configured": False,
                "auth_mode": "missing",
                "hint": "已启用云盘同步，但缺少 SEAFILE_BASE_URL。",
            }
        if self.api_token:
            return {
                "enabled": True,
                "configured": True,
                "auth_mode": "token",
                "hint": "Seafile API Token 已配置，首次同步时会自动创建或复用远程库。",
            }
        if self.username and self.password:
            return {
                "enabled": True,
                "configured": True,
                "auth_mode": "username_password",
                "hint": "Seafile 账号密码已配置，将在首次同步时换取临时 API Token。",
            }
        return {
            "enabled": True,
            "configured": False,
            "auth_mode": "missing",
            "hint": "已启用云盘同步，但缺少 SEAFILE_API_TOKEN 或账号密码。",
        }

    def remote_session_path(self, user_id: str, session_id: str) -> str:
        return "/" + PurePosixPath(
            self.remote_root,
            self._slug(user_id),
            self._slug(session_id),
        ).as_posix()

    async def sync_workspace(
        self,
        workspace_root: Path,
        *,
        user_id: str,
        session_id: str,
    ) -> SeafileSyncResult:
        remote_path = self.remote_session_path(user_id, session_id)
        if not self.enabled:
            return SeafileSyncResult(status="disabled", remote_path=remote_path)
        if not self.base_url:
            return SeafileSyncResult(
                status="error",
                remote_path=remote_path,
                error="SEAFILE_BASE_URL is required when cloud sync is enabled.",
            )

        lock = self._locks.setdefault(f"{user_id}/{session_id}", asyncio.Lock())
        async with lock:
            for attempt in range(self.sync_retries):
                try:
                    return await self._sync_locked(
                        workspace_root,
                        user_id=user_id,
                        session_id=session_id,
                        remote_path=remote_path,
                    )
                except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                    if attempt + 1 >= self.sync_retries:
                        return SeafileSyncResult(
                            status="error",
                            remote_path=remote_path,
                            repo_id=self.repo_id,
                            error=f"{exc.__class__.__name__}: {str(exc).strip()}",
                        )
                    await asyncio.sleep(0.4 * (attempt + 1))
                except Exception as exc:
                    return SeafileSyncResult(
                        status="error",
                        remote_path=remote_path,
                        repo_id=self.repo_id,
                        error=f"{exc.__class__.__name__}: {str(exc).strip()}",
                    )
            raise RuntimeError("Seafile sync exhausted its retry budget.")

    async def _sync_locked(
        self,
        workspace_root: Path,
        *,
        user_id: str,
        session_id: str,
        remote_path: str,
    ) -> SeafileSyncResult:
        workspace_root.mkdir(parents=True, exist_ok=True)
        headers = await self._auth_headers()
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            headers=headers,
            transport=self.transport,
        ) as client:
            repo_id = await self._resolve_repo_id(client)
            self.repo_id = repo_id
            await self._ensure_directory(client, repo_id, remote_path)
            for directory in WORKSPACE_DIRS:
                await self._ensure_directory(client, repo_id, f"{remote_path}/{directory}")

            previous = self._read_local_state(workspace_root)
            previous_signatures = previous.get("file_signatures", {})
            signatures: dict[str, str] = {}
            uploaded_files = 0
            files = [
                path
                for path in sorted(workspace_root.rglob("*"))
                if path.is_file()
                and path.relative_to(workspace_root).as_posix() != CLOUD_STATE_PATH.as_posix()
            ]
            for path in files:
                relative = path.relative_to(workspace_root).as_posix()
                signature = self._file_signature(path)
                signatures[relative] = signature
                if previous_signatures.get(relative) == signature:
                    continue
                parent = PurePosixPath(remote_path, relative).parent.as_posix()
                await self._ensure_directory(client, repo_id, parent)
                await self._upload_file(client, repo_id, parent, path)
                uploaded_files += 1

            share_url = ""
            share_link_error = ""
            if self.create_share_links:
                try:
                    share_url = await self._get_or_create_share_link(client, repo_id, remote_path)
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code if exc.response is not None else 0
                    if status_code not in SHARE_LINK_ENDPOINT_UNAVAILABLE:
                        raise
                    # Some institutional Seafile deployments disable the public
                    # share-link API while keeping normal file sync available.
                    # Do not turn a successfully mirrored workspace into a
                    # failed sync, and never expose the failed endpoint as a
                    # user-facing URL.
                    share_link_error = (
                        "文件已同步，但清华网盘未提供公开分享链接 "
                        f"（分享接口 HTTP {status_code}）。请登录网盘后按工作区路径访问。"
                    )
            result = SeafileSyncResult(
                status="synced",
                remote_path=remote_path,
                repo_id=repo_id,
                share_url=share_url,
                preview_url=share_url,
                download_url=share_url,
                synced_files=len(files),
                uploaded_files=uploaded_files,
                error=share_link_error,
            )
            self._write_local_state(
                workspace_root,
                {
                    **asdict(result),
                    "user_id": user_id,
                    "session_id": session_id,
                    "file_signatures": signatures,
                },
            )
            return result

    async def _auth_headers(self) -> dict[str, str]:
        if self.api_token:
            return {"Authorization": f"Token {self.api_token}"}
        if not self.username or not self.password:
            raise ValueError(
                "Set SEAFILE_API_TOKEN or both SEAFILE_USERNAME and SEAFILE_PASSWORD."
            )
        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = await client.post(
                f"{self.base_url}/api2/auth-token/",
                data={"username": self.username, "password": self.password},
            )
            response.raise_for_status()
            token = str(response.json().get("token", "")).strip()
            if not token:
                raise RuntimeError("Seafile authentication did not return an API token.")
            self.api_token = token
            return {"Authorization": f"Token {token}"}

    async def _resolve_repo_id(self, client: httpx.AsyncClient) -> str:
        if self.repo_id:
            return self.repo_id
        response = await client.get(f"{self.base_url}/api2/repos/")
        response.raise_for_status()
        repos = response.json()
        if isinstance(repos, dict):
            repos = repos.get("repos") or repos.get("data") or []
        for repo in repos if isinstance(repos, list) else []:
            if isinstance(repo, dict) and str(repo.get("name", "")) == self.repo_name:
                repo_id = str(repo.get("id") or repo.get("repo_id") or "")
                if repo_id:
                    return repo_id
        response = await client.post(
            f"{self.base_url}/api2/repos/",
            data={"name": self.repo_name, "desc": "Research Agent workspace backups"},
        )
        response.raise_for_status()
        payload = response.json()
        repo_id = str(payload.get("repo_id") or payload.get("id") or "")
        if not repo_id:
            raise RuntimeError("Seafile repository creation did not return a repository id.")
        return repo_id

    async def _ensure_directory(
        self,
        client: httpx.AsyncClient,
        repo_id: str,
        remote_path: str,
    ) -> None:
        current = PurePosixPath("/")
        for part in PurePosixPath(remote_path).parts:
            if part == "/":
                continue
            current /= part
            path = current.as_posix()
            check = await client.get(
                f"{self.base_url}/api2/repos/{repo_id}/dir/",
                params={"p": path},
            )
            if check.status_code == 200:
                continue
            if check.status_code not in {400, 404}:
                check.raise_for_status()
            create = await client.post(
                f"{self.base_url}/api2/repos/{repo_id}/dir/",
                params={"p": path},
                data={"operation": "mkdir"},
            )
            if create.status_code not in {200, 201}:
                body = create.text.lower()
                if create.status_code != 400 or "exist" not in body:
                    create.raise_for_status()

    async def _upload_file(
        self,
        client: httpx.AsyncClient,
        repo_id: str,
        parent_dir: str,
        path: Path,
    ) -> None:
        response = await client.get(
            f"{self.base_url}/api2/repos/{repo_id}/upload-link/",
            params={"p": parent_dir},
        )
        response.raise_for_status()
        try:
            upload_link = response.json()
        except json.JSONDecodeError:
            upload_link = response.text.strip().strip('"')
        if isinstance(upload_link, dict):
            upload_link = upload_link.get("url") or upload_link.get("upload_link") or ""
        upload_url = urljoin(f"{self.base_url}/", str(upload_link))
        if not upload_link:
            raise RuntimeError("Seafile did not return an upload link.")
        with path.open("rb") as source:
            upload = await client.post(
                upload_url,
                data={"parent_dir": parent_dir, "replace": "1"},
                files={"file": (path.name, source, "application/octet-stream")},
            )
        upload.raise_for_status()

    async def _get_or_create_share_link(
        self,
        client: httpx.AsyncClient,
        repo_id: str,
        remote_path: str,
    ) -> str:
        endpoint = f"{self.base_url}/api/v2.1/share-links/"
        response = await client.get(endpoint, params={"repo_id": repo_id, "path": remote_path})
        if response.status_code == 200:
            existing = self._share_link_from_payload(response.json())
            if existing:
                return existing
        data = {"repo_id": repo_id, "path": remote_path}
        if self.share_password:
            data["password"] = self.share_password
        response = await client.post(endpoint, data=data)
        response.raise_for_status()
        link = self._share_link_from_payload(response.json())
        if not link:
            raise RuntimeError("Seafile share-link creation did not return a link.")
        return link

    def _share_link_from_payload(self, payload: Any) -> str:
        if isinstance(payload, list):
            for item in payload:
                link = self._share_link_from_payload(item)
                if link:
                    return link
            return ""
        if not isinstance(payload, dict):
            return ""
        for key in ("link", "url", "share_link"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return urljoin(f"{self.base_url}/", value.strip())
        data = payload.get("data")
        return self._share_link_from_payload(data)

    def _read_local_state(self, workspace_root: Path) -> dict[str, Any]:
        path = workspace_root / Path(*CLOUD_STATE_PATH.parts)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_local_state(self, workspace_root: Path, payload: dict[str, Any]) -> None:
        path = workspace_root / Path(*CLOUD_STATE_PATH.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _file_signature(self, path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _slug(self, value: str) -> str:
        cleaned = "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in value.strip()
        )
        return cleaned or "local"
