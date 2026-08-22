from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import ResearchAgentService
from .config import config
from .models import UploadBatchRecord
from .upstream import extract_text_from_message
from .upstream import list_models as upstream_list_models
from .uploads import (
    classify_upload,
    next_upload_relative_path,
    normalize_upload_target,
    sanitize_upload_filename,
    upload_artifact_kind,
)
from .router.intent import has_execution_intent, is_informational_question


CHAT_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Research Agent</title>
  <style>
    :root { color-scheme:light; --bg:#f5f5f4; --surface:#fff; --ink:#202124; --muted:#70757a; --line:#dedede; --soft:#f0f1f2; --accent:#2457d6; --warn:#9b4b18; --error:#a12d2d; }
    * { box-sizing:border-box; }
    html, body { height:100%; }
    body { margin:0; font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--ink); }
    button, textarea, input { font:inherit; }
    button { letter-spacing:0; }
    [hidden] { display:none !important; }
    .wrap { width:min(100%, 960px); height:100dvh; margin:0 auto; padding:0 18px; }
    .card { height:100%; min-height:0; display:flex; flex-direction:column; background:var(--surface); border-inline:1px solid var(--line); overflow:hidden; }
    .topbar { min-height:58px; padding:10px 18px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:14px; background:rgba(255,255,255,.96); }
    .brand { flex:0 0 auto; font-size:16px; font-weight:650; }
    .session-meta { min-width:0; flex:1; display:flex; align-items:center; gap:10px; }
    .session-input { min-width:0; width:100%; max-width:280px; padding:5px 0; border:0; outline:0; background:transparent; color:var(--muted); font-size:12px; text-overflow:ellipsis; }
    .task-status { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--muted); font-size:12px; }
    .header-actions { display:flex; align-items:center; gap:8px; }
    .button { border:1px solid transparent; border-radius:7px; min-height:36px; padding:8px 12px; cursor:pointer; background:transparent; color:var(--ink); }
    .button:disabled { cursor:not-allowed; opacity:.55; }
    .button:focus-visible, .composer-action:focus-visible { outline:3px solid rgba(36,87,214,.2); outline-offset:2px; }
    .primary { background:var(--accent); color:#fff; }
    .secondary { border-color:var(--line); background:#fff; }
    .icon-button { width:36px; padding:0; display:grid; place-items:center; font-size:18px; }
    .chat { min-height:0; flex:1; overflow:auto; padding:26px clamp(18px,5vw,58px); display:flex; flex-direction:column; gap:12px; scroll-behavior:smooth; }
    .empty-state { margin:auto; color:#9aa0a6; font-size:14px; }
    .msg { max-width:78%; padding:11px 13px; border-radius:8px; white-space:pre-wrap; line-height:1.55; overflow-wrap:anywhere; }
    .user { align-self:flex-end; background:var(--accent); color:#fff; }
    .assistant { align-self:flex-start; border:1px solid var(--line); background:#fff; }
    .system { align-self:flex-start; width:min(100%, 720px); max-width:92%; background:var(--soft); color:#666b70; font-size:13px; }
    .system-title { margin-bottom:7px; color:#4b4f53; font-weight:650; }
    .task-meta { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:#9aa0a6; flex:0 0 auto; }
    .status-dot.running { background:#5f7fc7; }
    .status-dot.completed { background:#5d8b68; }
    .status-dot.failed { background:#b45c5c; }
    .progress-lines { display:flex; flex-direction:column; gap:5px; }
    .progress-line { padding-left:12px; position:relative; }
    .progress-line::before { content:""; position:absolute; left:1px; top:.65em; width:4px; height:4px; border-radius:50%; background:#aeb2b6; }
    .artifact-list { display:flex; flex-direction:column; gap:7px; }
    .artifact-card { padding:9px 10px; border:1px solid #d9dcdf; border-radius:6px; background:#fff; }
    .artifact-card a { color:#2457d6; text-decoration:none; word-break:break-all; }
    .artifact-preview { display:block; max-width:100%; max-height:320px; margin-top:8px; border:1px solid var(--line); border-radius:4px; background:#fff; object-fit:contain; }
    .cloud-links { display:flex; gap:12px; flex-wrap:wrap; margin-top:7px; }
    .cloud-links a { color:#2457d6; text-decoration:none; }
    .checkpoint-message { background:#f2f2f1; color:#4d5156; }
    .checkpoint-text { margin-bottom:10px; }
    .feedback { width:100%; min-height:76px; resize:vertical; padding:10px 11px; border:1px solid #cfd2d4; border-radius:6px; background:#fff; color:var(--ink); }
    .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:9px; }
    .muted { color:var(--muted); font-size:12px; }
    .warning { color:var(--warn); }
    .composer-area { position:sticky; bottom:0; padding:10px clamp(18px,5vw,58px) 18px; border-top:1px solid var(--line); background:rgba(255,255,255,.97); }
    .upload-status { min-height:18px; margin-bottom:5px; color:var(--muted); font-size:12px; }
    .upload-status.error { color:var(--error); }
    .upload-list { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:7px; }
    .upload-list:empty { display:none; }
    .upload-item { max-width:100%; padding:4px 7px; border:1px solid var(--line); border-radius:5px; background:#f7f7f7; color:#5f6368; font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .composer { min-height:64px; display:grid; grid-template-columns:36px minmax(0,1fr) 38px; align-items:end; gap:7px; padding:8px; border:1px solid #c9cccf; border-radius:8px; background:#fff; transition:border-color .15s, box-shadow .15s, background .15s; }
    .composer:focus-within { border-color:#899cc9; box-shadow:0 0 0 3px rgba(36,87,214,.08); }
    .composer.dragging { border-color:var(--accent); box-shadow:0 0 0 3px rgba(36,87,214,.12); background:#f7f9ff; }
    .composer textarea { width:100%; min-height:46px; max-height:180px; resize:none; padding:7px 3px; border:0; outline:0; background:transparent; color:var(--ink); line-height:1.5; }
    .composer-action { width:36px; height:36px; border:0; border-radius:6px; display:grid; place-items:center; cursor:pointer; background:transparent; color:#5f6368; font-size:20px; }
    .composer-action:hover { background:#f0f1f2; }
    .send-action { background:var(--accent); color:#fff; }
    .send-action:hover { background:#1f4bb7; }
    @media (max-width:640px) {
      .wrap { padding:0; }
      .card { border:0; }
      .topbar { padding:9px 12px; gap:9px; }
      .session-input { display:none; }
      .task-status { max-width:110px; }
      .chat { padding:18px 12px; }
      .msg { max-width:90%; }
      .system { max-width:100%; }
      .composer-area { padding:8px 10px 12px; }
      .new-session-label { display:none; }
      #new-session { width:36px; padding:0; font-size:20px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <main class="card">
      <header class="topbar">
        <div class="brand">Research Agent</div>
        <div class="session-meta">
          <input id="session" class="session-input" placeholder="尚未创建会话" readonly>
          <span id="status" class="task-status">就绪</span>
        </div>
        <div class="header-actions">
          <button id="cloud-sync" class="button secondary icon-button" type="button" title="同步当前会话到云盘" aria-label="同步当前会话到云盘">&#8635;</button>
          <button id="new-session" class="button secondary" type="button" title="创建新会话"><span aria-hidden="true">+</span> <span class="new-session-label">新会话</span></button>
        </div>
      </header>
      <div id="chat" class="chat" aria-live="polite">
        <div id="empty-state" class="empty-state">开始新的研究会话</div>
      </div>
      <footer class="composer-area">
        <div id="upload-status" class="upload-status" aria-live="polite"></div>
        <div id="upload-list" class="upload-list"></div>
        <div id="composer" class="composer">
          <button id="attach-file" class="composer-action" type="button" title="选择文件" aria-label="选择文件">+</button>
          <input id="file-input" type="file" multiple hidden>
          <textarea id="prompt" rows="1" placeholder="输入问题或命令"></textarea>
          <button id="send" class="composer-action send-action" type="button" title="发送" aria-label="发送">&#8593;</button>
        </div>
      </footer>
    </main>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const prompt = document.getElementById("prompt");
    const composer = document.getElementById("composer");
    const attachFileBtn = document.getElementById("attach-file");
    const send = document.getElementById("send");
    const sessionInput = document.getElementById("session");
    const newSessionBtn = document.getElementById("new-session");
    const statusEl = document.getElementById("status");
    const cloudSyncBtn = document.getElementById("cloud-sync");
    const fileInput = document.getElementById("file-input");
    const uploadStatus = document.getElementById("upload-status");
    const uploadList = document.getElementById("upload-list");
    let currentTaskId = "";
    let currentCheckpoint = null;
    let uploadInProgress = false;
    let pollTimer = null;
    let taskEventSource = null;
    let renderedResponseTaskId = "";
    let dragDepth = 0;

    function removeEmptyState() {
      document.getElementById("empty-state")?.remove();
    }

    function renderMessage(role, content) {
      removeEmptyState();
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      div.textContent = content;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    function systemMessage(key, title) {
      removeEmptyState();
      let block = chat.querySelector(`[data-system-key="${key}"]`);
      if (!block) {
        block = document.createElement("div");
        block.className = "msg system";
        block.dataset.systemKey = key;
        chat.appendChild(block);
      }
      block.innerHTML = "";
      if (title) {
        const heading = document.createElement("div");
        heading.className = "system-title";
        heading.textContent = title;
        block.appendChild(heading);
      }
      return block;
    }

    function removeSystemMessage(key) {
      chat.querySelector(`[data-system-key="${key}"]`)?.remove();
    }

    function renderArtifacts(taskId, artifacts) {
      const key = `artifacts-${taskId}`;
      if (!artifacts || artifacts.length === 0) {
        removeSystemMessage(key);
        return;
      }
      const block = systemMessage(key, "最新产物");
      const list = document.createElement("div");
      list.className = "artifact-list";
      for (const artifact of artifacts) {
        const card = document.createElement("div");
        card.className = "artifact-card";
        const a = document.createElement("a");
        a.href = artifact.url_path;
        a.target = "_blank";
        a.textContent = `${artifact.relative_path} - ${artifact.description}`;
        card.appendChild(a);
        if (artifact.kind === "image" || /\\.(png|jpg|jpeg|webp|gif|svg)$/i.test(artifact.relative_path || "")) {
          const img = document.createElement("img");
          img.src = artifact.url_path;
          img.alt = artifact.relative_path;
          img.className = "artifact-preview";
          card.appendChild(img);
        }
        list.appendChild(card);
      }
      block.appendChild(list);
      chat.scrollTop = chat.scrollHeight;
    }

    function renderProgress(data) {
      if (!data.task_id) return;
      const block = systemMessage(`progress-${data.task_id}`, "任务进度");
      const meta = document.createElement("div");
      meta.className = "task-meta";
      const dot = document.createElement("span");
      dot.className = `status-dot ${data.status || ""}`;
      const summary = document.createElement("span");
      const stage = data.current_stage_name || data.workflow_title || "准备中";
      summary.textContent = `${data.command || "任务"} · ${data.status || "running"} · ${stage}`;
      meta.append(dot, summary);
      block.appendChild(meta);
      const progress = data.progress || [];
      const progressEvents = data.progress_events || [];
      const items = progressEvents && progressEvents.length
        ? progressEvents.map(item => `${item.kind || "progress"} · ${item.message}`)
        : progress;
      if (!items || items.length === 0) {
        const pending = document.createElement("div");
        pending.className = "muted";
        pending.textContent = "正在准备任务...";
        block.appendChild(pending);
        return;
      }
      const lines = document.createElement("div");
      lines.className = "progress-lines";
      for (const item of items.slice(-12)) {
        const div = document.createElement("div");
        div.className = "progress-line";
        div.textContent = item;
        lines.appendChild(div);
      }
      block.appendChild(lines);
      chat.scrollTop = chat.scrollHeight;
    }

    function renderCloudWorkspace(cloud) {
      if (!cloud || !cloud.status) return;
      if (cloud.status === "disabled") {
        removeSystemMessage("cloud-workspace");
        return;
      }
      const block = systemMessage("cloud-workspace", "云端工作区");
      const status = document.createElement("div");
      status.textContent = cloud.status === "error"
          ? `同步失败：${cloud.error || "未知错误"}`
          : `已同步 ${cloud.synced_files || 0} 个文件到 ${cloud.remote_path || "云盘"}`;
      block.appendChild(status);
      if (cloud.status !== "error" && cloud.error) {
        const warning = document.createElement("div");
        warning.className = "muted warning";
        warning.textContent = cloud.error;
        block.appendChild(warning);
      }
      const linksWrap = document.createElement("div");
      linksWrap.className = "cloud-links";
      const links = [
        ["预览文件夹", cloud.preview_url || cloud.share_url],
        ["下载文件夹", cloud.download_url || cloud.share_url]
      ];
      for (const [label, url] of links) {
        if (!url) continue;
        const link = document.createElement("a");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = label;
        linksWrap.appendChild(link);
      }
      if (linksWrap.childElementCount) block.appendChild(linksWrap);
      const directUrl = cloud.preview_url || cloud.download_url || cloud.share_url;
      if (directUrl) {
        const direct = document.createElement("div");
        direct.className = "cloud-direct-url";
        direct.textContent = `网盘链接：${directUrl}`;
        block.appendChild(direct);
      }
      chat.scrollTop = chat.scrollHeight;
    }

    async function syncCloudWorkspace() {
      const sessionId = sessionInput.value.trim();
      if (!sessionId) {
        renderMessage("system", "请先创建会话或上传文件，再同步云端工作区。");
        return;
      }
      cloudSyncBtn.disabled = true;
      const block = systemMessage("cloud-workspace", "云端工作区");
      block.append("正在同步...");
      try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/sync`, { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "云盘同步失败");
        renderCloudWorkspace(data.cloud_workspace || {});
      } catch (error) {
        const errorBlock = systemMessage("cloud-workspace", "云端工作区");
        errorBlock.append(error.message || "云盘同步失败");
      } finally {
        cloudSyncBtn.disabled = false;
      }
    }

    function renderCheckpoint(checkpoint) {
      const key = currentTaskId ? `checkpoint-${currentTaskId}` : "checkpoint";
      if (!checkpoint) {
        removeSystemMessage(key);
        return;
      }
      const block = systemMessage(key, checkpoint.title || "需要确认");
      block.classList.add("checkpoint-message");
      const text = document.createElement("div");
      text.className = "checkpoint-text";
      text.textContent = checkpoint.prompt || "请确认后继续。";
      const feedback = document.createElement("textarea");
      feedback.className = "feedback";
      feedback.placeholder = "可填写修改意见；直接批准时可留空";
      const actions = document.createElement("div");
      actions.className = "actions";
      const approve = document.createElement("button");
      approve.type = "button";
      approve.className = "button primary";
      approve.textContent = "批准继续";
      approve.addEventListener("click", () => approveTask(feedback, approve, revise));
      const revise = document.createElement("button");
      revise.type = "button";
      revise.className = "button secondary";
      revise.textContent = "发送修改意见";
      revise.addEventListener("click", () => reviseTask(feedback, approve, revise, text));
      actions.append(approve, revise);
      block.append(text, feedback, actions);
      chat.scrollTop = chat.scrollHeight;
    }

    cloudSyncBtn.addEventListener("click", syncCloudWorkspace);

    async function createNewSession() {
      newSessionBtn.disabled = true;
      stopTaskPoll();
      stopTaskStream();
      try {
        const response = await fetch("/api/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: "local" })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `创建会话失败 (${response.status})`);
        currentTaskId = "";
        currentCheckpoint = null;
        sessionInput.value = data.session_id;
        localStorage.setItem("research-agent-session", data.session_id);
        localStorage.removeItem("research-agent-task");
        chat.innerHTML = "";
        uploadList.innerHTML = "";
        uploadStatus.classList.remove("error");
        uploadStatus.textContent = "";
        statusEl.textContent = "就绪";
        renderCloudWorkspace(data.cloud_workspace || {});
        const cloudUrl = (data.cloud_workspace || {}).preview_url || (data.cloud_workspace || {}).download_url || (data.cloud_workspace || {}).share_url || "";
        renderMessage("assistant", cloudUrl
          ? `新会话已创建：${data.session_id}\n清华网盘工作区链接：${cloudUrl}`
          : `新会话已创建：${data.session_id}`);
        prompt.focus();
      } catch (error) {
        renderMessage("assistant", String(error.message || error));
      } finally {
        newSessionBtn.disabled = false;
      }
    }

    function scheduleTaskPoll() {
      if (pollTimer || !currentTaskId) return;
      pollTimer = window.setTimeout(pollTask, 1500);
    }

    function stopTaskPoll() {
      if (!pollTimer) return;
      window.clearTimeout(pollTimer);
      pollTimer = null;
    }

    function stopTaskStream() {
      if (!taskEventSource) return;
      taskEventSource.close();
      taskEventSource = null;
    }

    function subscribeTaskStream(taskId) {
      stopTaskStream();
      if (!taskId || !window.EventSource) {
        scheduleTaskPoll();
        return;
      }
      taskEventSource = new EventSource(`/api/tasks/${encodeURIComponent(taskId)}/events`);
      const handlePacket = event => {
        try {
          const packet = JSON.parse(event.data || "{}");
          if (packet.task && packet.task.task_id === currentTaskId) {
            renderState(packet.task, true);
            if (packet.task.response_text && event.type === "done") {
              if (renderedResponseTaskId !== packet.task.task_id) {
                renderedResponseTaskId = packet.task.task_id;
                renderMessage("assistant", packet.task.response_text);
              }
            }
          }
          if (["done", "error"].includes(event.type)) {
            stopTaskStream();
          }
        } catch (error) {
          renderMessage("system", `实时进度解析失败：${String(error.message || error)}`);
        }
      };
      for (const eventName of ["snapshot", "progress", "done", "error"]) {
        taskEventSource.addEventListener(eventName, handlePacket);
      }
      taskEventSource.onerror = () => {
        if (!taskEventSource) return;
        stopTaskStream();
        renderMessage("system", "实时进度连接中断，已切换为状态轮询。" );
        scheduleTaskPoll();
      };
    }

    async function pollTask() {
      pollTimer = null;
      if (!currentTaskId) return;
      const taskId = currentTaskId;
      try {
        const response = await fetch(`/api/tasks/${taskId}`, { cache: "no-store" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `获取任务状态失败 (${response.status})`);
        if (taskId === currentTaskId) renderState(data);
      } catch (error) {
        renderMessage("system", `进度刷新失败：${String(error.message || error)}`);
        scheduleTaskPoll();
      }
    }

    function renderState(data, preserveStream = false) {
      currentTaskId = data.task_id || "";
      currentCheckpoint = data.checkpoint || null;
      if (data.session_id) {
        sessionInput.value = data.session_id;
        localStorage.setItem("research-agent-session", data.session_id);
      }
      if (currentTaskId) localStorage.setItem("research-agent-task", currentTaskId);
      statusEl.textContent = currentTaskId
        ? `${data.command || "任务"} · ${data.status || "running"}`
        : "就绪";
      renderProgress(data);
      renderArtifacts(currentTaskId, data.artifacts || []);
      renderCloudWorkspace(data.cloud_workspace || {});
      renderCheckpoint(currentCheckpoint);
      if (data.response_text && data.status === "completed" && renderedResponseTaskId !== data.task_id) {
        renderedResponseTaskId = data.task_id;
        renderMessage("assistant", data.response_text);
      }
      if (data.status === "running") {
        if (!taskEventSource) subscribeTaskStream(currentTaskId);
      } else if (!preserveStream) {
        stopTaskPoll();
        stopTaskStream();
      }
    }

    async function restoreTaskState() {
      const savedSession = localStorage.getItem("research-agent-session");
      const savedTask = localStorage.getItem("research-agent-task");
      if (savedSession) sessionInput.value = savedSession;
      try {
        if (savedTask) {
          const response = await fetch(`/api/tasks/${savedTask}`, { cache: "no-store" });
          if (response.ok) {
            renderState(await response.json());
            return;
          }
        }
        const response = await fetch("/api/tasks", { cache: "no-store" });
        if (!response.ok) return;
        const tasks = await response.json();
        const active = tasks.find(task => task.user_id === "local" && ["running", "waiting_human"].includes(task.status));
        if (active) {
          const detailResponse = await fetch(`/api/tasks/${active.task_id}`, { cache: "no-store" });
          if (detailResponse.ok) renderState(await detailResponse.json());
        }
      } catch (error) {
        renderMessage("system", `恢复任务状态失败：${String(error.message || error)}`);
      }
    }

    function formatFileSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function showUploadedFiles(files) {
      uploadList.innerHTML = "";
      for (const file of files) {
        const item = document.createElement("span");
        item.className = "upload-item";
        item.textContent = `${file.relative_path} (${formatFileSize(file.size)})`;
        uploadList.appendChild(item);
      }
    }

    async function uploadFiles(fileList) {
      const files = Array.from(fileList || []).filter(file => file.size > 0);
      if (!files.length || uploadInProgress) return;
      uploadInProgress = true;
      composer.classList.remove("dragging");
      uploadStatus.classList.remove("error");
      uploadStatus.textContent = `正在上传 ${files.length} 个文件...`;
      const formData = new FormData();
      if (sessionInput.value) formData.append("session_id", sessionInput.value);
      formData.append("user_id", "local");
      formData.append("target", "auto");
      for (const file of files) formData.append("files", file, file.name);
      try {
        const response = await fetch("/api/session/files", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `上传失败 (${response.status})`);
        sessionInput.value = data.session_id;
        localStorage.setItem("research-agent-session", data.session_id);
        uploadStatus.textContent = `已上传 ${data.files.length} 个文件`;
        showUploadedFiles(data.files);
        renderCloudWorkspace(data.cloud_workspace || {});
        const names = data.files.map(file => file.relative_path).join("\\n");
        renderMessage("system", `文件已加入当前会话：\\n${names}`);
      } catch (error) {
        uploadStatus.classList.add("error");
        uploadStatus.textContent = String(error.message || error);
      } finally {
        uploadInProgress = false;
        fileInput.value = "";
      }
    }

    async function submit() {
      const text = prompt.value.trim();
      if (!text) return;
      renderMessage("user", text);
      prompt.value = "";
      resizePrompt();
      send.disabled = true;
      try {
        const response = await fetch("/api/agent/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionInput.value || null, message: text })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`);
        renderState(data);
        if (!data.task_id) {
          renderMessage("assistant", data.text || "No response");
        } else if (data.text && data.status === "running") {
          renderMessage("assistant", data.text);
        }
        if (data.task_id && data.status === "running") subscribeTaskStream(data.task_id);
      } catch (error) {
        renderMessage("assistant", String(error.message || error));
      } finally {
        send.disabled = false;
      }
    }

    async function approveTask(feedbackEl, approveBtn, reviseBtn) {
      if (!currentTaskId) return;
      approveBtn.disabled = true;
      reviseBtn.disabled = true;
      try {
        const response = await fetch(`/api/tasks/${currentTaskId}/approve`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ feedback: feedbackEl.value.trim() })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `审批失败 (${response.status})`);
        renderMessage("system", data.text || "已批准，正在后台继续生成。" );
        feedbackEl.value = "";
        renderState(data);
      } catch (error) {
        renderMessage("assistant", String(error.message || error));
      } finally {
        approveBtn.disabled = false;
        reviseBtn.disabled = false;
      }
    }

    async function reviseTask(feedbackEl, approveBtn, reviseBtn, checkpointText) {
      if (!currentTaskId) return;
      const feedback = feedbackEl.value.trim();
      if (!feedback) {
        checkpointText.textContent = "请先填写修改意见，再点击“发送修改意见”。";
        checkpointText.classList.add("warning");
        return;
      }
      renderMessage("user", `修改意见：${feedback}`);
      approveBtn.disabled = true;
      reviseBtn.disabled = true;
      try {
        const response = await fetch(`/api/tasks/${currentTaskId}/reject`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ feedback })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `发送修改意见失败 (${response.status})`);
        renderMessage("system", data.text || "修改意见已发送，正在后台重新生成。" );
        feedbackEl.value = "";
        renderState(data);
      } catch (error) {
        renderMessage("assistant", String(error.message || error));
      } finally {
        approveBtn.disabled = false;
        reviseBtn.disabled = false;
      }
    }

    send.addEventListener("click", submit);
    newSessionBtn.addEventListener("click", createNewSession);
    attachFileBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => uploadFiles(fileInput.files));

    function resizePrompt() {
      prompt.style.height = "auto";
      prompt.style.height = `${Math.min(prompt.scrollHeight, 180)}px`;
    }

    prompt.addEventListener("input", resizePrompt);
    prompt.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submit();
      }
    });
    composer.addEventListener("dragenter", event => {
      if (!event.dataTransfer || !Array.from(event.dataTransfer.types).includes("Files")) return;
      event.preventDefault();
      dragDepth += 1;
      composer.classList.add("dragging");
    });
    composer.addEventListener("dragover", event => {
      if (!event.dataTransfer || !Array.from(event.dataTransfer.types).includes("Files")) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    });
    composer.addEventListener("dragleave", event => {
      if (!event.dataTransfer || !Array.from(event.dataTransfer.types).includes("Files")) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) composer.classList.remove("dragging");
    });
    composer.addEventListener("drop", event => {
      if (!event.dataTransfer || !event.dataTransfer.files.length) return;
      event.preventDefault();
      dragDepth = 0;
      composer.classList.remove("dragging");
      uploadFiles(event.dataTransfer.files);
    });
    resizePrompt();
    restoreTaskState();
  </script>
</body>
</html>"""


agent = ResearchAgentService()
background_tasks: dict[str, asyncio.Task[Any]] = {}
Path(config.artifact_root).mkdir(parents=True, exist_ok=True)

FIRST_TURN_INTRO = (
    "我是科研智能体(ResearchAgent)，可以使用“/”完成以下功能:\n"
    "-文献综述:/review\n"
    "-选题与点子发散:/idea\n"
    "-实验方案与计划:/plan\n"
    "-代码与实现:/code\n"
    "-论文写作:/write\n"
    "-rebuttal 回复审稿人:/rebuttal\n"
    "-图表与可视化:/fig\n"
    "-汇报与展示:/present\n"
    "-研究记忆/资料整理:/wiki\n"
    "你的问题已经接收到，请等待回复。"
)
from .workspace_access import touch_workspace_access
BACKGROUND_ACK = "已收到指令，正在执行...（这可能需要几分钟的时间，完成后会直接给你访问工作空间的链接）"
TEXT_FIRST_TURN_INTRO = (
    "你好，我是科研智能体 Research Agent，专注于文献梳理、选题发现、实验规划、"
    "论文写作、审稿回复和科研资料整理。你可以直接用文字描述需求，也可以使用 /review、"
    "/idea、/plan、/code、/write、/rebuttal、/fig、/present 或 /wiki。"
)


def _cloud_workspace_url(session: Any) -> str:
    """Return only a public Seafile URL for text-only API clients."""
    cloud = getattr(session, "cloud_workspace", None)
    if cloud is None:
        return ""
    return str(
        getattr(cloud, "preview_url", "")
        or getattr(cloud, "download_url", "")
        or getattr(cloud, "share_url", "")
        or ""
    )


def _first_turn_text(session: Any) -> str:
    """Build the one-time pure-text welcome/workspace handoff."""
    lines = [TEXT_FIRST_TURN_INTRO]
    cloud_url = _cloud_workspace_url(session)
    if cloud_url:
        lines.extend(["", "工作区已创建", f"清华网盘工作区链接：{cloud_url}"])
    return "\n".join(lines)


def reserve_first_turn_intro(session_id: str | None, user_id: str) -> tuple[str, str]:
    session = agent.store.get_or_create_session(session_id, user_id)
    intro = ""
    if agent.store.consume_first_turn_intro(session.user_id):
        intro = FIRST_TURN_INTRO
    return session.session_id, intro


def task_status_payload(task_id: str, *, text: str = "") -> dict[str, Any]:
    task = agent.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    checkpoint = (
        task.approvals[-1]
        if task.status == "waiting_human"
        and task.approvals
        and task.approvals[-1].status == "pending"
        else None
    )
    payload = task.model_dump()
    session = agent.store.load_session(task.session_id)
    response_text = task.response_text
    if response_text and session:
        response_text = agent._append_cloud_delivery_link(
            response_text,
            session,
            has_artifacts=bool(task.artifacts),
        )
    status_text = text or response_text or task.summary or task.error
    payload.update(
        {
            "text": status_text,
            "response_text": response_text,
            "artifacts": [artifact.model_dump() for artifact in task.artifacts[-6:]],
            "progress": task.progress_log[-10:],
            "progress_events": [event.model_dump() for event in task.progress_events[-50:]],
            "checkpoint": checkpoint.model_dump() if checkpoint else None,
            "cloud_workspace": session.cloud_workspace.model_dump() if session else {},
        }
    )
    return payload


def _sse_event(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def require_public_api_key(authorization: str | None) -> None:
    expected = config.public_api_key.strip()
    if not expected:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or token.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def run_task_in_background(task_id: str, operation: str, feedback: str) -> None:
    try:
        if operation == "start":
            await agent.execute_task(task_id)
        elif operation == "approve":
            await agent.approve_task(task_id, feedback)
        elif operation == "reject":
            await agent.reject_task(task_id, feedback)
        else:
            await agent.continue_task(task_id)
    except BaseException as exc:
        agent.record_task_failure(task_id, exc)
        if isinstance(exc, asyncio.CancelledError):
            raise


def schedule_task(task_id: str, operation: str, feedback: str = "") -> None:
    active = background_tasks.get(task_id)
    if active and not active.done():
        return
    task = asyncio.create_task(run_task_in_background(task_id, operation, feedback))
    background_tasks[task_id] = task
    task.add_done_callback(lambda completed, key=task_id: background_tasks.pop(key, None))


async def resume_interrupted_tasks() -> None:
    for task in agent.list_tasks():
        if task.status == "running":
            operation = (
                "approve"
                if task.approvals and task.approvals[-1].status == "pending"
                else "continue"
            )
            schedule_task(task.task_id, operation)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await resume_interrupted_tasks()
    yield
    pending = [task for task in background_tasks.values() if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


app = FastAPI(title="research-agent-platform", version="0.4.0", lifespan=lifespan)
app.mount("/workspace-files", StaticFiles(directory=config.artifact_root), name="workspace-files")


@app.get("/api/tasks/{task_id}/events")
async def api_task_events(task_id: str) -> StreamingResponse:
    if not agent.get_task(task_id):
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")

    async def event_stream():
        last_sequence = 0
        sent_snapshot = False
        while True:
            task = agent.get_task(task_id)
            if task is None:
                yield _sse_event("error", {"task_id": task_id, "error": "Unknown task"})
                return
            events = [event for event in task.progress_events if event.sequence > last_sequence]
            if not sent_snapshot:
                yield _sse_event("snapshot", {"task": task_status_payload(task_id)})
                sent_snapshot = True
            for event in events:
                last_sequence = max(last_sequence, event.sequence)
                yield _sse_event(
                    "progress",
                    {"task_id": task_id, "event": event.model_dump(), "task": task_status_payload(task_id)},
                )
            if task.status in {"completed", "failed", "waiting_human"}:
                yield _sse_event("done", {"task": task_status_payload(task_id)})
                return
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/chat")


@app.get("/health")
async def health() -> dict[str, str]:
    models = await upstream_list_models()
    first_model = ((models.get("data") or [{}])[0]).get("id", "")
    return {"status": "ok", "model": first_model, "workspace": config.artifact_root}


@app.get("/api/cloud/config")
def api_cloud_config() -> dict[str, Any]:
    return {
        **agent.cloud.configuration_status(),
        "delivery_required": config.cloud_delivery_required,
    }


@app.get("/chat")
def chat_page() -> HTMLResponse:
    return HTMLResponse(CHAT_PAGE)


@app.post("/api/sessions")
async def api_create_session(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    user_id = str((payload or {}).get("user_id") or "local")
    session = agent.store.create_session(user_id)
    workspace_root = agent.artifacts.session_root(
        user_id=session.user_id,
        session_id=session.session_id,
    )
    session.workspace_root = str(workspace_root.resolve())
    touch_workspace_access(workspace_root)
    agent.store.save_session(session)
    cloud_workspace = await agent.sync_session_workspace(session)
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "workspace_root": session.workspace_root,
        "cloud_workspace": cloud_workspace.model_dump(),
    }


@app.post("/api/agent/chat")
async def api_agent_chat(payload: dict[str, Any]) -> dict[str, Any]:
    user_id = str(payload.get("user_id") or "local")
    message = str(payload.get("message", ""))
    if is_informational_question(message) or not has_execution_intent(message):
        result = await agent.chat(payload.get("session_id"), message, user_id, sync_workspace=False)
        session = agent.store.load_session(result["session_id"])
        if session is not None and agent.store.consume_first_turn_intro(session.user_id):
            result["text"] = f"{FIRST_TURN_INTRO}\n\n{result['text']}"
        return result
    session = await agent._prepare_chat_session(
        payload.get("session_id"),
        message,
        user_id,
        sync_workspace=False,
    )
    touch_workspace_access(session.workspace_root)
    intro = ""
    if agent.store.consume_first_turn_intro(session.user_id):
        intro = FIRST_TURN_INTRO
    task = await agent._create_chat_task(session, message, sync_workspace=False)
    schedule_task(task.task_id, "start")
    return {
        "session_id": session.session_id,
        "task_id": task.task_id,
        "status": "running",
        "command": task.command,
        "workflow_title": task.workflow_title,
        "artifact_root": task.artifact_root,
        "artifacts": [],
        "progress": task.progress_log[-10:],
        "checkpoint": None,
        "cloud_workspace": session.cloud_workspace.model_dump(),
        "text": (f"{intro}\n\n" if intro else "") + BACKGROUND_ACK,
    }

@app.post("/api/session/files")
async def api_upload_session_files(
    session_id: str | None = Form(None),
    user_id: str = Form("local"),
    target: str = Form("auto"),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")
    if len(files) > config.upload_max_files:
        raise HTTPException(
            status_code=400,
            detail=f"At most {config.upload_max_files} files can be uploaded at once",
        )
    try:
        normalized_target = normalize_upload_target(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    max_bytes = config.upload_max_file_mb * 1024 * 1024
    for upload in files:
        if upload.size is not None and upload.size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename or 'file'} exceeds the {config.upload_max_file_mb} MB limit",
            )

    session = agent.store.get_or_create_session(session_id, user_id or "local")
    workspace_root = agent.artifacts.session_root(
        user_id=session.user_id,
        session_id=session.session_id,
    )
    session.workspace_root = str(workspace_root.resolve())
    touch_workspace_access(workspace_root)
    agent.store.save_session(session)
    artifacts: list[dict[str, Any]] = []
    for upload in files:
        try:
            filename = sanitize_upload_filename(upload.filename or "")
            directory = classify_upload(filename, normalized_target)
            content = await upload.read(max_bytes + 1)
            if len(content) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"{filename} exceeds the {config.upload_max_file_mb} MB limit",
                )
            relative_path = next_upload_relative_path(workspace_root, directory, filename)
            artifact = agent.artifacts.write_bytes(
                "session-upload",
                relative_path,
                content,
                kind=upload_artifact_kind(filename),
                description="User-uploaded research material.",
                task_root=workspace_root,
            )
            artifacts.append({**artifact.model_dump(), "size": len(content)})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await upload.close()

    upload_batch = UploadBatchRecord(
        relative_paths=[artifact["relative_path"] for artifact in artifacts]
    )
    session.upload_batches.append(upload_batch)
    session.upload_batches = session.upload_batches[-20:]
    agent.store.save_session(session)
    await agent.sync_session_workspace(session)

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "workspace_root": str(workspace_root.resolve()),
        "upload_batch_id": upload_batch.upload_batch_id,
        "files": artifacts,
        "cloud_workspace": session.cloud_workspace.model_dump(),
    }


@app.post("/api/sessions/{session_id}/sync")
async def api_sync_session_workspace(session_id: str) -> dict[str, Any]:
    session = agent.store.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    touch_workspace_access(session.workspace_root)
    cloud_workspace = await agent.sync_session_workspace(session)
    return {
        "session_id": session.session_id,
        "workspace_root": session.workspace_root,
        "cloud_workspace": cloud_workspace.model_dump(),
    }


@app.get("/api/tasks")
def api_tasks(session_id: str | None = None) -> list[dict[str, Any]]:
    return [task.model_dump() for task in agent.list_tasks(session_id)]


@app.get("/api/tasks/{task_id}")
def api_task(task_id: str) -> dict[str, Any]:
    return task_status_payload(task_id)


@app.get("/api/tasks/{task_id}/files")
def api_task_files(task_id: str) -> dict[str, Any]:
    task = agent.get_task(task_id)
    if not task:
        return {"error": f"Unknown task: {task_id}"}
    root = Path(task.artifact_root)
    if not root.exists():
        return {"task_id": task_id, "files": []}
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "absolute_path": str(path.resolve()),
                "url_path": agent.artifacts.url_for(root, path.relative_to(root).as_posix()),
                "size": path.stat().st_size,
            }
        )
    session = agent.store.load_session(task.session_id)
    return {
        "task_id": task_id,
        "artifact_root": task.artifact_root,
        "files": files,
        "cloud_workspace": session.cloud_workspace.model_dump() if session else {},
    }


@app.post("/api/tasks/{task_id}/approve")
async def api_approve(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    task = agent.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    active = background_tasks.get(task_id)
    if active and not active.done():
        return task_status_payload(task_id, text="任务已在后台执行，请查看实时进度。")
    if task.status != "waiting_human":
        raise HTTPException(status_code=409, detail=f"Task is not waiting for approval: {task.status}")
    agent.mark_task_scheduled(task_id, "approve")
    schedule_task(task_id, "approve", str(payload.get("feedback", "")))
    return task_status_payload(task_id, text="已批准，正在后台生成逐页内容、页面图片和 PPT。")


@app.post("/api/tasks/{task_id}/reject")
async def api_reject(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    task = agent.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    active = background_tasks.get(task_id)
    if active and not active.done():
        return task_status_payload(task_id, text="任务已在后台执行，请查看实时进度。")
    if task.status != "waiting_human":
        raise HTTPException(status_code=409, detail=f"Task is not waiting for revision: {task.status}")
    feedback = str(payload.get("feedback", "")).strip()
    if not feedback:
        raise HTTPException(status_code=400, detail="Revision feedback is required.")
    agent.mark_task_scheduled(task_id, "reject")
    schedule_task(task_id, "reject", feedback)
    return task_status_payload(task_id, text="修改意见已发送，正在后台重新生成大纲。")


@app.post("/api/tasks/{task_id}/resume")
async def api_resume(task_id: str) -> dict[str, Any]:
    task = agent.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    active = background_tasks.get(task_id)
    if active and not active.done():
        return task_status_payload(task_id, text="任务已在后台执行，请查看实时进度。")
    if task.status not in {"running", "failed"}:
        raise HTTPException(status_code=409, detail=f"Task cannot be resumed from status: {task.status}")
    task.status = "running"
    task.error = ""
    task.summary = "正在从最近的工作流断点恢复。"
    agent._log_progress(task, task.summary)
    agent.store.save_task(task)
    session = agent.store.load_session(task.session_id)
    if session:
        session.active_task_id = task.task_id
        agent.store.save_session(session)
    operation = (
        "approve"
        if task.approvals and task.approvals[-1].status == "pending"
        else "continue"
    )
    schedule_task(task_id, operation)
    return task_status_payload(task_id, text="任务已从断点恢复，正在后台继续生成。")


def chat_completion_payload(
    payload: dict[str, Any],
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    assistant_text = agent_result["text"]
    response_id = agent_result["task_id"] or agent_result["session_id"]
    return {
        "id": f"chatcmpl-{response_id}",
        "object": "chat.completion",
        "created": 0,
        "model": payload.get("model", "research-agent-platform"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_agent_task": {
            "task_id": agent_result["task_id"],
            "status": agent_result["status"],
            # Do not expose the server filesystem to text-only clients.
            "artifact_root": "",
            "session_id": agent_result["session_id"],
            "progress": agent_result.get("progress", []),
        },
    }


def chat_completion_stream_payload(
    payload: dict[str, Any],
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    completion = chat_completion_payload(payload, agent_result)
    choice = completion["choices"][0]
    return {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": choice["message"]["content"],
                },
                "finish_reason": choice["finish_reason"],
            }
        ],
        "usage": completion["usage"],
        "x_agent_task": completion["x_agent_task"],
    }


def _openai_stream_chunk(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_public_api_key(authorization)
    return await upstream_list_models()


@app.get("/v1")
async def openai_compatible_discovery() -> dict[str, Any]:
    """Describe the public OpenAI-compatible base path for gateway probes."""
    return {
        "object": "api",
        "base_path": "/v1",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/responses"],
        "authentication": "Bearer token in the Authorization header",
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: dict[str, Any],
    authorization: str | None = Header(default=None),
):
    require_public_api_key(authorization)
    messages = payload.get("messages", [])
    session_id = None
    user_id = "local"
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        session_id = metadata.get("session_id")
        user_id = str(metadata.get("user_id") or metadata.get("user") or user_id)
    if not session_id:
        session_id = payload.get("user")
    if user_id == "local" and payload.get("user"):
        user_id = str(payload.get("user"))

    latest_user = ""
    for item in reversed(messages):
        if item.get("role") == "user":
            latest_user = extract_text_from_message(item.get("content", ""))
            break
    if not latest_user:
        latest_user = "Please summarize the current task state."

    if payload.get("stream"):
        if is_informational_question(latest_user) or not has_execution_intent(latest_user):
            result = await agent.chat(session_id, latest_user, user_id, sync_workspace=False)
            session = agent.store.load_session(result["session_id"])
            if session is not None and agent.store.consume_first_turn_intro(session.user_id):
                result["text"] = f"{_first_turn_text(session)}\n\n{result['text']}"

            async def direct_event_stream():
                yield _openai_stream_chunk(chat_completion_stream_payload(payload, result))
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                direct_event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        session = await agent._prepare_chat_session(session_id, latest_user, user_id, sync_workspace=False)
        intro = ""
        if agent.store.consume_first_turn_intro(session.user_id):
            intro = _first_turn_text(session)
        task = await agent._create_chat_task(session, latest_user, sync_workspace=False)
        schedule_task(task.task_id, "start")

        async def event_stream():
            if intro:
                yield _openai_stream_chunk(
                    chat_completion_stream_payload(
                        payload,
                        {
                            "text": intro,
                            "task_id": "",
                            "status": "running",
                            "artifact_root": "",
                            "session_id": session.session_id,
                            "progress": [],
                        },
                    )
                )
            yield _openai_stream_chunk(
                chat_completion_stream_payload(
                    payload,
                    {
                        "text": BACKGROUND_ACK,
                        "task_id": task.task_id,
                        "status": "running",
                        "artifact_root": task.artifact_root,
                        "session_id": session.session_id,
                        "progress": task.progress_log[-10:],
                    },
                )
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    # OpenAI-compatible chat must not wait on best-effort Seafile delivery.
    # Workspace access is touched synchronously and the existing background
    # scheduler can finish delivery; keeping this request local prevents a
    # congested cloud endpoint from turning ordinary multi-turn chat into
    # 90-second timeouts/429s.
    agent_result = await agent.chat(session_id, latest_user, user_id, sync_workspace=False)
    session = agent.store.load_session(agent_result["session_id"])
    if session is not None and agent.store.consume_first_turn_intro(session.user_id):
        intro = _first_turn_text(session)
        if intro:
            if agent_result.get("status") == "idle" and not agent_result.get("task_id"):
                agent_result["text"] = intro
            else:
                agent_result["text"] = f"{intro}\n\n{agent_result['text']}"
    return chat_completion_payload(payload, agent_result)


@app.post("/v1/responses")
async def responses(
    payload: dict[str, Any],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require_public_api_key(authorization)
    session_id = None
    user_id = "local"
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        session_id = metadata.get("session_id")
        user_id = str(metadata.get("user_id") or metadata.get("user") or user_id)
    if user_id == "local" and payload.get("user"):
        user_id = str(payload.get("user"))
    result = await agent.chat(session_id, str(payload.get("input", "")), user_id, sync_workspace=False)
    session = agent.store.load_session(result["session_id"])
    if session is not None and agent.store.consume_first_turn_intro(session.user_id):
        intro = _first_turn_text(session)
        if intro:
            result["text"] = f"{intro}\n\n{result['text']}"
    text = result["text"]
    response_id = result["task_id"] or result["session_id"]
    return {
        "id": f"resp-{response_id}",
        "object": "response",
        "created_at": 0,
        "model": payload.get("model", "research-agent-platform"),
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "metadata": {
            "task_id": result["task_id"],
            "status": result["status"],
            "artifact_root": "",
            "session_id": result["session_id"],
            "progress": result.get("progress", []),
        },
    }



