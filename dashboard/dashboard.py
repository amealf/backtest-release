#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local HTML dashboard for backtest outputs.

Run:
    python dashboard/dashboard.py
Open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8765
AUTO_OPEN_BROWSER = True

# Set this to your result folder. Relative path is based on project root.
# Example: "DAT_ASCII_XAGUSD_T_202501_15s long outcome"
HTML_SOURCE_FOLDER = "DAT_ASCII_XAGUSD_T_202501_15s long outcome"

IGNORE_DIR_NAMES = {".git", "__pycache__"}
MIN_SIDEBAR_WIDTH = 240
MAX_SIDEBAR_WIDTH = 760
DEFAULT_SIDEBAR_WIDTH = 320

# Project root = parent of dashboard/ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_scan_root() -> Path:
    folder = (HTML_SOURCE_FOLDER or "").strip()
    if not folder:
        return PROJECT_ROOT.resolve()
    path = Path(folder)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


SCAN_ROOT = _resolve_scan_root()


def _scan_html_files() -> list[dict]:
    if not SCAN_ROOT.exists() or not SCAN_ROOT.is_dir():
        return []

    files = []
    for html_file in SCAN_ROOT.rglob("*.html"):
        if any(part in IGNORE_DIR_NAMES for part in html_file.parts):
            continue
        try:
            rel_to_scan = html_file.relative_to(SCAN_ROOT).as_posix()
            folder_rel = html_file.parent.relative_to(SCAN_ROOT).as_posix()
            stat = html_file.stat()
        except Exception:
            continue

        files.append(
            {
                "name": html_file.name,
                "token": rel_to_scan,
                "folder": folder_rel if folder_rel != "." else "",
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "mtime_ts": stat.st_mtime,
                "size_kb": round(stat.st_size / 1024.0, 1),
            }
        )

    files.sort(key=lambda x: x["mtime_ts"], reverse=True)
    return files


def _resolve_safe_html_path(token: str) -> Path | None:
    if not token:
        return None

    rel_clean = unquote(token).replace("\\", "/")
    candidate = (SCAN_ROOT / rel_clean).resolve()

    try:
        candidate.relative_to(SCAN_ROOT)
    except ValueError:
        return None

    if not candidate.is_file() or candidate.suffix.lower() != ".html":
        return None

    return candidate


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def _send_bytes(
        self,
        status: int,
        payload: bytes,
        content_type: str,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(
        self,
        status: int,
        text: str,
        content_type: str,
        cache_control: str = "no-store",
    ) -> None:
        self._send_bytes(
            status, text.encode("utf-8"), content_type, cache_control=cache_control
        )

    def _send_json(self, status: int, obj) -> None:
        self._send_text(
            status,
            json.dumps(obj, ensure_ascii=False),
            "application/json; charset=utf-8",
        )

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send_text(200, DASHBOARD_HTML, "text/html; charset=utf-8")
            return

        if path == "/api/files":
            self._send_json(
                200,
                {
                    "files": _scan_html_files(),
                    "scan_root": str(SCAN_ROOT),
                    "scan_exists": SCAN_ROOT.exists() and SCAN_ROOT.is_dir(),
                },
            )
            return

        if path == "/view":
            query = parse_qs(parsed.query)
            token = query.get("file", [""])[0]
            safe_path = _resolve_safe_html_path(token)
            if safe_path is None:
                self._send_text(404, "HTML file not found.", "text/plain; charset=utf-8")
                return
            try:
                content = safe_path.read_bytes()
            except Exception as exc:
                self._send_text(500, f"Failed to read file: {exc}", "text/plain; charset=utf-8")
                return
            self._send_bytes(
                200,
                content,
                "text/html; charset=utf-8",
                cache_control="public, max-age=3600",
            )
            return

        super().do_GET()


DASHBOARD_HTML = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Backtest HTML Dashboard</title>
  <style>
    :root {{
      --bg: #f3f4f6;
      --sidebar-bg: #ffffff;
      --sidebar-border: #d1d5db;
      --text: #111827;
      --sub: #6b7280;
      --hover: #f9fafb;
      --active: #eef6ff;
      --active-border: #3b82f6;
      --accent: #2563eb;
      --main-bg: #ffffff;
      --splitter: #e5e7eb;
      --sidebar-width: {DEFAULT_SIDEBAR_WIDTH}px;
    }}

    * {{ box-sizing: border-box; }}

    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}

    .app {{
      width: 100%;
      height: 100%;
      display: grid;
      grid-template-columns: var(--sidebar-width) 8px minmax(0, 1fr);
      grid-template-rows: 100%;
    }}

    .app.sidebar-hidden {{
      grid-template-columns: 0 0 minmax(0, 1fr);
    }}

    .sidebar {{
      min-width: {MIN_SIDEBAR_WIDTH}px;
      max-width: {MAX_SIDEBAR_WIDTH}px;
      border-right: 1px solid var(--sidebar-border);
      background: var(--sidebar-bg);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    .app.sidebar-hidden .sidebar {{
      display: none;
    }}

    .splitter {{
      background: var(--splitter);
      cursor: col-resize;
      user-select: none;
    }}

    .app.sidebar-hidden .splitter {{
      display: none;
    }}

    .sidebar-header {{
      padding: 10px;
      border-bottom: 1px solid var(--sidebar-border);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
    }}

    .search {{
      width: 100%;
      height: 34px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 0 10px;
      font-size: 13px;
      outline: none;
      color: var(--text);
      background: #fff;
    }}

    .search:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }}

    .btn {{
      height: 34px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #fff;
      color: #111827;
      padding: 0 10px;
      font-size: 12px;
      cursor: pointer;
    }}

    .btn:hover {{
      background: #f9fafb;
    }}

    .file-list {{
      margin: 0;
      padding: 8px;
      list-style: none;
      overflow: auto;
      flex: 1;
    }}

    .file-item {{
      border: 1px solid transparent;
      border-radius: 10px;
      padding: 9px 10px;
      margin-bottom: 8px;
      background: #fff;
      cursor: default;
      transition: border-color 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
    }}

    .file-item:hover {{
      background: var(--hover);
      border-color: #93c5fd;
      box-shadow: inset 0 0 0 1px #93c5fd;
    }}

    .file-item.active {{
      background: var(--active);
      border-color: var(--active-border);
      box-shadow: inset 0 0 0 1px var(--active-border);
    }}

    .name {{
      color: #111827;
      font-size: 12px;
      line-height: 1.35;
      word-break: break-all;
      font-weight: 600;
    }}

    .meta {{
      margin-top: 4px;
      color: var(--sub);
      font-size: 11px;
      line-height: 1.35;
      word-break: break-all;
    }}

    .main {{
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: 40px minmax(0, 1fr);
      background: var(--main-bg);
    }}

    .main-header {{
      border-bottom: 1px solid #e5e7eb;
      display: grid;
      grid-template-columns: auto 1fr;
      align-items: center;
      gap: 8px;
      padding: 4px 8px;
      font-size: 12px;
      color: #374151;
      min-width: 0;
    }}

    .path-text {{
      min-width: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .viewer-wrap {{
      position: relative;
      width: 100%;
      height: 100%;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      background: #fff;
    }}

    iframe {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      border: 0;
      display: block;
      background: #fff;
    }}

    .empty {{
      width: 100%;
      height: 100%;
      display: grid;
      place-items: center;
      color: #6b7280;
      font-size: 14px;
      text-align: center;
      padding: 16px;
    }}

    @media (max-width: 980px) {{
      :root {{ --sidebar-width: 300px; }}
    }}
  </style>
</head>
<body>
  <div id=\"app\" class=\"app\">
    <aside id=\"sidebar\" class=\"sidebar\">
      <div class=\"sidebar-header\">
        <input id=\"search\" class=\"search\" placeholder=\"Filter html file name...\" />
        <button id=\"refreshBtn\" class=\"btn\">Refresh</button>
      </div>
      <ul id=\"fileList\" class=\"file-list\"></ul>
    </aside>

    <div id=\"splitter\" class=\"splitter\" title=\"Drag to resize sidebar\"></div>

    <main class=\"main\">
      <div class=\"main-header\">
        <button id=\"toggleSidebarBtn\" class=\"btn\">Hide list</button>
        <div id=\"mainHeaderPath\" class=\"path-text\">No file selected</div>
      </div>
      <div id=\"viewerWrap\" class=\"viewer-wrap\">
        <div class=\"empty\">Hover a file on the left list to preview.</div>
      </div>
    </main>
  </div>

  <script>
    const MIN_SIDEBAR_WIDTH = {MIN_SIDEBAR_WIDTH};
    const MAX_SIDEBAR_WIDTH = {MAX_SIDEBAR_WIDTH};
    const LIST_PAGE_SIZE = 10;
    const IFRAME_CACHE_LIMIT = 20;
    const PRELOAD_CONCURRENCY = 2;

    const appEl = document.getElementById("app");
    const splitterEl = document.getElementById("splitter");
    const searchEl = document.getElementById("search");
    const refreshBtn = document.getElementById("refreshBtn");
    const toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
    const fileListEl = document.getElementById("fileList");
    const mainHeaderPathEl = document.getElementById("mainHeaderPath");
    const viewerWrapEl = document.getElementById("viewerWrap");

    let files = [];
    let filtered = [];
    let selectedToken = "";
    let hoverTimer = null;
    let sidebarHidden = false;
    let isDragging = false;
    let visibleCount = 0;
    let renderedCount = 0;

    const iframeCache = new Map();
    const preloadQueue = [];
    const preloadSet = new Set();
    let preloadInFlightCount = 0;

    const hoverDelayMs = 35;

    function setSidebarWidth(px) {{
      const v = Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, px));
      document.documentElement.style.setProperty("--sidebar-width", `${{v}}px`);
    }}

    function setSidebarHidden(hidden) {{
      sidebarHidden = hidden;
      appEl.classList.toggle("sidebar-hidden", hidden);
      toggleSidebarBtn.textContent = hidden ? "Show list" : "Hide list";
    }}

    function escapeHtml(s) {{
      return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function displayName(name) {{
      return String(name || "")
        .replace(/\s+(Long|Short)\s+interactive\.html?$/i, "")
        .trim();
    }}

    function updateActiveState() {{
      const nodes = fileListEl.querySelectorAll(".file-item[data-token]");
      nodes.forEach((li) => {{
        li.classList.toggle("active", li.dataset.token === selectedToken);
      }});
    }}

    function createListItem(f) {{
      const li = document.createElement("li");
      li.className = "file-item" + (f.token === selectedToken ? " active" : "");
      li.dataset.token = f.token;

      const folderText = f.folder ? `${{escapeHtml(f.folder)}}<br>` : "";
      li.innerHTML = `
        <div class=\"name\">${{escapeHtml(displayName(f.name))}}</div>
        <div class=\"meta\">${{folderText}}${{f.mtime}} | ${{f.size_kb}} KB</div>
      `;

      li.addEventListener("mouseenter", () => {{
        if (hoverTimer) clearTimeout(hoverTimer);
        hoverTimer = setTimeout(() => selectFile(f.token), hoverDelayMs);
      }});
      return li;
    }}

    function renderList(reset = false) {{
      if (reset) {{
        fileListEl.innerHTML = "";
        renderedCount = 0;
        fileListEl.scrollTop = 0;
      }}

      const target = Math.min(visibleCount, filtered.length);
      for (let i = renderedCount; i < target; i++) {{
        fileListEl.appendChild(createListItem(filtered[i]));
      }}
      renderedCount = target;

      if (filtered.length === 0) {{
        fileListEl.innerHTML = "";
        const li = document.createElement("li");
        li.className = "file-item";
        li.innerHTML = `<div class=\"name\" style=\"color:#6b7280;\">No matched html files</div>`;
        fileListEl.appendChild(li);
      }}
    }}

    function loadMoreIfNeeded() {{
      const nearBottom = fileListEl.scrollTop + fileListEl.clientHeight >= fileListEl.scrollHeight - 40;
      if (!nearBottom) return;
      if (visibleCount >= filtered.length) return;
      visibleCount = Math.min(filtered.length, visibleCount + LIST_PAGE_SIZE);
      renderList(false);
      queuePreloadVisible();
      updateActiveState();
    }}

    function touchCache(token) {{
      if (!iframeCache.has(token)) return;
      const frame = iframeCache.get(token);
      iframeCache.delete(token);
      iframeCache.set(token, frame);
    }}

    function enforceCacheLimit() {{
      while (iframeCache.size > IFRAME_CACHE_LIMIT) {{
        const oldest = iframeCache.keys().next().value;
        if (oldest === selectedToken && iframeCache.size > 1) {{
          touchCache(oldest);
          continue;
        }}
        const frame = iframeCache.get(oldest);
        iframeCache.delete(oldest);
        if (frame && frame.isConnected) frame.remove();
      }}
    }}

    function createIframe(token) {{
      const frame = document.createElement("iframe");
      frame.src = `/view?file=${{encodeURIComponent(token)}}`;
      frame.loading = "eager";
      frame.dataset.token = token;
      frame.style.display = "block";
      frame.style.visibility = "hidden";
      frame.style.pointerEvents = "none";
      frame.style.zIndex = "0";
      return frame;
    }}

    function getOrCreateFrame(token) {{
      if (iframeCache.has(token)) {{
        return iframeCache.get(token);
      }}
      const frame = createIframe(token);
      if (viewerWrapEl.querySelector(".empty")) {{
        viewerWrapEl.innerHTML = "";
      }}
      viewerWrapEl.appendChild(frame);
      iframeCache.set(token, frame);
      enforceCacheLimit();
      return frame;
    }}

    function showFrame(frame) {{
      if (!frame) return;
      if (!frame.isConnected) {{
        if (viewerWrapEl.querySelector(".empty")) {{
          viewerWrapEl.innerHTML = "";
        }}
        viewerWrapEl.appendChild(frame);
      }}
      const nodes = viewerWrapEl.querySelectorAll("iframe");
      nodes.forEach((node) => {{
        const active = node === frame;
        node.style.display = "block";
        node.style.visibility = active ? "visible" : "hidden";
        node.style.pointerEvents = active ? "auto" : "none";
        node.style.zIndex = active ? "1" : "0";
      }});
    }}

    function queuePreloadVisible() {{
      const tokens = filtered.slice(0, visibleCount).map((x) => x.token);
      for (const token of tokens) {{
        if (iframeCache.has(token) || preloadSet.has(token)) continue;
        preloadSet.add(token);
        preloadQueue.push(token);
      }}
      processPreloadQueue();
    }}

    function processPreloadQueue() {{
      while (preloadInFlightCount < PRELOAD_CONCURRENCY && preloadQueue.length > 0) {{
        const token = preloadQueue.shift();
        preloadSet.delete(token);
        if (!token || iframeCache.has(token)) {{
          continue;
        }}

        preloadInFlightCount += 1;
        const frame = createIframe(token);
        if (viewerWrapEl.querySelector(".empty")) {{
          viewerWrapEl.innerHTML = "";
        }}
        viewerWrapEl.appendChild(frame);
        iframeCache.set(token, frame);
        enforceCacheLimit();

        let done = false;
        const finish = () => {{
          if (done) return;
          done = true;
          preloadInFlightCount = Math.max(0, preloadInFlightCount - 1);
          processPreloadQueue();
        }};

        frame.addEventListener("load", finish, {{ once: true }});
        frame.addEventListener("error", finish, {{ once: true }});
        setTimeout(finish, 8000);
      }}
    }}

    function selectFile(token) {{
      if (!token) return;
      if (token === selectedToken) return;

      selectedToken = token;
      updateActiveState();

      const f = files.find((x) => x.token === token);
      mainHeaderPathEl.textContent = f ? `${{f.folder ? f.folder + " / " : ""}}${{displayName(f.name)}}` : token;

      const frame = getOrCreateFrame(token);
      showFrame(frame);
      touchCache(token);
      enforceCacheLimit();
    }}

    async function fetchFiles() {{
      const res = await fetch("/api/files");
      if (!res.ok) throw new Error("Failed to load file list");

      const data = await res.json();
      files = data.files || [];

      if (!data.scan_exists) {{
        viewerWrapEl.innerHTML = `<div class=\"empty\">Scan folder does not exist:<br>${{escapeHtml(data.scan_root || "")}}</div>`;
      }}

      applyFilter();
    }}

    function applyFilter() {{
      const kw = (searchEl.value || "").trim().toLowerCase();
      filtered = !kw
        ? files
        : files.filter((f) => f.name.toLowerCase().includes(kw)
          || displayName(f.name).toLowerCase().includes(kw)
          || (f.folder || "").toLowerCase().includes(kw));

      visibleCount = Math.min(filtered.length, LIST_PAGE_SIZE);
      renderList(true);

      if (filtered.length === 0) {{
        selectedToken = "";
        mainHeaderPathEl.textContent = "No file selected";
        return;
      }}

      queuePreloadVisible();

      const stillExists = filtered.some((f) => f.token === selectedToken);
      if (!stillExists) {{
        selectFile(filtered[0].token);
      }} else {{
        updateActiveState();
        const f = files.find((x) => x.token === selectedToken);
        mainHeaderPathEl.textContent = f ? `${{f.folder ? f.folder + " / " : ""}}${{displayName(f.name)}}` : selectedToken;
        const frame = iframeCache.get(selectedToken) || getOrCreateFrame(selectedToken);
        showFrame(frame);
      }}
    }}

    splitterEl.addEventListener("mousedown", (e) => {{
      if (sidebarHidden) return;
      isDragging = true;
      document.body.style.cursor = "col-resize";
      e.preventDefault();
    }});

    window.addEventListener("mousemove", (e) => {{
      if (!isDragging) return;
      setSidebarWidth(e.clientX);
    }});

    window.addEventListener("mouseup", () => {{
      if (!isDragging) return;
      isDragging = false;
      document.body.style.cursor = "";
    }});

    toggleSidebarBtn.addEventListener("click", () => {{
      setSidebarHidden(!sidebarHidden);
    }});

    fileListEl.addEventListener("scroll", loadMoreIfNeeded);
    searchEl.addEventListener("input", applyFilter);

    refreshBtn.addEventListener("click", async () => {{
      await fetchFiles();
      queuePreloadVisible();
    }});

    fetchFiles().catch((err) => {{
      viewerWrapEl.innerHTML = `<div class=\"empty\">Load failed: ${{escapeHtml(err.message)}}</div>`;
    }});
  </script>
</body>
</html>
"""


def run_dashboard_server() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"[Dashboard] project root: {PROJECT_ROOT}")
    print(f"[Dashboard] scan root: {SCAN_ROOT}")
    if not SCAN_ROOT.exists() or not SCAN_ROOT.is_dir():
        print("[Dashboard] warning: scan folder does not exist.")
    print(f"[Dashboard] open: {url}")
    print("[Dashboard] stop: Ctrl+C")

    if AUTO_OPEN_BROWSER:
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"[Dashboard] browser open failed: {exc}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_dashboard_server()
