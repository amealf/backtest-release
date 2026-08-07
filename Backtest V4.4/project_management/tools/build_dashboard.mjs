import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const defaultManagementRoot = path.dirname(scriptDirectory);
const generatedMarker = "<!-- generated-by: manage-project-context -->";

function parseArguments(values) {
  const options = { managementRoot: defaultManagementRoot, output: "", force: false, help: false };
  for (let index = 0; index < values.length; index += 1) {
    const token = values[index];
    if (token === "--help" || token === "-h") options.help = true;
    else if (token === "--force") options.force = true;
    else if (token === "--management-root") options.managementRoot = values[++index] || "";
    else if (token === "--output") options.output = values[++index] || "";
    else throw new Error(`Unknown argument: ${token}`);
  }
  return options;
}

function safeJson(value) {
  return JSON.stringify(value)
    .replaceAll("<", "\\u003c")
    .replaceAll("\u2028", "\\u2028")
    .replaceAll("\u2029", "\\u2029");
}

async function pathType(target) {
  try {
    const stat = await fs.stat(target);
    if (stat.isFile()) return "file";
    if (stat.isDirectory()) return "directory";
    return "other";
  } catch (error) {
    if (error.code === "ENOENT") return "missing";
    throw error;
  }
}

function proposalName(target, sequence) {
  const extension = path.extname(target);
  const stem = extension ? target.slice(0, -extension.length) : target;
  const suffix = sequence === 1 ? "" : `.${sequence}`;
  return `${stem}.project-management.proposed${suffix}${extension}`;
}

async function nextProposal(target) {
  for (let sequence = 1; ; sequence += 1) {
    const candidate = proposalName(target, sequence);
    if ((await pathType(candidate)) === "missing") return candidate;
  }
}

function assertDocumentPath(document, managementRoot, projectRoot) {
  if (!document.path || !document.id || !document.pair || !document.lang || !document.group) {
    throw new Error("Every manifest document requires id, pair, group, path, and lang.");
  }
  const absolute = path.resolve(managementRoot, ...document.path.split("/"));
  if (absolute !== projectRoot && !absolute.startsWith(`${projectRoot}${path.sep}`)) {
    throw new Error(`Document path escapes the project root: ${document.path}`);
  }
  return absolute;
}

async function loadDocuments(manifest, managementRoot, projectRoot) {
  const documents = [];
  for (const spec of manifest.documents) {
    const absolutePath = assertDocumentPath(spec, managementRoot, projectRoot);
    try {
      const [markdown, stat] = await Promise.all([
        fs.readFile(absolutePath, "utf8"),
        fs.stat(absolutePath),
      ]);
      documents.push({ ...spec, markdown, modified: stat.mtime.toISOString(), missing: false });
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      const missingLabel = spec.lang === "zh" ? "文档缺失" : "Missing document";
      const guidance = spec.lang === "zh"
        ? "该路径列在 manifest 中，但文件不存在。请检查初始化报告或冲突提案。"
        : "This path is listed in the manifest, but the file does not exist. Review the initialization report or conflict proposals.";
      documents.push({
        ...spec,
        markdown: `# ${missingLabel}\n\n${guidance}\n\nPath: ${spec.path}\n`,
        modified: null,
        missing: true,
      });
    }
  }
  return documents;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    console.log("node build_dashboard.mjs [--management-root <path>] [--output <path>] [--force]");
    return;
  }
  if (!options.managementRoot || !path.isAbsolute(options.managementRoot)) {
    throw new Error("--management-root must be an absolute path.");
  }

  const managementRoot = path.resolve(options.managementRoot);
  const manifestPath = path.join(managementRoot, "project-management.json");
  const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
  if (manifest.generatedBy !== "manage-project-context") throw new Error("Manifest ownership marker is missing.");
  if (!Array.isArray(manifest.documents)) throw new Error("Manifest documents must be an array.");

  const projectRoot = path.resolve(managementRoot, ...(manifest.projectRootRelative || "..").split("/"));
  const documents = await loadDocuments(manifest, managementRoot, projectRoot);
  const projectName = manifest.projectName || path.basename(projectRoot);

const html = String.raw`<!doctype html>
<!-- generated-by: manage-project-context -->
<html lang="zh-CN" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project Context</title>
  <script>
    try {
      document.documentElement.dataset.theme = localStorage.getItem("project-management-theme") || "dark";
    } catch (error) {
      document.documentElement.dataset.theme = "dark";
    }
  </script>
  <style>
    :root {
      color-scheme: dark;
      --bg: #080808;
      --surface: #111111;
      --surface-strong: #1a1a1a;
      --toolbar: rgb(8 8 8 / 0.97);
      --outline-panel: #111111;
      --outline-panel-hover: #202020;
      --outline-line: #6f6f6b;
      --outline-active: #f1f1ef;
      --outline-link-active: #387dc9;
      --ink: #efefed;
      --body: #c7c7c5;
      --sidebar-ink: #b2b2b0;
      --muted: #969694;
      --line: #252525;
      --line-strong: #393939;
      --primary: #387dc9;
      --primary-action: #2d67a7;
      --primary-dark: #387dc9;
      --primary-soft: #162b42;
      --bold: #387dc9;
      --accent: #387dc9;
      --code-surface: #131313;
      --inline-code-surface: #2b211d;
      --code-ink: #f1a06b;
      --selection: #2b5f8d;
      --scroll-track: #080808;
      --scroll-thumb: #343434;
      --scroll-thumb-hover: #4a4a4a;
      --shadow: none;
      --focus-ring: rgb(63 143 216 / 0.52);
      --sidebar-width: 270px;
      --toolbar-height: 45px;
      --content-width: 780px;
      --header-visual-offset: -18px;
      --body-visual-offset: 22px;
      --reference-header-inset: 144px;
      --reference-body-inset: 166px;
      --reference-header-width: 780px;
      --page-title-size: 30px;
      --page-title-line: 38px;
      --heading-1-size: 24px;
      --heading-1-line: 31px;
      --heading-2-size: 19px;
      --heading-2-line: 25px;
      --heading-3-size: 16px;
      --heading-3-line: 22px;
      --body-size: 16px;
      --body-line: 24px;
      --sidebar-item-size: 14px;
      --sidebar-item-line: 21px;
      --outline-item-size: 13px;
      --outline-item-line: 16.9px;
    }

    :root[data-theme="light"] {
      color-scheme: light;
      --bg: #ffffff;
      --surface: #f7f8fa;
      --surface-strong: #eaf0f6;
      --toolbar: rgb(255 255 255 / 0.95);
      --outline-panel: #ffffff;
      --outline-panel-hover: #edf2f7;
      --outline-line: #66758b;
      --outline-active: #1f2937;
      --outline-link-active: #387dc9;
      --ink: #1c2637;
      --body: #42516a;
      --sidebar-ink: #606f84;
      --muted: #71809a;
      --line: #dbe3ec;
      --line-strong: #cbd6e2;
      --primary: #387dc9;
      --primary-action: #2d67a7;
      --primary-dark: #387dc9;
      --primary-soft: #e8f1fb;
      --bold: #387dc9;
      --accent: #387dc9;
      --code-surface: #f3f6f9;
      --inline-code-surface: #fff0e8;
      --code-ink: #a84d11;
      --selection: #cfe3f8;
      --scroll-track: #eef2f6;
      --scroll-thumb: #b8c4cf;
      --scroll-thumb-hover: #92a4b4;
      --shadow: 0 12px 30px rgb(31 48 68 / 0.09);
      --focus-ring: rgb(46 112 178 / 0.45);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; background: var(--bg); }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--body);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, "Microsoft YaHei", sans-serif;
      font-size: var(--body-size);
      line-height: var(--body-line);
      letter-spacing: 0;
    }
    ::selection { background: var(--selection); }
    html, body, .sidebar, .outline {
      scrollbar-color: var(--scroll-thumb) var(--scroll-track);
      scrollbar-width: thin;
    }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--scroll-track); }
    ::-webkit-scrollbar-thumb {
      border: 2px solid var(--scroll-track);
      border-radius: 6px;
      background: var(--scroll-thumb);
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--scroll-thumb-hover); }
    ::-webkit-scrollbar-corner { background: var(--scroll-track); }
    button, input, select { font: inherit; letter-spacing: 0; }
    button, a { -webkit-tap-highlight-color: transparent; }
    a { color: var(--accent); text-underline-offset: 3px; }
    a:hover { color: var(--primary-dark); }
    :focus-visible { outline: 3px solid var(--focus-ring); outline-offset: 2px; }

    .app-shell {
      display: grid;
      grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
      min-height: 100vh;
    }
    .brand {
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      border: 1px solid var(--line-strong);
      padding: 0;
      border-radius: 6px;
      background: var(--bg);
      color: var(--ink);
      cursor: pointer;
    }
    .brand:hover { background: var(--surface-strong); }
    .brand-icon {
      width: 16px;
      height: 14px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2px;
      color: currentColor;
    }
    .brand-icon span {
      display: block;
      border: 1.5px solid currentColor;
      border-radius: 1px;
    }
    .brand-icon span:first-child { border-right-width: 1px; }
    .brand-icon span:last-child { border-left-width: 1px; }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    .brand-copy { min-width: 0; flex: 1 1 auto; }
    .brand-title { margin: 0; color: var(--ink); font-size: 17px; font-weight: 600; line-height: 23px; white-space: normal; overflow-wrap: anywhere; }
    .brand-subtitle { margin: 2px 0 0; color: var(--muted); font-size: 12px; line-height: 18px; }
    .sidebar-header { display: flex; align-items: center; gap: 10px; padding: 14px 16px; border-bottom: 1px solid var(--line); }
    .search-region { padding: 10px 12px; border-bottom: 1px solid var(--line); }
    .search-wrap { position: relative; width: 100%; }
    .search-wrap::before {
      content: "";
      position: absolute;
      left: 13px;
      top: 50%;
      width: 10px;
      height: 10px;
      border: 1.5px solid var(--muted);
      border-radius: 50%;
      transform: translateY(-62%);
      pointer-events: none;
    }
    .search-wrap::after {
      content: "";
      position: absolute;
      left: 23px;
      top: 55%;
      width: 6px;
      height: 1.5px;
      background: var(--muted);
      transform: rotate(45deg);
      pointer-events: none;
    }
    #search {
      width: 100%;
      height: 34px;
      padding: 0 14px 0 36px;
      border: 1px solid var(--line-strong);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
    }
    .theme-icon-button {
      width: 36px;
      height: 36px;
      display: grid;
      place-items: center;
      border: 1px solid var(--line-strong);
      border-radius: 6px;
      background: var(--bg);
      color: var(--ink);
      cursor: pointer;
    }
    .theme-icon-button:hover { background: var(--surface-strong); }
    .theme-icon { font-size: 19px; line-height: 1; }
    .language-icon-button {
      min-width: 36px;
      height: 36px;
      display: grid;
      place-items: center;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      color: var(--ink);
      font-size: 12px;
      font-weight: 650;
      line-height: 1;
      cursor: pointer;
    }
    .language-icon-button:hover { background: var(--surface-strong); }

    .mobile-document-select { display: none; padding: 14px 20px 0; }
    #document-select {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line-strong);
      border-radius: 6px;
      background: var(--bg);
      color: var(--ink);
      padding: 0 10px;
    }

    .sidebar {
      position: sticky;
      top: 0;
      align-self: start;
      height: 100vh;
      overflow-y: auto;
      border-right: 1px solid var(--line);
      background: var(--surface);
    }
    .sidebar-navigation { padding: 10px 8px 24px; }
    .nav-group + .nav-group { margin-top: 18px; }
    .nav-group-title {
      margin: 0 10px 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
      line-height: 12px;
    }
    .nav-list { display: grid; gap: 0; }
    .nav-item {
      width: 100%;
      min-height: 30px;
      display: grid;
      grid-template-columns: 3px 1fr;
      align-items: center;
      gap: 8px;
      border: 0;
      border-radius: 5px;
      padding: 3px 8px 3px 6px;
      background: transparent;
      color: var(--sidebar-ink);
      font-size: var(--sidebar-item-size);
      font-weight: 500;
      line-height: var(--sidebar-item-line);
      text-align: left;
      cursor: pointer;
    }
    .nav-item:hover { background: var(--surface-strong); }
    .nav-item.active { background: var(--primary-soft); color: var(--primary-dark); font-weight: 600; }
    .nav-marker { width: 3px; height: 18px; border-radius: 2px; background: transparent; }
    .nav-item.active .nav-marker { background: var(--primary); }
    .nav-item[hidden] { display: none; }
    .nav-empty { display: none; margin: 14px 8px; color: var(--muted); font-size: 13px; }
    .home-nav-group { margin-bottom: 18px; }

    .content-column { min-width: 0; }
    .content-toolbar {
      position: sticky;
      top: 0;
      z-index: 10;
      min-height: var(--toolbar-height);
      display: flex;
      align-items: center;
      gap: 20px;
      justify-content: flex-end;
      padding: 0 20px;
      border-bottom: 0;
      background: var(--bg);
      backdrop-filter: none;
    }
    .toolbar-location { display: none; }
    .toolbar-location strong { color: var(--ink); font-size: 14px; font-weight: 600; }
    .toolbar-actions { display: flex; align-items: center; gap: 2px; flex: 0 0 auto; }
    .toolbar-theme-toggle { border-color: transparent; background: transparent; box-shadow: none; }
    .main {
      min-width: 0;
      padding: 30px 80px 80px;
      background: var(--bg);
    }
    .overview-home { display: none; }
    .overview-note { margin: 0 0 1.35em; }
    .overview-section + .overview-section { margin-top: 2.5em; }
    .overview-section-header { margin-bottom: 1.1em; }
    .overview-section-header h2 { margin-bottom: 0.35em; }
    .overview-section-header p { color: var(--muted); }
    .flow-track {
      display: grid;
      gap: 28px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .flow-step {
      position: relative;
      min-width: 0;
    }
    .flow-step:not(:last-child)::after {
      content: "";
      position: absolute;
      left: 50%;
      bottom: -21px;
      width: 8px;
      height: 8px;
      border-right: 2px solid var(--outline-line);
      border-bottom: 2px solid var(--outline-line);
      transform: translateX(-50%) rotate(45deg);
    }
    .flow-node {
      max-width: 760px;
      margin: 0 auto;
      padding: 14px 16px;
      border: 1px solid var(--line-strong);
      border-radius: 10px;
      background: var(--surface);
    }
    .flow-node.flow-primary {
      border-color: var(--primary);
      background: var(--primary-soft);
    }
    .flow-node-title {
      margin: 0;
      color: var(--ink);
      font-size: 15px;
      font-weight: 700;
      line-height: 21px;
    }
    .flow-node-copy {
      margin: 5px 0 0;
      color: var(--body);
      font-size: 13px;
      line-height: 20px;
    }
    .flow-node code {
      color: var(--code-ink);
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 0.92em;
    }
    .mode-branch,
    .trade-branch {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      max-width: 980px;
      margin: 0 auto;
    }
    .mode-node,
    .trade-node {
      min-width: 0;
      padding: 14px 15px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--bg);
    }
    .mode-node:hover,
    .trade-node:hover { border-color: var(--primary); }
    .mode-name,
    .trade-name {
      display: block;
      color: var(--ink);
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 13px;
      font-weight: 700;
      line-height: 19px;
      overflow-wrap: anywhere;
    }
    .mode-copy,
    .trade-copy {
      display: block;
      margin-top: 5px;
      color: var(--body);
      font-size: 13px;
      line-height: 20px;
    }
    .flow-split-label {
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      line-height: 18px;
      text-align: center;
    }
    .directory-groups {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      column-gap: 46px;
      row-gap: 32px;
    }
    .directory-group {
      min-width: 0;
      padding-top: 12px;
      border-top: 1px solid var(--line-strong);
    }
    .directory-group h3 { margin-top: 0; }
    .directory-list {
      display: grid;
      gap: 2px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .directory-link {
      width: 100%;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 14px;
      border: 0;
      border-radius: 6px;
      padding: 7px 8px;
      background: transparent;
      color: var(--sidebar-ink);
      font-size: 14px;
      line-height: 20px;
      text-align: left;
      cursor: pointer;
    }
    .directory-link:hover { background: var(--surface-strong); color: var(--ink); }
    .directory-link::after {
      content: "→";
      flex: 0 0 auto;
      color: var(--muted);
    }
    [hidden] { display: none !important; }
    .document-header {
      max-width: var(--content-width);
      margin: 0 auto 52px;
      padding: 0;
      position: relative;
      left: var(--header-visual-offset);
    }
    .eyebrow { margin: 0 0 12px; color: var(--muted); font-size: 12px; font-weight: 700; line-height: 18px; }
    .document-header-row { display: block; }
    .document-title { margin: 0; color: var(--ink); font-size: var(--page-title-size); font-weight: 700; line-height: var(--page-title-line); letter-spacing: 0; text-wrap: balance; }
    .document-lead { max-width: 620px; margin: 10px 0 0; color: var(--body); font-size: 14px; font-weight: 400; line-height: 21px; }
    .document-meta {
      display: grid;
      grid-template-columns: 156px minmax(0, 1fr);
      gap: 28px;
      margin-top: 24px;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--line);
    }
    .document-meta-item { min-width: 0; }
    .document-meta-label {
      display: block;
      margin-bottom: 2px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      line-height: 16px;
    }
    .document-meta-value {
      display: block;
      color: var(--body);
      font-size: 12px;
      line-height: 18px;
      overflow-wrap: anywhere;
    }
    .document-path { font-family: inherit; text-wrap: pretty; }
    .document-path-segment { white-space: nowrap; }
    .document-footer {
      max-width: var(--content-width);
      margin: 52px auto 0;
      padding: 15px 0 0;
      border-top: 1px solid var(--line);
      position: relative;
      left: var(--body-visual-offset);
    }
    .document-updated { color: var(--muted); font-size: 12px; line-height: 18px; }

    .markdown-body { max-width: var(--content-width); margin: 0 auto; position: relative; left: var(--body-visual-offset); overflow-wrap: anywhere; }
    .markdown-body > h1:first-child { display: none; }
    .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {
      color: var(--ink);
      letter-spacing: 0;
      scroll-margin-top: calc(var(--toolbar-height) + 18px);
    }
    .markdown-body h2 { margin: 1.5em 0 0.45em; padding: 0; border: 0; font-size: var(--heading-1-size); font-weight: 600; line-height: var(--heading-1-line); }
    .markdown-body h3 { margin: 1.4em 0 0.4em; font-size: var(--heading-2-size); font-weight: 600; line-height: var(--heading-2-line); }
    .markdown-body h4 { margin: 1.25em 0 0.35em; font-size: var(--heading-3-size); font-weight: 600; line-height: var(--heading-3-line); }
    .markdown-body p { margin: 0.5em 0; }
    .markdown-body strong, .markdown-body b { color: var(--bold); font-weight: 760; }
    .markdown-body ul, .markdown-body ol { padding-left: 1.45em; }
    .markdown-body li + li { margin-top: 0.28em; }
    .markdown-body blockquote {
      margin: 1.2em 0;
      padding: 10px 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--body);
    }
    .markdown-body code {
      border: 0;
      border-radius: 6px;
      background: var(--inline-code-surface);
      padding: 0.08em 0.34em;
      color: var(--code-ink);
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 0.88em;
    }
    .markdown-body pre {
      overflow: auto;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--code-surface);
    }
    .markdown-body pre code { border: 0; background: transparent; padding: 0; color: var(--ink); }
    .markdown-body img {
      display: block;
      width: auto;
      max-width: 100%;
      height: auto;
      margin: 1.35em auto;
      border: 1px solid var(--line-strong);
      border-radius: 6px;
      background: var(--surface);
    }
    .markdown-body table {
      display: block;
      width: 100%;
      overflow-x: auto;
      border-collapse: collapse;
      margin: 1.25em 0;
      font-size: 13px;
    }
    .markdown-body th, .markdown-body td {
      min-width: 120px;
      padding: 9px 11px;
      border: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    .markdown-body th { background: var(--surface-strong); font-weight: 760; }
    .markdown-body tbody tr:nth-child(even) { background: var(--surface); }
    .markdown-body hr { border: 0; border-top: 1px solid var(--line); margin: 2em 0; }

    .outline {
      position: fixed;
      top: 50%;
      right: 16px;
      z-index: 30;
      width: 20px;
      overflow: visible;
      padding: 0;
      transform: translateY(-50%);
    }
    .outline-trigger {
      width: 100%;
      display: grid;
      align-content: start;
      justify-items: center;
      gap: 12px;
      padding: 0;
      border: 0;
      background: transparent;
      color: var(--outline-line);
      cursor: pointer;
    }
    .outline-trigger:hover { color: var(--ink); }
    .outline-trigger span {
      display: block;
      width: 20px;
      height: 2px;
      border-radius: 4px;
      background: currentColor;
      opacity: 0.9;
    }
    .outline-trigger span.active {
      background: var(--outline-active);
      box-shadow: 0 0 5px rgb(255 255 255 / 0.46);
      opacity: 1;
    }
    .outline-panel {
      position: absolute;
      top: 50%;
      right: 0;
      width: min(232px, calc(100vw - 84px));
      max-height: min(520px, calc(100vh - 72px));
      display: flex;
      flex-direction: column;
      padding: 12px;
      border: 1px solid #383836;
      border-radius: 14px;
      background: var(--outline-panel);
      box-shadow: var(--shadow);
      opacity: 0;
      pointer-events: none;
      transform: translateY(-50%);
      transition: opacity 140ms ease;
    }
    .outline-panel::after {
      content: "";
      position: absolute;
      top: 0;
      right: -28px;
      width: 28px;
      height: 100%;
    }
    .outline:hover .outline-panel, .outline:focus-within .outline-panel {
      opacity: 1;
      pointer-events: auto;
    }
    .outline-list { display: grid; gap: 2px; overflow-y: auto; padding-right: 4px; }
    .outline-link {
      display: block;
      padding: 5px 7px;
      border-radius: 5px;
      color: #b4b4b0;
      font-size: var(--outline-item-size);
      font-weight: 400;
      line-height: 19px;
      text-decoration: none;
    }
    .outline-text { min-width: 0; white-space: normal; overflow-wrap: anywhere; }
    .outline-link.level-3 { margin-left: 12px; }
    .outline-link.level-4 { margin-left: 24px; color: var(--muted); font-size: 12px; }
    .outline-link:hover, .outline-link.active { background: var(--outline-panel-hover); color: var(--outline-link-active); }
    @media (min-width: 1400px) {
      .main { padding: 45px 0 80px; }
      .document-header {
        width: min(var(--reference-header-width), calc(100% - var(--reference-header-inset) - 48px));
        max-width: none;
        margin: 0 0 30px var(--reference-header-inset);
        left: 0;
      }
      .eyebrow { margin-bottom: 14px; }
      .document-lead { margin-top: 12px; }
      .document-meta {
        position: relative;
        margin-top: 34px;
        border-bottom: 0;
      }
      .document-meta::after {
        content: "";
        position: absolute;
        right: -30px;
        bottom: 0;
        left: 0;
        height: 1px;
        background: var(--line);
      }
      .markdown-body {
        width: min(var(--content-width), calc(100% - var(--reference-body-inset) - 48px));
        max-width: none;
        margin: 0 0 0 var(--reference-body-inset);
        left: 0;
      }
      .markdown-body h2 {
        margin: 28px 0 20px;
        padding-bottom: 5px;
        font-size: 27px;
        font-weight: 700;
        line-height: 35px;
      }
      .document-footer {
        width: min(var(--content-width), calc(100% - var(--reference-body-inset) - 48px));
        max-width: none;
        margin: 52px 0 0 var(--reference-body-inset);
        left: 0;
      }
    }
    @media (max-width: 1100px) {
      .app-shell { grid-template-columns: var(--sidebar-width) minmax(0, 1fr); }
    }
    @media (max-width: 780px) {
      .app-shell { display: block; }
      .sidebar { position: static; width: 100%; height: auto; overflow: visible; border-right: 0; border-bottom: 1px solid var(--line); }
      .sidebar-header { padding: 16px 18px; }
      .search-region { padding: 16px 18px 0; border-bottom: 0; }
      .mobile-document-select { display: block; }
      .sidebar-navigation { display: none; }
      .content-toolbar { position: static; min-height: 50px; flex-wrap: wrap; padding: 10px 18px; }
      .toolbar-location { width: auto; }
      .main { padding: 24px 18px 60px; }
      .mode-branch, .trade-branch, .directory-groups { grid-template-columns: 1fr; }
      .mode-branch, .trade-branch { gap: 9px; }
      .flow-node, .mode-node, .trade-node { padding: 13px 14px; }
      .document-header, .markdown-body, .document-footer { left: 0; }
      .document-header { margin-bottom: 42px; }
      .document-title { font-size: 32px; }
      .document-lead { font-size: 14px; line-height: 21px; }
      .document-meta { grid-template-columns: 1fr; gap: 14px; margin-top: 20px; padding-bottom: 20px; }
      .markdown-body h2 { font-size: 22px; line-height: 29px; }
      .markdown-body h3 { font-size: 18px; line-height: 24px; }
      .markdown-body h4 { font-size: 16px; line-height: 22px; }
      .markdown-body table { width: calc(100vw - 36px); }
      .outline { right: 14px; }
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar" aria-label="Project documents">
      <header class="sidebar-header">
        <button type="button" class="brand" id="home-button" aria-label="返回项目首页" title="返回项目首页">
          <span class="brand-icon" aria-hidden="true"><span></span><span></span></span>
          <span class="sr-only" id="home-label">返回项目首页</span>
        </button>
        <div class="brand-copy">
          <p class="brand-title" id="project-name">Project Context</p>
          <p class="brand-subtitle" id="project-subtitle">项目文档与工作记录</p>
        </div>
      </header>
      <div class="search-region">
        <div class="search-wrap">
          <input id="search" type="search" autocomplete="off" placeholder="Search documents and content" aria-label="Search documents and content">
        </div>
      </div>
      <div class="mobile-document-select">
        <select id="document-select" aria-label="Select document"></select>
      </div>
      <nav class="sidebar-navigation" aria-label="Project documents">
        <div id="document-navigation"></div>
        <p class="nav-empty" id="nav-empty">No matching documents.</p>
      </nav>
    </aside>

    <section class="content-column">
      <header class="content-toolbar">
        <div class="toolbar-location"><strong id="toolbar-title"></strong></div>
        <div class="toolbar-actions">
          <button type="button" class="language-icon-button" id="language-toggle" aria-label="切换到英文" title="切换到英文">
            <span class="language-icon" aria-hidden="true">CN</span>
            <span class="sr-only">切换到英文</span>
          </button>
          <button type="button" class="theme-icon-button toolbar-theme-toggle" data-theme-toggle aria-label="切换黑白背景" title="切换黑白背景">
            <span class="theme-icon" aria-hidden="true">◐</span>
            <span class="sr-only">切换黑白背景</span>
          </button>
        </div>
      </header>
      <main class="main">
      <section class="overview-home" id="overview-home" aria-labelledby="overview-title"></section>
      <section class="document-view" id="document-view" hidden>
        <header class="document-header">
          <div class="eyebrow" id="document-group"></div>
          <div class="document-header-row">
            <h1 class="document-title" id="document-title"></h1>
            <p class="document-lead" id="document-lead"></p>
          </div>
          <div class="document-meta">
            <div class="document-meta-item">
              <strong class="document-meta-label" id="document-policy-label"></strong>
              <span class="document-meta-value" id="document-policy"></span>
            </div>
            <div class="document-meta-item">
              <strong class="document-meta-label" id="document-path-label"></strong>
              <span class="document-meta-value document-path" id="document-path"></span>
            </div>
          </div>
        </header>
        <article class="markdown-body" id="markdown-content"></article>
        <footer class="document-footer"><div class="document-updated" id="document-updated"></div></footer>
      </section>
      </main>
    </section>

    <aside class="outline" aria-label="Document outline">
      <button type="button" class="outline-trigger" id="outline-trigger" aria-label="本页目录" aria-expanded="false" title="本页目录">
      </button>
      <div class="outline-panel" id="outline-panel">
      <div class="outline-list" id="outline-list"></div>
      </div>
    </aside>
  </div>
  <script src="assets/marked.umd.js"></script>
  <script>
    const PROJECT_NAME = __PROJECT_NAME_JSON__;
    const MANIFEST = __MANIFEST_JSON__;
    const DOCUMENTS = __DOCS_JSON__;
    const GROUP_ORDER = Object.keys(MANIFEST.groups || {});
    const COPY = {
      en: {
        homeLabel: "Return to project home",
        search: "Search documents and content",
        themeToggle: "Toggle black and white background",
        languageToggle: "Switch to Chinese",
        empty: "No matching documents.",
        updatePolicy: "Update policy",
        filePath: "File path",
        modified: "Updated",
        subtitle: "Project documents and working records",
        homeOption: "Project guide",
        groups: { priority: "Read first", planning: "Goals and constraints", state: "Current state", work: "Active work", domain: "Domain rules", reference: "Project reference", archive: "History and archive" },
      },
      zh: {
        homeLabel: "返回项目首页",
        search: "搜索文档标题和正文",
        themeToggle: "切换黑白背景",
        languageToggle: "切换到英文",
        empty: "没有匹配的文档。",
        updatePolicy: "更新策略",
        filePath: "文件路径",
        modified: "更新时间",
        subtitle: "项目文档与工作记录",
        homeOption: "项目导览",
        groups: { priority: "优先阅读", planning: "目标与约束", state: "当前状态", work: "当前工作", domain: "领域规则", reference: "项目参考", archive: "历史与存档" },
      },
    };

    const HOME_CONTENT = {
      zh: {
        title: "项目导览",
        lead: "两张流程图分别说明 V4.4 怎样组织研究，以及一笔交易怎样从已完成的 15 秒 bar 进入逐笔记录。下方是完整的项目文档目录。",
        note: "本页供人阅读和导航。AI 的优先阅读顺序不包含本页；如有冲突，以当前项目规则、英文管理文档和真实产物为准。",
        policy: "供人阅读和导航；AI 阅读顺序不包含本页。",
        path: "project_management\\index.html#home.zh",
        programTitle: "回测程序与研究工作流",
        programLead: "参数从哪里来，由四个正式模式决定。模式选定后，计划冻结、计算、闭合、发布和解释都走同一套流程。完成一轮后，可以停止，也可以按证据选择下一种模式。",
        programSteps: [
          { title: "填写研究合同", copy: "指定品种、15 秒数据与样本区间、品种配置、成本、gap／低活跃规则、参数预算和本轮要回答的问题。", primary: true },
          { title: "选择正式模式", copy: "在读取本轮新证据前确定模式。运行开始后，模式名称和参数来源保持不变。", branchLabel: "四个正式模式", branches: [
            { name: "fresh_search", copy: "指定品种，从零开始做宽范围参数探索，不导入来源候选或父阶段。" },
            { name: "continuation_search", copy: "同一品种、同一排名谱系，根据已完成父阶段继续探索。" },
            { name: "transfer_exact", copy: "在目标结果出现前冻结来源品种候选，并在目标品种原样运行。" },
            { name: "target_local_refinement", copy: "完成精确迁移后，在冻结锚点周围运行由程序逐项校验的有限邻域。" }
          ] },
          { title: "冻结证据与执行计划", copy: "绑定源码、数据、品种配置、候选或父阶段、完整坐标、输出目录和资源，并完成唯一性检查及 completed＋active＋pending 反连接。" },
          { title: "验证并运行原始回测", copy: "核对计划身份和资源后分批运行。每个坐标只写入自己的阶段目录，已有阶段和历史结果保持不变。" },
          { title: "闭合原始结果", copy: "全部批次、坐标、逐笔数量和哈希一致后，标记为 IMMUTABLE_CLOSED。未闭合的结果不进入解释和累计排名。", primary: true },
          { title: "生成分析与累计 HTML", copy: "根据不可变逐笔记录计算指标，并刷新该品种唯一的累计总入口和共享逐笔入口。累计发布只保留一个写入者。" },
          { title: "读结果，决定下一轮", copy: "查看收益、回撤、单笔、交易数、稳定性和依赖性。可以到此停止，也可以另行授权 continuation、transfer、refinement 或 fresh campaign。参数不会自动进入接受状态。" }
        ],
        tradeTitle: "一笔交易的完整生命周期",
        tradeLead: "每一步只使用当时已经完成且可用的 15 秒原子。图中省略展示细节，只保留影响成交和退出的状态。",
        tradeSteps: [
          { title: "完成一根新的 15 秒 bar", copy: "确认连续交易段、真实／synthetic 状态，以及截至当前可以进入 baseline 的 TR 原子。", primary: true },
          { title: "空仓：寻找 H 和开仓条件", copy: "在 E 内寻找有序高点 H，用 H 以前的 BH／TRW 原子计算 baseline，再计算 H 到当前低点的 drop 和 K×baseline threshold。" },
          { title: "检查正值信号", copy: "baseline、drop、threshold 都必须有限且大于 0；drop ≥ threshold 时形成开仓信号，其余情况等待下一根 bar。" },
          { title: "完成开仓成交", copy: "真实 bar 触发时，若 open 已穿过理论价则按 open，否则按理论价；synthetic bar 的信号进入 pending，最多保留 120 根连续 bar，在第一根真实成交 bar 的 open 建仓。", branchLabel: "成交路径", branches: [
            { name: "real bar", copy: "open 穿价 → open；否则触发价。相等也成交。" },
            { name: "synthetic signal", copy: "保留信号 → 等第一根真实 bar 的 open。" }
          ] },
          { title: "逐 bar 更新持仓状态", copy: "从 H 开始维护 W 的最大净下跌基准，M 生成反弹退出线，S 判断下跌是否停止推进。" },
          { title: "选择退出路径", copy: "反弹、速度、非真实成交等待和样本末强平，各自遵守对应的因果成交规则。", branchLabel: "退出路径", branches: [
            { name: "rebound", copy: "open 已高于退出价按 open；否则 high 触发时按理论价。严格同 bar 新低例外按 close。" },
            { name: "speed", copy: "S 窗口确认下跌停止推进后退出。" },
            { name: "pending exit", copy: "非真实 bar 形成退出后，等待下一根真实成交 bar 的 open。" },
            { name: "sample end", copy: "样本结束仍持仓时，用样本末 close 强制平仓。" }
          ] },
          { title: "写入逐笔和组合结果", copy: "记录成交、收益、MFE／MAE、gap／synthetic／低活跃依赖和退出原因，再汇总为组合指标和 HTML 证据。", primary: true }
        ],
        directoryTitle: "项目文档目录",
        directoryLead: "目录根据项目管理清单生成。点击文档即可进入内容页；点击左上角项目图标可以回到这里。"
      },
      en: {
        title: "Project guide",
        lead: "Use two workflows to understand how V4.4 organizes research and how one trade moves from completed 15-second bars into its final transaction record. The directory below links every management document.",
        note: "This page is a human-facing navigation aid. It is not strategy, execution, or evidence authority and is not part of the AI priority reading order. Current project rules, English management documents, and real artifacts govern any discrepancy.",
        policy: "Human orientation and navigation; excluded from the AI reading order.",
        path: "project_management\\index.html#home.en",
        programTitle: "Backtest program and research workflow",
        programLead: "The four official modes define where parameters come from. After mode selection, every campaign shares the same plan-freeze, compute, closure, delivery, and interpretation pipeline. Later work either selects another authorized mode or stops.",
        programSteps: [
          { title: "Provide the research contract", copy: "Specify instrument, 15-second data and sample, instrument profile, cost, gap and low-activity policies, coordinate budget, and the round question.", primary: true },
          { title: "Choose one official mode", copy: "Freeze the mode before reading new round evidence; its parameter provenance cannot change during execution.", branchLabel: "Four official modes", branches: [
            { name: "fresh_search", copy: "Explore one named instrument from scratch with broad coverage and no imported candidates or parent stage." },
            { name: "continuation_search", copy: "Continue from a completed parent in the same instrument and ranking lineage." },
            { name: "transfer_exact", copy: "Freeze source-instrument candidates before target evidence, then run them unchanged on the target." },
            { name: "target_local_refinement", copy: "After exact transfer, run a machine-validated bounded neighborhood around frozen anchors." }
          ] },
          { title: "Freeze evidence and execution plan", copy: "Bind source, data, profile, candidate or parent, expanded coordinates, output root, and resources; pass uniqueness and completed+active+pending anti-join." },
          { title: "Validate and run raw compute", copy: "Validate plan identity and resources, then run resumable batches in an isolated stage root while preserving prior stages." },
          { title: "Close immutable raw evidence", copy: "Mark IMMUTABLE_CLOSED only after batches, coordinates, trades, and hashes reconcile. Partial results cannot be interpreted or ranked.", primary: true },
          { title: "Analyze and publish cumulative HTML", copy: "Build metrics from immutable trades and refresh the instrument's shared cumulative main and trade-review entries with one cumulative writer." },
          { title: "Interpret and decide", copy: "Report return, drawdown, median trade, frequency, stability, and dependencies. Stop or authorize a new continuation, transfer, refinement, or fresh campaign; no parameter is accepted automatically." }
        ],
        tradeTitle: "Lifecycle of one trade",
        tradeLead: "Every decision uses only completed and causally available 15-second atoms. Presentation details are omitted while fill and exit state remain explicit.",
        tradeSteps: [
          { title: "Complete a new 15-second bar", copy: "Resolve continuity, real versus synthetic status, and which TR atoms are available to the baseline at this time.", primary: true },
          { title: "Flat state: find H and entry inputs", copy: "Find ordered high H inside E, form the BH/TRW baseline from eligible completed atoms, and calculate H-to-current-low drop plus K×baseline threshold." },
          { title: "Positive signal gate", copy: "An entry signal requires finite baseline, drop, and threshold; all three must be above zero and drop must be at least the threshold. Otherwise remain flat." },
          { title: "Fill the entry", copy: "On a real trigger bar, an open through the theoretical price fills at open; otherwise fill at threshold. A synthetic signal is retained for up to 120 continuous bars and fills at the first real open.", branchLabel: "Fill paths", branches: [
            { name: "real bar", copy: "Open through price → open; otherwise threshold. Equality fills." },
            { name: "synthetic signal", copy: "Retain signal → first real bar open." }
          ] },
          { title: "Update the open position", copy: "Maintain the maximum W net-decline baseline from H, convert it through M into a rebound line, and let S detect when decline stops advancing." },
          { title: "Choose an exit path", copy: "Rebound, speed, non-real pending, and sample-end exits each retain their causal fill rule.", branchLabel: "Exit paths", branches: [
            { name: "rebound", copy: "Open above exit fills at open; otherwise a high trigger fills at theory. A strict same-bar new-low exception fills at close." },
            { name: "speed", copy: "Exit when the S window confirms that decline has stopped advancing." },
            { name: "pending exit", copy: "A non-real exit waits for the next real-trade open." },
            { name: "sample end", copy: "Any remaining position closes at the declared sample-end close." }
          ] },
          { title: "Write trade and coordinate evidence", copy: "Record fills, return, MFE/MAE, gap/synthetic/low-activity dependence, and exit reason, then aggregate coordinate metrics and HTML evidence.", primary: true }
        ],
        directoryTitle: "Project document directory",
        directoryLead: "The directory is generated from the project-management manifest. Open any formal document below; use the project icon at top left to return here."
      }
    };

    const state = {
      language: "zh",
      pair: "agent-rules",
      view: "home",
      query: "",
      theme: document.documentElement.dataset.theme === "light" ? "light" : "dark",
    };
    const byId = new Map(DOCUMENTS.map(function (doc) { return [doc.id, doc]; }));
    const content = document.getElementById("markdown-content");
    const navigation = document.getElementById("document-navigation");
    const documentSelect = document.getElementById("document-select");
    const search = document.getElementById("search");
    const homeButton = document.getElementById("home-button");
    const languageToggle = document.getElementById("language-toggle");
    const overviewHome = document.getElementById("overview-home");
    const documentView = document.getElementById("document-view");
    const outlineShell = document.querySelector(".outline");
    const outlineTrigger = document.getElementById("outline-trigger");
    let outlineEntries = [];
    let outlineFrame = 0;

    marked.setOptions({ gfm: true, breaks: false });

    function currentDocument() {
      return byId.get(state.pair + "." + state.language) || byId.get("agent-rules." + state.language) || localizedDocuments()[0];
    }

    function groupLabel(group) {
      const labels = MANIFEST.groups && MANIFEST.groups[group];
      return (labels && (labels[state.language] || labels.en || labels.zh)) || COPY[state.language].groups[group] || group;
    }

    function localizedDocuments() {
      return DOCUMENTS.filter(function (doc) { return doc.lang === state.language; });
    }

    function slug(text, index) {
      const value = text.toLowerCase().trim().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "");
      return value || "section-" + index;
    }

    function renderOutline(root) {
      const outline = document.getElementById("outline-list");
      outline.replaceChildren();
      outlineEntries = [];
      Array.from((root || content).querySelectorAll("h2, h3, h4")).forEach(function (heading, index) {
        heading.id = slug(heading.textContent, index);
        const link = document.createElement("a");
        link.className = "outline-link level-" + heading.tagName.slice(1);
        link.href = "#" + heading.id;
        const text = document.createElement("span");
        text.className = "outline-text";
        text.textContent = heading.textContent;
        link.append(text);
        link.addEventListener("click", function (event) {
          event.preventDefault();
          heading.scrollIntoView({ block: "start" });
          link.blur();
        });
        outline.appendChild(link);
        outlineEntries.push({ heading: heading, link: link });
      });
      const rail = document.createDocumentFragment();
      outlineEntries.forEach(function () { rail.appendChild(document.createElement("span")); });
      outlineTrigger.replaceChildren(rail);
      scheduleOutlineUpdate();
    }

    function updateOutlineState() {
      outlineFrame = 0;
      if (!outlineEntries.length) return;
      const marker = 110;
      let activeIndex = 0;
      outlineEntries.forEach(function (entry, index) {
        if (entry.heading.getBoundingClientRect().top <= marker) activeIndex = index;
      });
      outlineEntries.forEach(function (entry, index) {
        entry.link.classList.toggle("active", index === activeIndex);
      });
      const bars = Array.from(outlineTrigger.children);
      bars.forEach(function (bar, index) {
        bar.classList.toggle("active", index === activeIndex);
      });
    }

    function scheduleOutlineUpdate() {
      if (outlineFrame) return;
      outlineFrame = requestAnimationFrame(updateOutlineState);
    }

    function renderNavigation() {
      const copy = COPY[state.language];
      const query = state.query.trim().toLocaleLowerCase();
      navigation.replaceChildren();
      documentSelect.replaceChildren();
      let matchCount = 0;

      const homeOption = document.createElement("option");
      homeOption.value = "__home__";
      homeOption.textContent = copy.homeOption;
      homeOption.selected = state.view === "home";
      documentSelect.appendChild(homeOption);

      const homeSection = document.createElement("section");
      homeSection.className = "nav-group home-nav-group";
      const homeList = document.createElement("div");
      homeList.className = "nav-list";
      const homeItem = document.createElement("button");
      homeItem.type = "button";
      homeItem.className = "nav-item" + (state.view === "home" ? " active" : "");
      homeItem.innerHTML = '<span class="nav-marker"></span><span></span>';
      homeItem.lastElementChild.textContent = copy.homeOption;
      homeItem.addEventListener("click", goHome);
      homeList.appendChild(homeItem);
      homeSection.appendChild(homeList);
      navigation.appendChild(homeSection);
      matchCount += 1;

      GROUP_ORDER.forEach(function (group) {
        const docs = localizedDocuments().filter(function (doc) { return doc.group === group; });
        const matchingDocs = docs.filter(function (doc) {
          const haystack = (doc.title + " " + doc.path + " " + doc.markdown).toLocaleLowerCase();
          return !query || haystack.includes(query);
        });
        if (!matchingDocs.length) return;
        const section = document.createElement("section");
        section.className = "nav-group";
        const heading = document.createElement("h2");
        heading.className = "nav-group-title";
        heading.textContent = groupLabel(group);
        const list = document.createElement("div");
        list.className = "nav-list";

        matchingDocs.forEach(function (doc) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "nav-item" + (state.view === "document" && doc.pair === state.pair ? " active" : "");
          button.innerHTML = '<span class="nav-marker"></span><span></span>';
          button.lastElementChild.textContent = doc.title;
          button.addEventListener("click", function () { selectPair(doc.pair); });
          list.appendChild(button);

          const option = document.createElement("option");
          option.value = doc.pair;
          option.textContent = groupLabel(group) + " · " + doc.title;
          option.selected = state.view === "document" && doc.pair === state.pair;
          documentSelect.appendChild(option);
          matchCount += 1;
        });

        section.append(heading, list);
        navigation.appendChild(section);
      });

      document.getElementById("nav-empty").style.display = matchCount ? "none" : "block";
    }

    function sanitizeRenderedContent() {
      content.querySelectorAll("script, iframe, object, embed, link, style").forEach(function (element) { element.remove(); });
      content.querySelectorAll("*").forEach(function (element) {
        Array.from(element.attributes).forEach(function (attribute) {
          if (/^on/i.test(attribute.name)) element.removeAttribute(attribute.name);
        });
        ["href", "src"].forEach(function (attributeName) {
          const value = element.getAttribute(attributeName);
          if (value && /^\s*javascript:/i.test(value)) element.removeAttribute(attributeName);
        });
      });
    }

    function makeElement(tag, className, text) {
      const element = document.createElement(tag);
      if (className) element.className = className;
      if (text != null) element.textContent = text;
      return element;
    }

    function updateChrome(copy, title) {
      document.getElementById("project-name").textContent = PROJECT_NAME;
      document.getElementById("project-subtitle").textContent = copy.subtitle;
      document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
      homeButton.setAttribute("aria-label", copy.homeLabel);
      homeButton.setAttribute("title", copy.homeLabel);
      document.getElementById("home-label").textContent = copy.homeLabel;
      search.placeholder = copy.search;
      search.setAttribute("aria-label", copy.search);
      document.getElementById("nav-empty").textContent = copy.empty;
      document.getElementById("toolbar-title").textContent = title;
      document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
        button.setAttribute("aria-label", copy.themeToggle);
        button.setAttribute("title", copy.themeToggle);
        button.setAttribute("aria-pressed", String(state.theme === "light"));
        button.querySelector(".sr-only").textContent = copy.themeToggle;
        button.querySelector(".theme-icon").textContent = state.theme === "dark" ? "◐" : "◑";
      });
      languageToggle.setAttribute("aria-label", copy.languageToggle);
      languageToggle.setAttribute("title", copy.languageToggle);
      languageToggle.querySelector(".sr-only").textContent = copy.languageToggle;
      languageToggle.querySelector(".language-icon").textContent = state.language === "zh" ? "CN" : "EN";
    }

    function buildFlow(specification) {
      const track = makeElement("ol", "flow-track");
      specification.forEach(function (step) {
        const item = makeElement("li", "flow-step");
        const node = makeElement("div", "flow-node" + (step.primary ? " flow-primary" : ""));
        node.append(
          makeElement("h3", "flow-node-title", step.title),
          makeElement("p", "flow-node-copy", step.copy)
        );
        item.appendChild(node);
        if (step.branches && step.branches.length) {
          item.appendChild(makeElement("p", "flow-split-label", step.branchLabel));
          const branch = makeElement("div", step.branches.length > 2 ? "trade-branch" : "mode-branch");
          step.branches.forEach(function (entry) {
            const branchNode = makeElement("div", step.branches.length > 2 ? "trade-node" : "mode-node");
            branchNode.append(
              makeElement("span", step.branches.length > 2 ? "trade-name" : "mode-name", entry.name),
              makeElement("span", step.branches.length > 2 ? "trade-copy" : "mode-copy", entry.copy)
            );
            branch.appendChild(branchNode);
          });
          item.appendChild(branch);
        }
        track.appendChild(item);
      });
      return track;
    }

    function buildOverviewSection(title, lead, steps) {
      const section = makeElement("section", "overview-section");
      const header = makeElement("header", "overview-section-header");
      header.append(
        makeElement("h2", "", title),
        makeElement("p", "", lead)
      );
      section.append(header, buildFlow(steps));
      return section;
    }

    function buildDirectory(home) {
      const section = makeElement("section", "overview-section");
      const header = makeElement("header", "overview-section-header");
      header.append(
        makeElement("h2", "", home.directoryTitle),
        makeElement("p", "", home.directoryLead)
      );
      const groups = makeElement("div", "directory-groups");
      GROUP_ORDER.forEach(function (group) {
        const docs = localizedDocuments().filter(function (doc) { return doc.group === group; });
        if (!docs.length) return;
        const block = makeElement("section", "directory-group");
        block.appendChild(makeElement("h3", "", groupLabel(group)));
        const list = makeElement("ul", "directory-list");
        docs.forEach(function (doc) {
          const item = document.createElement("li");
          const button = makeElement("button", "directory-link", doc.title);
          button.type = "button";
          button.addEventListener("click", function () { selectPair(doc.pair); });
          item.appendChild(button);
          list.appendChild(item);
        });
        block.appendChild(list);
        groups.appendChild(block);
      });
      section.append(header, groups);
      return section;
    }

    function renderHome() {
      const copy = COPY[state.language];
      const home = HOME_CONTENT[state.language];
      state.view = "home";
      overviewHome.hidden = true;
      documentView.hidden = false;
      updateChrome(copy, home.title);
      document.getElementById("document-group").textContent = groupLabel("priority");
      document.getElementById("document-title").textContent = home.title;
      document.getElementById("document-lead").textContent = home.lead;
      document.getElementById("document-policy-label").textContent = copy.updatePolicy;
      document.getElementById("document-policy").textContent = home.policy;
      document.getElementById("document-path-label").textContent = copy.filePath;
      renderDocumentPath(home.path);
      document.getElementById("document-updated").textContent = "";
      const note = makeElement("blockquote", "overview-note");
      note.appendChild(makeElement("p", "", home.note));
      content.replaceChildren(
        note,
        buildOverviewSection(home.programTitle, home.programLead, home.programSteps),
        buildOverviewSection(home.tradeTitle, home.tradeLead, home.tradeSteps),
        buildDirectory(home)
      );
      renderOutline(content);
      renderNavigation();
      document.title = home.title + " · " + PROJECT_NAME;
    }

    function renderDocumentPath(path) {
      const documentPath = document.getElementById("document-path");
      documentPath.replaceChildren();
      path.replaceAll("/", "\\").split("\\").forEach(function (segment, index, segments) {
        const pathSegment = document.createElement("span");
        pathSegment.className = "document-path-segment";
        pathSegment.textContent = (index > 0 ? "\\" : "") + segment;
        documentPath.appendChild(pathSegment);
        if (index < segments.length - 1) documentPath.appendChild(document.createElement("wbr"));
      });
    }

    function renderDocument() {
      const doc = currentDocument();
      const copy = COPY[state.language];
      state.view = "document";
      overviewHome.hidden = true;
      documentView.hidden = false;
      updateChrome(copy, doc.title);
      document.getElementById("document-group").textContent = groupLabel(doc.group);
      document.getElementById("document-title").textContent = doc.title;
      document.getElementById("document-lead").textContent = doc.lead || doc.title;
      document.getElementById("document-policy-label").textContent = copy.updatePolicy;
      document.getElementById("document-policy").textContent = doc.policy || "";
      document.getElementById("document-path-label").textContent = copy.filePath;
      renderDocumentPath(doc.path);
      document.getElementById("document-updated").textContent = doc.modified
        ? copy.modified + " " + new Date(doc.modified).toLocaleString(state.language === "zh" ? "zh-CN" : "en-GB", { dateStyle: "medium", timeStyle: "short" })
        : "";
      content.innerHTML = marked.parse(doc.markdown);
      sanitizeRenderedContent();
      const documentDirectory = doc.path.replace(/[^/]+$/, "");
      content.querySelectorAll("img[src]").forEach(function (image) {
        const source = image.getAttribute("src");
        if (source && !/^(?:[a-z]+:|\/|#)/i.test(source)) {
          image.setAttribute("src", documentDirectory + source);
        }
      });
      content.querySelectorAll("a").forEach(function (link) {
        if (/^https?:/i.test(link.href)) {
          link.target = "_blank";
          link.rel = "noreferrer";
        }
      });
      renderOutline(content);
      renderNavigation();
      document.title = doc.title + " · " + PROJECT_NAME;
    }

    function renderCurrentView() {
      if (state.view === "home") renderHome();
      else renderDocument();
    }

    function selectPair(pair) {
      state.pair = pair;
      state.view = "document";
      history.replaceState(null, "", "#" + pair + "." + state.language);
      renderDocument();
      window.scrollTo({ top: 0, behavior: "auto" });
    }

    function goHome() {
      state.view = "home";
      state.query = "";
      search.value = "";
      history.replaceState(null, "", "#home." + state.language);
      renderHome();
      window.scrollTo({ top: 0, behavior: "auto" });
    }

    function selectTheme(theme) {
      state.theme = theme === "light" ? "light" : "dark";
      document.documentElement.dataset.theme = state.theme;
      try {
        localStorage.setItem("project-management-theme", state.theme);
      } catch (error) {
        // The page still works when local storage is blocked.
      }
      renderCurrentView();
    }

    function toggleTheme() {
      selectTheme(state.theme === "dark" ? "light" : "dark");
    }

    function toggleLanguage() {
      state.language = state.language === "zh" ? "en" : "zh";
      history.replaceState(null, "", "#" + (state.view === "home" ? "home" : state.pair) + "." + state.language);
      renderCurrentView();
    }

    document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
      button.addEventListener("click", toggleTheme);
    });
    languageToggle.addEventListener("click", toggleLanguage);
    homeButton.addEventListener("click", goHome);
    outlineTrigger.addEventListener("pointerdown", function (event) {
      event.preventDefault();
    });
    outlineShell.addEventListener("pointerenter", function () { outlineTrigger.setAttribute("aria-expanded", "true"); });
    outlineShell.addEventListener("pointerleave", function () {
      if (!outlineShell.matches(":focus-within")) outlineTrigger.setAttribute("aria-expanded", "false");
    });
    outlineShell.addEventListener("focusin", function () { outlineTrigger.setAttribute("aria-expanded", "true"); });
    outlineShell.addEventListener("focusout", function () {
      requestAnimationFrame(function () {
        if (!outlineShell.matches(":focus-within")) outlineTrigger.setAttribute("aria-expanded", "false");
      });
    });
    window.addEventListener("scroll", scheduleOutlineUpdate, { passive: true });
    documentSelect.addEventListener("change", function () {
      if (documentSelect.value === "__home__") goHome();
      else selectPair(documentSelect.value);
    });
    search.addEventListener("input", function () {
      state.query = search.value;
      renderNavigation();
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "/" && document.activeElement !== search) {
        event.preventDefault();
        search.focus();
      }
      if (event.key === "Escape" && document.activeElement === search) {
        search.value = "";
        state.query = "";
        search.blur();
        renderNavigation();
      }
    });

    const hashMatch = location.hash.match(/^#([a-z0-9-]+)\.(zh|en)$/);
    if (hashMatch && hashMatch[1] === "home") {
      state.view = "home";
      state.language = hashMatch[2];
    } else if (hashMatch && byId.has(hashMatch[1] + "." + hashMatch[2])) {
      state.view = "document";
      state.pair = hashMatch[1];
      state.language = hashMatch[2];
    }
    renderCurrentView();
  </script>
</body>
</html>`
    .replace("__PROJECT_NAME_JSON__", safeJson(projectName))
    .replace("__MANIFEST_JSON__", safeJson(manifest))
    .replace("__DOCS_JSON__", safeJson(documents));

  if (!html.includes(generatedMarker)) throw new Error("Generated dashboard is missing its ownership marker.");
  if (/__(?:PROJECT_NAME|MANIFEST|DOCS)_JSON__/.test(html)) throw new Error("Dashboard data replacement failed.");

  const outputPath = options.output ? path.resolve(options.output) : path.join(managementRoot, "index.html");
  const type = await pathType(outputPath);
  let status = "created";
  let writtenPath = outputPath;
  if (type === "file") {
    const existing = await fs.readFile(outputPath, "utf8");
    if (existing === html) status = "unchanged";
    else if (options.force || existing.includes(generatedMarker)) status = "updated";
    else {
      status = "proposal";
      writtenPath = await nextProposal(outputPath);
    }
  } else if (type !== "missing") {
    status = "proposal";
    writtenPath = await nextProposal(outputPath);
  }

  if (status !== "unchanged") {
    await fs.mkdir(path.dirname(writtenPath), { recursive: true });
    await fs.writeFile(writtenPath, html, status === "proposal" ? { encoding: "utf8", flag: "wx" } : "utf8");
  }

  console.log(JSON.stringify({
    ok: true,
    status,
    outputPath: writtenPath,
    protectedTarget: status === "proposal" ? outputPath : null,
    documents: documents.length,
    missingDocuments: documents.filter((document) => document.missing).map((document) => document.path),
  }));
}

main().catch((error) => {
  console.error(JSON.stringify({ ok: false, error: error.message }));
  process.exitCode = 1;
});
