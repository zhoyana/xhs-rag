"use strict";

// 同源调用后端：页面由后端托管在 /ui，API 也在同一 origin，无需 CORS。
const api = async (method, path, body) => {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* 空响应也允许 */ }
  if (!res.ok) {
    const detail = (data && (data.detail || data.error)) || `请求失败（${res.status}）`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
};

// SSE 走 POST，所以不能用 EventSource（它只支持 GET）。手动读 body 流并切帧。
const streamRequest = async (path, body, onEvent, signal) => {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail = `请求失败（${res.status}）`;
    try {
      const data = await res.json();
      if (data && (data.detail || data.error)) detail = data.detail || data.error;
    } catch (_) { /* 保持默认提示 */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE 以空行分帧；最后一段可能不完整，留在 buffer 里等下一块。
    const frames = buffer.split("\n\n");
    buffer = frames.pop();
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try { onEvent(JSON.parse(line.slice(5).trim())); } catch (_) { /* 忽略坏帧 */ }
    }
  }
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const escapeHtml = (s) => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

const fmtTime = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString("zh-CN", { hour12: false });
};

const relativeTime = (iso) => {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  if (isNaN(t)) return "";
  const diff = (Date.now() - t) / 1000;
  if (diff < 60) return "刚刚";
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)} 天前`;
  return new Date(iso).toLocaleDateString("zh-CN");
};

let toastTimer = null;
const toast = (msg, isError = false) => {
  const el = $("#toast");
  el.textContent = msg;
  el.style.background = isError ? "#d03050" : "";
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
};

// ---------------------------------------------------------------- Markdown
// 只实现模型实际会用的子集：标题、列表、引用、表格、代码块、行内样式。
// 先摘出代码块与行内代码做占位，避免里面的 * # | 被当成 Markdown 语法。
function mdInline(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/~~([^~]+)~~/g, "<del>$1</del>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g,
      '$1<a href="$2" target="_blank" rel="noreferrer">$2</a>');
}

function renderMarkdown(text) {
  const src = String(text == null ? "" : text).replace(/\r\n?/g, "\n");
  const blocks = [];
  const codes = [];
  let body = src.replace(/```([^\n]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const label = lang.trim();
    blocks.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  body = body.replace(/`([^`\n]+)`/g, (_, code) => {
    codes.push(`<code>${escapeHtml(code)}</code>`);
    return `\u0001${codes.length - 1}\u0001`;
  });

  const lines = escapeHtml(body).split("\n");
  const html = [];
  let para = [];
  let list = null;
  const flushPara = () => {
    if (para.length) { html.push(`<p>${para.join("<br>")}</p>`); para = []; }
  };
  const closeList = () => { if (list) { html.push(`</${list}>`); list = null; } };
  const openList = (kind) => {
    if (list !== kind) { closeList(); html.push(`<${kind}>`); list = kind; }
  };
  const cells = (row) => row.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (!trimmed) { flushPara(); closeList(); continue; }
    if (/^\u0000\d+\u0000$/.test(trimmed)) { flushPara(); closeList(); html.push(trimmed); continue; }

    let m;
    if ((m = trimmed.match(/^(#{1,6})\s+(.*)$/))) {
      flushPara(); closeList();
      const level = Math.min(m[1].length, 3);
      html.push(`<h${level}>${mdInline(m[2])}</h${level}>`);
      continue;
    }
    if (/^([-*_])\s*\1\s*\1[\s\S]*$/.test(trimmed) && trimmed.replace(/[-*_\s]/g, "") === "") {
      flushPara(); closeList(); html.push("<hr />"); continue;
    }
    if (/^&gt;\s?/.test(trimmed)) {
      flushPara(); closeList();
      html.push(`<blockquote>${mdInline(trimmed.replace(/^&gt;\s?/, ""))}</blockquote>`);
      continue;
    }
    // 表格：本行含竖线，且下一行是 |---| 分隔行时才认定。
    if (trimmed.includes("|") && i + 1 < lines.length) {
      const sep = lines[i + 1].trim();
      if (/^[|\s:-]+$/.test(sep) && sep.includes("-")) {
        flushPara(); closeList();
        const head = cells(trimmed);
        let t = `<table><thead><tr>${head.map((c) => `<th>${mdInline(c)}</th>`).join("")}</tr></thead><tbody>`;
        i += 2;
        for (; i < lines.length && lines[i].trim().includes("|"); i++) {
          t += `<tr>${cells(lines[i].trim()).map((c) => `<td>${mdInline(c)}</td>`).join("")}</tr>`;
        }
        i--;
        html.push(`${t}</tbody></table>`);
        continue;
      }
    }
    if ((m = trimmed.match(/^[-*+]\s+(.*)$/))) {
      flushPara(); openList("ul"); html.push(`<li>${mdInline(m[1])}</li>`); continue;
    }
    if ((m = trimmed.match(/^\d+[.)]\s+(.*)$/))) {
      flushPara(); openList("ol"); html.push(`<li>${mdInline(m[1])}</li>`); continue;
    }
    closeList();
    para.push(mdInline(trimmed));
  }
  flushPara(); closeList();

  let out = html.join("\n");
  out = out.replace(/\u0000(\d+)\u0000/g, (_, n) => blocks[Number(n)]);
  out = out.replace(/\u0001(\d+)\u0001/g, (_, n) => codes[Number(n)]);
  return out;
}

// ----------------------------- 标签切换 -----------------------------
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    $$(".view").forEach((v) => v.classList.remove("active"));
    tab.classList.add("active");
    $(`#view-${tab.dataset.view}`).classList.add("active");
  });
});

// ----------------------------- 笔记库 -----------------------------
let currentNoteId = null;
const noteTitles = {};   // note_id → 标题，用于把对话里的引用 ID 显示成人看得懂的名字

async function loadNotes() {
  const data = await api("GET", "/notes?limit=100");
  const list = $("#note-list");
  list.innerHTML = "";
  if (!data.items.length) {
    list.innerHTML = `<li class="muted">还没有笔记，先粘贴链接批量导入。</li>`;
    return;
  }
  for (const n of data.items) {
    noteTitles[n.note_id] = n.title || "";
    const li = document.createElement("li");
    const hasImg = (n.image_text || "").trim() ? '<span class="badge img">图</span>' : "";
    li.innerHTML = `<div class="note-title">${escapeHtml(n.title || "(无标题)")}</div>
      <div class="note-sub">${fmtTime(n.created_at)} ${hasImg}</div>`;
    li.addEventListener("click", () => openNote(n.note_id, li));
    list.appendChild(li);
  }
}

async function openNote(noteId, li) {
  currentNoteId = noteId;
  $$("#note-list li").forEach((x) => x.classList.remove("active"));
  if (li) li.classList.add("active");
  const note = await api("GET", `/notes/${noteId}`);
  const detail = $("#note-detail");
  const badges = [];
  if ((note.image_text || "").trim()) badges.push('<span class="badge img">含图片文字</span>');
  (note.warnings || []).forEach((w) => badges.push(`<span class="badge warn">${escapeHtml(w)}</span>`));

  let html = `<h2>${escapeHtml(note.title || "(无标题)")}</h2>
    <div class="note-sub">${note.note_id} · ${fmtTime(note.created_at)}</div>
    <div style="margin:8px 0">${badges.join("")}</div>
    <div class="detail-actions">
      <button id="act-read" class="primary">补读图片</button>
      <button id="act-download" class="ghost">下载图片</button>
      <button id="act-analyze" class="ghost">AI 分析</button>
    </div>`;

  html += `<div class="section-h">正文</div><pre class="pre">${escapeHtml(note.content || "(无)")}</pre>`;
  if ((note.image_text || "").trim())
    html += `<div class="section-h">图片文字</div><pre class="pre">${escapeHtml(note.image_text)}</pre>`;

  // 异步补充：分析 / 图片 / 收藏夹 / 评论
  detail.innerHTML = html;
  $("#act-read").onclick = () => readImages(noteId);
  $("#act-download").onclick = () => downloadAssets(noteId);
  $("#act-analyze").onclick = () => analyzeNote(noteId);

  loadAnalysis(noteId, detail);
  loadAssets(noteId, detail);
  loadFoldersForNote(noteId, detail);
  renderComments(note, detail);
}

function renderComments(note, detail) {
  const comments = note.comments;
  if (!comments || !comments.list || !comments.list.length) return;
  let html = `<div class="section-h">评论（${comments.list.length} 条，不进检索，仅供精读）</div>`;
  for (const c of comments.list) {
    const ip = c.ip_location ? ` · ${escapeHtml(c.ip_location)}` : "";
    const like = c.like_count ? ` · ${escapeHtml(c.like_count)} 赞` : "";
    html += `<div class="pre" style="margin-bottom:6px"><b>${escapeHtml(c.author || "匿名")}</b>${ip}${like}
${escapeHtml(c.content || "")}</div>`;
    (c.subComments || []).forEach((s) => {
      html += `<div class="pre" style="margin:2px 0 6px 16px;background:var(--panel)"><b>${escapeHtml(s.author || "匿名")}</b>：${escapeHtml(s.content || "")}</div>`;
    });
  }
  detail.insertAdjacentHTML("beforeend", html);
}

async function loadAnalysis(noteId, detail) {
  try {
    const a = await api("GET", `/notes/${noteId}/analysis`);
    const text = JSON.stringify(a.analysis || a, null, 2);
    detail.insertAdjacentHTML("beforeend",
      `<div class="section-h">AI 分析</div><pre class="pre">${escapeHtml(text)}</pre>`);
  } catch (_) { /* 尚未分析 */ }
}

async function loadAssets(noteId, detail) {
  try {
    const d = await api("GET", `/notes/${noteId}/assets`);
    const items = (d.items || []).filter((x) => x.file_url);
    if (!items.length) return;
    let html = `<div class="section-h">图片（${items.length} 张）</div>`;
    for (const it of items) html += `<img class="asset" src="${it.file_url}" alt="笔记图片" />`;
    detail.insertAdjacentHTML("beforeend", html);
  } catch (_) { /* 无图片 */ }
}

async function loadFoldersForNote(noteId, detail) {
  try {
    const d = await api("GET", `/notes/${noteId}/folders`);
    if (!d.items || !d.items.length) return;
    const names = d.items.map((f) => escapeHtml(f.name)).join("、");
    detail.insertAdjacentHTML("beforeend",
      `<div class="section-h">收藏夹</div><div class="note-sub">${names}</div>`);
  } catch (_) { /* 未收藏 */ }
}

async function readImages(noteId) {
  const force = confirm("补读图片会调用视觉模型（消耗额度）。确定继续？");
  if (!force) return;
  try {
    const r = await api("POST", `/notes/${noteId}/read-images?force=true`);
    toast(`补读完成：读出 ${r.read || 0} 张图片`);
    loadNotes();
    openNote(noteId);
  } catch (e) { toast(e.message, true); }
}

async function downloadAssets(noteId) {
  try {
    const r = await api("POST", `/notes/${noteId}/assets/download`);
    toast(`下载完成：本地下载 ${r.downloaded || 0} 张，跳过 ${r.skipped || 0} 张`);
    openNote(noteId);
  } catch (e) { toast(e.message, true); }
}

async function analyzeNote(noteId) {
  try {
    await api("POST", `/notes/${noteId}/analyze?force=true`);
    toast("分析完成");
    openNote(noteId);
  } catch (e) { toast(e.message, true); }
}

async function importBatch() {
  const text = $("#import-box").value.trim();
  if (!text) { toast("请先粘贴链接", true); return; }
  $("#btn-import").disabled = true;
  try {
    const r = await api("POST", "/notes/import-batch", { share_text: text });
    $("#import-result").textContent =
      `成功 ${r.succeeded} · 失败 ${r.failed} · 重复 ${r.duplicates}`;
    toast(`导入成功 ${r.succeeded} 条`);
    loadNotes();
  } catch (e) { toast(e.message, true); }
  finally { $("#btn-import").disabled = false; }
}

async function readAllImages() {
  if (!confirm("将遍历全部已入库笔记，为已下载图片的笔记补读图（消耗较多视觉模型额度）。确定？")) return;
  try {
    const r = await api("POST", "/notes/read-images", { all: true });
    toast(`批量补读完成：成功 ${r.succeeded} / 共 ${r.total}`);
  } catch (e) { toast(e.message, true); }
}

// ----------------------------- 收藏夹 -----------------------------
async function loadFolders() {
  const d = await api("GET", "/folders");
  const list = $("#folder-list");
  list.innerHTML = "";
  if (!d.items.length) { list.innerHTML = `<li class="muted">还没有收藏夹。</li>`; return; }
  for (const f of d.items) {
    const li = document.createElement("li");
    const color = f.color ? `<span style="color:${escapeHtml(f.color)}">●</span> ` : "";
    li.innerHTML = `<div class="note-title">${color}${escapeHtml(f.name)}</div>
      <div class="note-sub">${escapeHtml(f.description || "")} · ${(f.note_count ?? 0)} 篇</div>`;
    li.addEventListener("click", () => openFolder(f.id, f.name, li));
    list.appendChild(li);
  }
}

async function openFolder(folderId, name, li) {
  $$("#folder-list li").forEach((x) => x.classList.remove("active"));
  if (li) li.classList.add("active");
  const d = await api("GET", `/folders/${folderId}/notes?limit=100`);
  const box = $("#folder-notes");
  if (!d.items.length) { box.innerHTML = `<p class="muted">「${escapeHtml(name)}」还没有笔记。</p>`; return; }
  let html = `<h3>${escapeHtml(name)}</h3><ul class="note-list">`;
  for (const n of d.items) {
    html += `<li><div class="note-title">${escapeHtml(n.title || "(无标题)")}</div>
      <div class="note-sub">${fmtTime(n.created_at)}</div></li>`;
  }
  html += `</ul>`;
  box.innerHTML = html;
}

async function newFolder() {
  const name = $("#folder-name").value.trim();
  if (!name) { toast("请填写收藏夹名称", true); return; }
  try {
    await api("POST", "/folders", { name, description: $("#folder-desc").value.trim() });
    $("#folder-name").value = ""; $("#folder-desc").value = "";
    toast("收藏夹已创建");
    loadFolders();
  } catch (e) { toast(e.message, true); }
}

// ----------------------------- 检索 -----------------------------
async function doSearch() {
  const query = $("#search-box").value.trim();
  if (!query) { toast("请输入检索问题", true); return; }
  const mode = $("#search-mode").value;
  const list = $("#search-results");
  list.innerHTML = `<li class="muted">检索中…</li>`;
  try {
    const r = await api("POST", "/search", { query, mode, top_k: 8 });
    list.innerHTML = "";
    if (!r.items.length) { list.innerHTML = `<li class="muted">没有命中，换个问法试试。</li>`; return; }
    for (const it of r.items) {
      const li = document.createElement("li");
      li.innerHTML = `<div class="note-title">${escapeHtml(it.title || it.note_id)}</div>
        <div class="note-sub">相关度 ${it.score != null ? it.score.toFixed(3) : "?"}</div>
        <pre class="pre">${escapeHtml(it.content || "")}</pre>
        <button class="ghost open-note">打开笔记</button>`;
      li.querySelector(".open-note").onclick = () => gotoNote(it.note_id);
      list.appendChild(li);
    }
  } catch (e) { list.innerHTML = `<li class="error">${escapeHtml(e.message)}</li>`; }
}

// ----------------------------- 对话（Codex 风格） -----------------------------
const TOOL_LABELS = {
  search_notes: "检索笔记库",
  get_note: "读取笔记原文",
  recall_memories: "回忆长期记忆",
  draft_plan: "起草计划",
  search_feeds: "搜索小红书",
  get_feed_detail: "读笔记详情",
  collect_note_by_link: "按链接采集",
  save_note_by_link: "收藏进知识库",
};

const chatState = {
  threadId: localStorage.getItem("xhs_thread") || null,
  threads: [],
  streaming: false,
  controller: null,
  pinned: true,
};

const chatStreamEl = () => $("#chat-stream");
const chatScrollEl = () => $("#chat-scroll");

function setPinned(v) {
  chatState.pinned = v;
  $("#btn-jump-bottom").classList.toggle("show", !v);
}

function scrollToBottom(force) {
  const sc = chatScrollEl();
  if (!sc) return;
  if (force || chatState.pinned) sc.scrollTop = sc.scrollHeight;
}

function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
}

function emptyState() {
  const div = document.createElement("div");
  div.className = "chat-empty";
  div.innerHTML = `<div class="ce-mark">✳</div>
    <h3>向你的小红书知识库提问</h3>
    <p>我会先检索你收藏的笔记，再给出带出处的回答；也能按链接采集站外笔记。</p>
    <div class="ce-chips"></div>`;
  const chips = div.querySelector(".ce-chips");
  const samples = [
    "我收藏的面经里，笔试都考了什么？",
    "帮我梳理 RAG 相关的笔记要点",
    "网易 Agent 岗面了哪些内容？",
    "给我一版一周面试准备计划",
  ];
  for (const q of samples) {
    const b = document.createElement("button");
    b.className = "ce-chip";
    b.textContent = q;
    b.onclick = () => {
      const input = $("#chat-text");
      input.value = q;
      autoGrow(input);
      input.focus();
    };
    chips.appendChild(b);
  }
  return div;
}

function stepEl(t) {
  const div = document.createElement("div");
  const failed = t.ok === false;
  div.className = `step${failed ? " bad" : ""}`;
  const label = TOOL_LABELS[t.name] || t.name;
  const bits = [];
  if (t.result_count != null) bits.push(`${t.result_count} 条命中`);
  if (t.top_score != null) bits.push(`最高分 ${Number(t.top_score).toFixed(2)}`);
  if (t.duration_ms != null) bits.push(`${(t.duration_ms / 1000).toFixed(1)}s`);
  if (failed) bits.push(t.error_code || "失败");
  if (t.ok === null || t.ok === undefined) bits.push("进行中…");
  const args = t.args && typeof t.args === "object"
    ? Object.entries(t.args).map(([k, v]) => `${k}=${String(v).slice(0, 32)}`).join(" ")
    : "";
  div.innerHTML = `<span>${failed ? "⚠️" : "🔧"}</span>
    <span class="st-name">${escapeHtml(label)}</span>
    <span class="st-meta">${escapeHtml(args)}</span>
    <span class="st-meta" style="margin-left:auto">${escapeHtml(bits.join(" · "))}</span>`;
  return div;
}

function refsEl(meta) {
  const raw = (meta && (meta.cited_note_ids || meta.source_note_ids)) || [];
  const ids = Array.from(new Set(raw)).filter(Boolean);
  if (!ids.length) return null;
  const box = document.createElement("div");
  box.className = "refs";
  box.innerHTML = `<span class="refs-label">出处</span>`;
  for (const id of ids) {
    const chip = document.createElement("button");
    chip.className = "ref-chip";
    chip.textContent = noteTitles[id] || `笔记 ${String(id).slice(0, 8)}`;
    chip.title = id;
    chip.onclick = () => gotoNote(id);
    box.appendChild(chip);
  }
  return box;
}

function footEl(msg) {
  const div = document.createElement("div");
  div.className = "turn-foot";
  const who = msg.role === "user" ? "你" : "助手";
  const when = msg.created_at ? fmtTime(msg.created_at) : "";
  div.innerHTML = `<span>${who}${when ? ` · ${when}` : ""}</span>`;
  const copy = document.createElement("button");
  copy.className = "ghost";
  copy.textContent = "复制";
  copy.onclick = () => {
    const text = msg.content || "";
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text)
        .then(() => toast("已复制"), () => toast("复制失败", true));
    }
  };
  div.appendChild(copy);
  return div;
}

function turnEl(msg) {
  const wrap = document.createElement("div");
  wrap.className = `turn ${msg.role}`;
  if (msg.role === "user") {
    const bubble = document.createElement("div");
    bubble.className = "bubble-user";
    bubble.textContent = msg.content || "";
    wrap.appendChild(bubble);
    wrap.appendChild(footEl(msg));
    return wrap;
  }
  const meta = msg.metadata || {};
  const traces = meta.tool_calls || [];
  if (traces.length) wrap.appendChild(stepsEl(traces, traces.reduce((s, t) => s + (t.duration_ms || 0), 0)));
  const body = document.createElement("div");
  body.className = "md";
  body.innerHTML = renderMarkdown(msg.content || "") || '<p class="muted">（本轮没有文字回答）</p>';
  wrap.appendChild(body);
  const refs = refsEl(meta);
  if (refs) wrap.appendChild(refs);
  wrap.appendChild(footEl(msg));
  return wrap;
}

function stepsEl(traces, totalMs) {
  const det = document.createElement("details");
  det.className = "steps";
  const failed = traces.filter((t) => t.ok === false).length;
  det.innerHTML = `<summary></summary>`;
  const body = document.createElement("div");
  for (const t of traces) body.appendChild(stepEl(t));
  det.appendChild(body);
  setStepsSummary(det, traces, totalMs, failed);
  return det;
}

function setStepsSummary(det, traces, totalMs, failed) {
  const label = det.dataset.live === "1"
    ? "正在工作"
    : `已完成 ${traces.length} 步`;
  const bits = [`${(totalMs / 1000).toFixed(1)}s`];
  if (failed) bits.push(`${failed} 步失败`);
  det.querySelector("summary").textContent = `${label} · ${bits.join(" · ")}`;
}

// 流式回答：边收边渲染，收尾时替换成落库后的正式消息。
function createLiveTurn() {
  const wrap = document.createElement("div");
  wrap.className = "turn assistant";
  const working = document.createElement("div");
  working.className = "working";
  working.innerHTML = `<span class="pulse"></span><span class="work-text">正在思考…</span>`;
  wrap.appendChild(working);
  chatStreamEl().appendChild(wrap);

  const body = document.createElement("div");
  body.className = "md";
  let det = null;
  let stepsBody = null;
  let draft = "";
  let status = "正在思考…";
  const traces = [];
  const pending = [];
  const started = Date.now();
  const workText = working.querySelector(".work-text");
  const timer = setInterval(() => {
    workText.textContent = `${status} · ${((Date.now() - started) / 1000).toFixed(1)}s`;
  }, 100);

  const stopTimer = () => clearInterval(timer);
  const totalMs = () => traces.reduce((s, t) => s + (t.duration_ms || 0), 0);

  const ensureSteps = () => {
    if (det) return;
    det = document.createElement("details");
    det.className = "steps";
    det.dataset.live = "1";
    det.open = true;
    det.innerHTML = `<summary></summary>`;
    stepsBody = document.createElement("div");
    det.appendChild(stepsBody);
    // 插到“进行中”指示器之后、正文之前，保证顺序是：进度 → 步骤 → 回答。
    wrap.insertBefore(det, body.parentNode ? body : null);
    updateSummary();
  };

  const updateSummary = () => {
    if (det) setStepsSummary(det, traces, totalMs(), traces.filter((t) => t.ok === false).length);
  };

  return {
    el: wrap,
    appendText(text) {
      draft += text;
      body.innerHTML = `${renderMarkdown(draft)}<span class="caret"></span>`;
      if (!body.parentNode) wrap.appendChild(body);
      scrollToBottom();
    },
    resetDraft() {
      // 模型该调工具却想直接作答，编排层要求它重来，此处丢弃这段草稿。
      draft = "";
      body.innerHTML = "";
      status = "重新规划中…";
      scrollToBottom();
    },
    toolStart(ev) {
      ensureSteps();
      status = `正在${TOOL_LABELS[ev.name] || ev.name}`;
      const row = stepEl({ name: ev.name, args: ev.args, ok: null });
      stepsBody.appendChild(row);
      pending.push(row);
      updateSummary();
      scrollToBottom();
    },
    toolEnd(ev) {
      ensureSteps();
      const row = pending.shift();
      const done = stepEl(ev);
      if (row && row.parentNode) row.replaceWith(done);
      else stepsBody.appendChild(done);
      traces.push(ev);
      status = "正在整理回答…";
      updateSummary();
      scrollToBottom();
    },
    finish(message) {
      stopTimer();
      working.remove();
      if (det) {
        det.dataset.live = "0";
        det.open = false;   // 回答出来之后步骤默认收起，不挡正文
        updateSummary();
      }
      const meta = (message && message.metadata) || { tool_calls: traces };
      const finalText = (message && message.content) || draft;
      body.innerHTML = renderMarkdown(finalText) || '<p class="muted">（本轮没有文字回答）</p>';
      if (!body.parentNode) wrap.appendChild(body);
      const refs = refsEl(meta);
      if (refs) wrap.appendChild(refs);
      wrap.appendChild(footEl(message || {
        role: "assistant", content: finalText, created_at: new Date().toISOString(),
      }));
      scrollToBottom();
    },
    fail(text) {
      stopTimer();
      working.remove();
      if (det) { det.dataset.live = "0"; det.open = false; updateSummary(); }
      body.innerHTML = `${renderMarkdown(draft)}<p class="error">${escapeHtml(text)}</p>`;
      if (!body.parentNode) wrap.appendChild(body);
      scrollToBottom();
    },
  };
}

async function loadThreads() {
  try {
    const d = await api("GET", "/threads?limit=100");
    chatState.threads = d.items || [];
  } catch (_) {
    chatState.threads = [];
  }
  renderThreadList();
}

function renderThreadList() {
  const box = $("#thread-list");
  if (!box) return;
  const kw = ($("#thread-filter").value || "").trim().toLowerCase();
  const items = chatState.threads.filter(
    (t) => !kw || String(t.title || "").toLowerCase().includes(kw)
  );
  $("#thread-count").textContent = `${chatState.threads.length} 个对话`;
  box.innerHTML = "";
  if (!items.length) {
    box.innerHTML = `<div class="muted" style="padding:10px;font-size:12px">${
      chatState.threads.length ? "没有匹配的对话" : "还没有对话，点上面新建一个。"}</div>`;
    return;
  }
  for (const t of items) {
    const div = document.createElement("div");
    div.className = `thread-item${t.id === chatState.threadId ? " active" : ""}`;
    const preview = t.last_message || "还没有消息";
    div.innerHTML = `<div class="t-title">${escapeHtml(t.title || "未命名对话")}</div>
      <div class="t-preview">${escapeHtml(preview)}</div>
      <div class="t-when">${relativeTime(t.updated_at)} · ${t.message_count || 0} 条</div>`;
    div.onclick = () => openThread(t.id);
    box.appendChild(div);
  }
}

async function openThread(id) {
  chatState.threadId = id;
  localStorage.setItem("xhs_thread", id);
  renderThreadList();
  const stream = chatStreamEl();
  stream.innerHTML = `<div class="muted" style="font-size:13px">正在载入会话…</div>`;
  try {
    const t = await api("GET", `/threads/${id}?message_limit=200`);
    renderThread(t);
  } catch (e) {
    stream.innerHTML = `<div class="error" style="font-size:13px">载入失败：${escapeHtml(e.message)}</div>`;
  }
}

function renderThread(t) {
  const stream = chatStreamEl();
  stream.innerHTML = "";
  $("#thread-title").value = t.title || "";
  $("#thread-meta").textContent =
    `${(t.messages || []).length} 条消息 · 更新于 ${relativeTime(t.updated_at)}`;
  if (!(t.messages || []).length) {
    stream.appendChild(emptyState());
  } else {
    for (const m of t.messages) stream.appendChild(turnEl(m));
  }
  setPinned(true);
  scrollToBottom(true);
}

async function newThread() {
  if (chatState.streaming) { toast("正在生成中，先等一下", true); return; }
  try {
    const t = await api("POST", "/threads", { title: "新对话" });
    await loadThreads();
    await openThread(t.id);
    $("#chat-text").focus();
  } catch (e) { toast(e.message, true); }
}

async function renameThread() {
  const id = chatState.threadId;
  if (!id) return;
  const input = $("#thread-title");
  const title = (input.value || "").trim();
  const current = (chatState.threads.find((t) => t.id === id) || {}).title || "";
  if (!title) { input.value = current; return; }
  if (title === current) return;
  try {
    await api("PATCH", `/threads/${id}`, { title });
    toast("标题已更新");
    loadThreads();
  } catch (e) { toast(e.message, true); }
}

async function deleteThread() {
  const id = chatState.threadId;
  if (!id) { toast("当前没有选中的对话", true); return; }
  if (!confirm("删除这个对话及其中全部消息？不可恢复。")) return;
  try {
    await api("DELETE", `/threads/${id}`);
    chatState.threadId = null;
    localStorage.removeItem("xhs_thread");
    await loadThreads();
    $("#thread-title").value = "";
    $("#thread-meta").textContent = "";
    const stream = chatStreamEl();
    stream.innerHTML = "";
    stream.appendChild(emptyState());
    toast("对话已删除");
  } catch (e) { toast(e.message, true); }
}

async function consolidateThread() {
  const id = chatState.threadId;
  if (!id) { toast("先选一个对话", true); return; }
  try {
    await api("POST", `/threads/${id}/consolidate`);
    toast("已提交提炼，稍后在记忆候选里审核");
  } catch (e) { toast(e.message, true); }
}

async function ensureThread() {
  if (chatState.threadId) {
    try {
      await api("GET", `/threads/${chatState.threadId}?message_limit=1`);
      return chatState.threadId;
    } catch (_) {
      chatState.threadId = null;
      localStorage.removeItem("xhs_thread");
    }
  }
  try {
    const t = await api("POST", "/threads", { title: "新对话" });
    chatState.threadId = t.id;
    localStorage.setItem("xhs_thread", t.id);
    await loadThreads();
    return t.id;
  } catch (e) {
    toast(e.message, true);
    return null;
  }
}

// 首个问题自动当标题：省掉每次都手动命名，也贴近 Codex 的会话命名习惯。
async function autoTitle(id, text) {
  const current = (chatState.threads.find((t) => t.id === id) || {}).title;
  if (current && current !== "新对话") return;
  const title = text.replace(/\s+/g, " ").trim().slice(0, 20);
  if (!title) return;
  try {
    await api("PATCH", `/threads/${id}`, { title });
    $("#thread-title").value = title;
  } catch (_) { /* 命名失败不影响对话 */ }
}

function setStreaming(on) {
  chatState.streaming = on;
  const btn = $("#btn-send");
  const input = $("#chat-text");
  btn.classList.toggle("stop", on);
  btn.textContent = on ? "■" : "↑";
  btn.title = on ? "停止生成" : "发送（Enter）";
  input.placeholder = on
    ? "生成中…可以点右侧按钮停止"
    : "问点什么…例如：我收藏的面经里笔试都考什么？（Enter 发送，Shift+Enter 换行）";
  $("#composer-status").textContent = on ? "生成中…" : "";
}

async function sendChat() {
  const input = $("#chat-text");
  const text = (input.value || "").trim();
  if (!text || chatState.streaming) return;
  const id = await ensureThread();
  if (!id) return;

  input.value = "";
  autoGrow(input);
  setPinned(true);
  const empty = chatStreamEl().querySelector(".chat-empty");
  if (empty) empty.remove();

  chatStreamEl().appendChild(
    turnEl({ role: "user", content: text, created_at: new Date().toISOString() })
  );
  const live = createLiveTurn();
  scrollToBottom(true);
  setStreaming(true);
  autoTitle(id, text);
  chatState.controller = new AbortController();

  let finished = false;
  try {
    await streamRequest(`/threads/${id}/messages/stream`, { content: text }, (ev) => {
      if (ev.type === "delta") live.appendText(ev.text);
      else if (ev.type === "reset") live.resetDraft();
      else if (ev.type === "tool_start") live.toolStart(ev);
      else if (ev.type === "tool_end") live.toolEnd(ev);
      else if (ev.type === "done") { finished = true; live.finish(ev.message); }
      else if (ev.type === "error") { finished = true; live.fail(ev.message || "生成失败"); }
    }, chatState.controller.signal);
    if (!finished) live.fail("连接中断，回答可能不完整。");
  } catch (e) {
    if (e.name === "AbortError") live.fail("已停止生成（本轮不会写入历史）。");
    else live.fail(e.message || "请求失败");
  } finally {
    setStreaming(false);
    chatState.controller = null;
    scrollToBottom();
    // 会话列表要重排（updated_at 变了）并刷新预览。
    loadThreads().then(() => {
      const t = chatState.threads.find((x) => x.id === chatState.threadId);
      if (t) $("#thread-meta").textContent =
        `更新于 ${relativeTime(t.updated_at)}`;
    });
  }
}

// ----------------------------- 待办 -----------------------------
async function loadTasks() {
  const status = $("#task-filter").value || undefined;
  const url = status ? `/tasks?status=${status}` : "/tasks";
  const d = await api("GET", url);
  const list = $("#task-list");
  list.innerHTML = "";
  if (!d.items.length) { list.innerHTML = `<li class="muted">没有待办。</li>`; return; }
  for (const t of d.items) {
    const li = document.createElement("li");
    li.innerHTML = `<div class="note-title">${escapeHtml(t.title)}</div>
      <div class="task-meta">${fmtTime(t.scheduled_at)} · <span class="status-${t.status}">${t.status}</span></div>
      ${t.details ? `<div class="note-sub">${escapeHtml(t.details)}</div>` : ""}
      <div class="task-actions">
        ${t.status !== "completed" ? '<button class="ghost act-complete">完成</button>' : ""}
        ${t.status !== "cancelled" ? '<button class="ghost act-cancel">取消</button>' : ""}
      </div>`;
    const comp = li.querySelector(".act-complete");
    const canc = li.querySelector(".act-cancel");
    if (comp) comp.onclick = async () => { await api("POST", `/tasks/${t.id}/complete`); loadTasks(); };
    if (canc) canc.onclick = async () => { await api("DELETE", `/tasks/${t.id}`); loadTasks(); };
    list.appendChild(li);
  }
}

// ----------------------------- 事件绑定 -----------------------------
$("#btn-import").onclick = importBatch;
$("#btn-read-all").onclick = readAllImages;
$("#btn-refresh-notes").onclick = loadNotes;
$("#btn-new-folder").onclick = newFolder;
$("#btn-refresh-folders").onclick = loadFolders;
$("#btn-search").onclick = doSearch;
$("#search-box").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });

$("#btn-new-thread").onclick = newThread;
$("#btn-delete-thread").onclick = deleteThread;
$("#btn-consolidate").onclick = consolidateThread;
$("#thread-filter").addEventListener("input", renderThreadList);
$("#thread-title").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); e.target.blur(); }
});
$("#thread-title").addEventListener("blur", renameThread);

$("#btn-send").onclick = () => {
  if (chatState.streaming) { chatState.controller?.abort(); return; }
  sendChat();
};
$("#chat-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
$("#chat-text").addEventListener("input", (e) => autoGrow(e.target));

chatScrollEl().addEventListener("scroll", () => {
  const sc = chatScrollEl();
  const atBottom = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 80;
  setPinned(atBottom);
});
$("#btn-jump-bottom").onclick = () => { setPinned(true); scrollToBottom(true); };

// 主题：默认浅色，切换后记住选择；深色接近 Codex 终端的观感。
const applyTheme = (dark) => {
  document.body.classList.toggle("theme-dark", dark);
  localStorage.setItem("xhs_theme", dark ? "dark" : "light");
  $("#btn-toggle-theme").textContent = dark ? "浅色主题" : "深色主题";
};
$("#btn-toggle-theme").onclick = () => applyTheme(!document.body.classList.contains("theme-dark"));

$("#btn-refresh-tasks").onclick = loadTasks;
$("#task-filter").onchange = loadTasks;

// ----------------------------- 初始化 -----------------------------
async function init() {
  applyTheme(localStorage.getItem("xhs_theme") === "dark");
  try {
    const h = await api("GET", "/health");
    $("#health").textContent = h.status === "ok" ? "后端在线" : "后端异常";
    $("#health").style.color = "var(--ok)";
  } catch (_) {
    $("#health").textContent = "后端未连接";
    $("#health").style.color = "var(--danger)";
  }
  loadNotes();
  loadFolders();
  loadTasks();
  await loadThreads();
  if (chatState.threadId) {
    openThread(chatState.threadId);
  } else {
    chatStreamEl().appendChild(emptyState());
  }
}
init();
