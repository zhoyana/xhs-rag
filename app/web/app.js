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
// 抽成函数是为了让"导入完成 → 跳到收件箱"这类跳转复用同一段逻辑，
// 而不是各自去改一遍 class。
function switchView(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
}

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
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
    // 记住归属的 note_id：对话里的「出处」「原始命中」靠它定位到具体某一行。
    li.dataset.note = n.note_id;
    li.addEventListener("click", () => openNote(n.note_id, li));
    list.appendChild(li);
  }
}

// 从对话里的出处/原始命中跳到笔记详情：先切回笔记库，再定位并打开那一篇。
// 之前只有调用没有实现，点出处会在控制台报 ReferenceError 而界面毫无反应。
async function gotoNote(noteId) {
  if (!noteId) return;
  switchView("notes");
  await loadNotes();
  const li = Array.from($$("#note-list li")).find((el) => el.dataset.note === noteId);
  if (li) li.scrollIntoView({ block: "nearest" });
  await openNote(noteId, li);
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
    loadNotes();
    // 导入只是把内容存下来；真正决定知识库好不好用的是"这篇为什么被收藏、
    // 该放进哪个收藏夹"。所以导入一结束就把这几篇交给 AI 读一遍并给出建议，
    // 用户只需要在收件箱里确认——而不是自己去笔记详情里一项项点。
    const fresh = (r.items || [])
      .filter((it) => it.note_id && it.status !== "failed" && it.status !== "duplicate")
      .map((it) => it.note_id);
    $("#import-box").value = "";
    switchView("inbox");
    await loadInbox();
    if (fresh.length) {
      toast(`导入 ${r.succeeded} 条，AI 开始整理…`);
      enqueuePrepare(fresh);
    } else {
      toast(`导入成功 ${r.succeeded} 条`);
    }
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

// ----------------------------- 收件箱（导入后的整理队列） -----------------------------
// 每张卡片自己记住"AI 说了什么 + 我改成了什么"，重渲染时从这份状态出发，
// 避免反复请求后端也避免把用户的编辑丢失。
const inboxState = {
  cards: new Map(),     // note_id → card 状态
  queue: [],            // 待 prepare 的 note_id
  running: false,
  stop: false,
  total: 0,
};

const TZ = "Asia/Shanghai";

// 待办时间只给三个够用的选项：整理笔记时不该被"选个精确时间"卡住。
function scheduleIso(daysAhead) {
  const d = new Date();
  d.setDate(d.getDate() + daysAhead);
  const pad = (n) => String(n).padStart(2, "0");
  const offsetMin = -d.getTimezoneOffset();
  const sign = offsetMin >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMin);
  // 后端要求时间必须带时区，偏移量从本机取，手拼字符串比 toISOString 更好读。
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
    + `T20:00:00${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`;
}

function cardFromItem(item) {
  return {
    note_id: item.note_id,
    title: item.title || "(无标题)",
    created_at: item.created_at,
    excerpt: item.excerpt || "",
    analysis: null,
    folders: [],
    selected: new Set(),
    suggestionReasons: [],
    actions: [],          // {text, on}
    category: "",
    intent: "",
    proposal: { category: "", intent: "" },
    editing: false,
    createTasks: false,
    scheduleDays: 0,
    touched: false,       // 用户是否动过分类/意图/收藏夹，动过就不再被 AI 结果覆盖
    // 分析和建议都齐了的笔记，进页面就能直接渲染（prepare 命中缓存，不花模型钱）；
    // 缺任何一项都先停在 idle，由用户决定什么时候花这次调用。
    autoReady: !!(item.has_analysis && item.has_suggestion),
    state: "idle",        // idle | loading | ready | error
    blockers: [],
  };
}

async function loadInbox() {
  const box = $("#inbox-list");
  let d;
  try {
    d = await api("GET", "/inbox?limit=50");
  } catch (e) {
    box.innerHTML = `<p class="error">${escapeHtml(e.message)}</p>`;
    return;
  }
  inboxState.total = d.total || 0;
  renderInboxBadge();
  const keep = new Set();
  for (const item of d.items || []) {
    keep.add(item.note_id);
    if (!inboxState.cards.has(item.note_id)) inboxState.cards.set(item.note_id, cardFromItem(item));
  }
  // 已经被整理掉的卡片要清出本地状态，否则重新拉取后会挂在列表里。
  for (const id of Array.from(inboxState.cards.keys())) {
    if (!keep.has(id)) inboxState.cards.delete(id);
  }
  renderInbox();
  // 已有完整结果的卡片直接补齐显示，不占用用户的点击、也不触发模型调用。
  const cached = Array.from(inboxState.cards.values())
    .filter((c) => c.autoReady && c.state === "idle")
    .map((c) => c.note_id);
  if (cached.length) enqueuePrepare(cached);
}

function renderInboxBadge() {
  const badge = $("#inbox-badge");
  const n = inboxState.total;
  badge.textContent = n > 99 ? "99+" : String(n);
  badge.hidden = n === 0;
}

function renderInbox() {
  const box = $("#inbox-list");
  box.innerHTML = "";
  const cards = Array.from(inboxState.cards.values());
  // 按钮上直接写清"还剩几篇"和"会调模型"：分析是要花钱的，不该点下去才发现。
  const remaining = cards.filter((c) => c.state === "idle" || c.state === "error").length;
  const btn = $("#btn-prepare-all");
  btn.hidden = remaining === 0;
  btn.textContent = `让 AI 整理剩余 ${remaining} 篇`;
  btn.title = `会为这 ${remaining} 篇各调用一次模型做分析`;
  if (!cards.length) {
    box.innerHTML = `<div class="inbox-empty">
      <div class="ie-mark">✓</div>
      <h3>收件箱空了</h3>
      <p>没有待整理的笔记。新导入的笔记会自动出现在这里，由 AI 先读一遍并给出分类、意图和收藏夹建议。</p>
    </div>`;
    return;
  }
  for (const card of cards) box.appendChild(inboxCardEl(card));
}

function inboxCardEl(card) {
  const el = document.createElement("article");
  el.className = "icard";
  el.dataset.note = card.note_id;

  const head = document.createElement("header");
  head.className = "icard-head";
  head.innerHTML = `<div class="icard-title" title="${escapeHtml(card.note_id)}">${escapeHtml(card.title)}</div>
    <div class="icard-sub">${fmtTime(card.created_at)}</div>`;
  el.appendChild(head);

  if (card.excerpt) {
    const ex = document.createElement("p");
    ex.className = "icard-excerpt";
    ex.textContent = card.excerpt;
    el.appendChild(ex);
  }

  el.appendChild(inboxAiEl(card));

  const actions = document.createElement("div");
  actions.className = "icard-actions";
  el.appendChild(actions);
  renderCardActions(actions, card);
  return el;
}

function inboxAiEl(card) {
  const box = document.createElement("div");
  box.className = "icard-ai";

  if (card.state === "loading") {
    box.classList.add("busy");
    box.innerHTML = `<span class="pulse"></span><span>AI 正在读这篇笔记，推断你的收藏意图…</span>`;
    return box;
  }
  if (card.state === "idle") {
    box.innerHTML = `<div class="icard-idle">
      <span>这篇还没整理过。</span>
      <button class="ghost act-prepare">让 AI 看看</button>
    </div>`;
    box.querySelector(".act-prepare").onclick = () => enqueuePrepare([card.note_id]);
    return box;
  }
  if (card.state === "error") {
    box.innerHTML = `<div class="icard-blocked">
      <b>没能整理：</b>${escapeHtml(card.blockers.join("；") || "模型暂时不可用")}
      <button class="ghost act-prepare">重试</button>
    </div>`;
    box.querySelector(".act-prepare").onclick = () => enqueuePrepare([card.note_id], true);
    return box;
  }

  // ready：展示 AI 推断 + 可编辑项。收藏夹芯片始终可点，改归档不必先进入编辑态。
  const a = card.analysis || {};
  const conf = a.intent_confidence;
  const confText = typeof conf === "number" ? `（置信度 ${conf.toFixed(2)}）` : "";

  const catRow = document.createElement("div");
  catRow.className = "ikv";
  catRow.innerHTML = `<span class="ik">分类</span>`;
  if (card.editing) {
    const input = document.createElement("input");
    input.className = "iv-input";
    input.value = card.category;
    input.oninput = () => { card.category = input.value; card.touched = true; };
    catRow.appendChild(input);
  } else {
    catRow.insertAdjacentHTML("beforeend",
      `<span class="iv">${escapeHtml(card.category || "未分类")}</span>`);
  }
  box.appendChild(catRow);

  const intRow = document.createElement("div");
  intRow.className = "ikv";
  intRow.innerHTML = `<span class="ik">收藏意图</span>`;
  if (card.editing) {
    const ta = document.createElement("textarea");
    ta.className = "iv-input";
    ta.rows = 2;
    ta.value = card.intent;
    ta.oninput = () => { card.intent = ta.value; card.touched = true; };
    intRow.appendChild(ta);
  } else {
    intRow.insertAdjacentHTML("beforeend",
      `<span class="iv">${escapeHtml(card.intent || "未推断")}<span class="muted">${escapeHtml(confText)}</span></span>`);
  }
  box.appendChild(intRow);

  const foldRow = document.createElement("div");
  foldRow.className = "ikv";
  foldRow.innerHTML = `<span class="ik">归档到</span>`;
  const chips = document.createElement("span");
  chips.className = "ifolders";
  if (!card.folders.length) {
    chips.innerHTML = `<span class="muted">还没有收藏夹，先去「收藏夹」建一个</span>`;
  }
  for (const f of card.folders) {
    const on = card.selected.has(f.id);
    const chip = document.createElement("button");
    chip.className = `fchip${on ? " on" : ""}`;
    chip.textContent = `${on ? "✓ " : ""}${f.name}`;
    chip.title = f.description || "";
    chip.onclick = () => {
      // 最多 3 个：和后端 folder_ids 的上限保持一致，先在这里挡住比让请求 422 好。
      if (card.selected.has(f.id)) card.selected.delete(f.id);
      else if (card.selected.size < 3) card.selected.add(f.id);
      else { toast("最多只能放进 3 个收藏夹", true); return; }
      card.touched = true;
      renderInbox();
    };
    chips.appendChild(chip);
  }
  foldRow.appendChild(chips);
  box.appendChild(foldRow);

  if (card.suggestionReasons.length) {
    const det = document.createElement("details");
    det.className = "ireasons";
    det.innerHTML = `<summary>AI 为什么这么放</summary><ul>${
      card.suggestionReasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`;
    box.appendChild(det);
  }

  if (card.actions.length) {
    const det = document.createElement("details");
    det.className = "iactions";
    det.open = true;
    det.innerHTML = `<summary>建议行动（${card.actions.filter((x) => x.on).length}/${card.actions.length}）</summary>`;
    const ul = document.createElement("div");
    for (let i = 0; i < card.actions.length; i += 1) {
      const act = card.actions[i];
      const row = document.createElement("label");
      row.className = "iact";
      row.innerHTML = `<input type="checkbox" ${act.on ? "checked" : ""}>`;
      const span = document.createElement("span");
      span.textContent = act.text;
      row.appendChild(span);
      row.querySelector("input").onchange = (e) => { act.on = e.target.checked; renderInbox(); };
      ul.appendChild(row);
    }
    det.appendChild(ul);
    box.appendChild(det);
  }

  if (card.blockers.length) {
    box.insertAdjacentHTML("beforeend",
      `<div class="icard-blocked">${escapeHtml(card.blockers.join("；"))}</div>`);
  }
  return box;
}

function renderCardActions(bar, card) {
  bar.innerHTML = "";
  if (card.state === "loading") {
    bar.innerHTML = `<span class="muted">AI 正在处理…</span>`;
    return;
  }
  if (card.state === "idle" || card.state === "error") {
    bar.innerHTML = `<span class="muted">整理后才能确认归档。</span>`;
    return;
  }

  if (card.editing) {
    const taskWrap = document.createElement("label");
    taskWrap.className = "itask";
    taskWrap.innerHTML = `<input type="checkbox" ${card.createTasks ? "checked" : ""}>
      <span>顺手把勾选的行动建成待办</span>`;
    taskWrap.querySelector("input").onchange = (e) => {
      card.createTasks = e.target.checked;
      renderInbox();
    };
    bar.appendChild(taskWrap);
    if (card.createTasks) {
      const sel = document.createElement("select");
      sel.className = "isel";
      for (const [i, label] of [[0, "今晚 20:00"], [1, "明晚 20:00"], [2, "后天 20:00"]]) {
        sel.insertAdjacentHTML("beforeend", `<option value="${i}">${label}</option>`);
      }
      sel.value = String(card.scheduleDays ?? 0);
      sel.onchange = (e) => { card.scheduleDays = Number(e.target.value); };
      bar.appendChild(sel);
    }
  }

  const save = document.createElement("button");
  save.className = "primary";
  save.textContent = card.editing ? "保存并归档" : "确认并归档";
  // 没选收藏夹就没法归档，先禁用；真要"什么都不做"用右边的忽略。
  save.disabled = card.selected.size === 0;
  save.onclick = () => triageCard(card);
  bar.appendChild(save);

  const edit = document.createElement("button");
  edit.className = "ghost";
  edit.textContent = card.editing ? "取消修改" : "改一下";
  edit.onclick = () => {
    if (card.editing) {
      // 放弃编辑：分类和意图回到 AI 原话，其余选择保留。
      card.category = card.proposal.category;
      card.intent = card.proposal.intent;
    }
    card.editing = !card.editing;
    renderInbox();
  };
  bar.appendChild(edit);

  const dismiss = document.createElement("button");
  dismiss.className = "ghost danger";
  dismiss.textContent = "忽略";
  dismiss.title = "这篇不需要归档，直接移出收件箱";
  dismiss.onclick = async () => {
    if (!confirm("忽略这篇笔记？它会移出收件箱，不再提示整理。")) return;
    await sendTriage(card, "dismissed");
  };
  bar.appendChild(dismiss);
}

function setCardReady(card, data) {
  card.title = data.title || card.title;
  card.created_at = data.created_at || card.created_at;
  card.excerpt = data.excerpt || card.excerpt;
  card.folders = data.folders || [];
  card.blockers = data.blockers || [];

  const a = data.analysis || null;
  if (!a) {
    card.state = "error";
    if (!card.blockers.length) card.blockers = ["模型没能给出分析结果"];
    return;
  }
  card.analysis = a;
  card.proposal = {
    category: a.category || "其他",
    intent: a.inferred_intent || "",
  };
  // 用户已经改过就别覆盖，只有第一次填充时才用 AI 的原话。
  if (!card.touched) {
    card.category = card.proposal.category;
    card.intent = card.proposal.intent;
  }

  const sug = data.suggestion || null;
  const suggested = (sug && Array.isArray(sug.suggestions)) ? sug.suggestions : [];
  const names = new Map(card.folders.map((f) => [f.id, f.name]));
  card.suggestionReasons = suggested
    .filter((s) => names.has(s.folder_id))
    .map((s) => `放进「${names.get(s.folder_id)}」：${s.reason}`);
  if (!card.touched) {
    card.selected = new Set(suggested.map((s) => s.folder_id).filter((id) => names.has(id)));
  }
  if (!card.selected.size && card.folders.length) card.selected.add(card.folders[0].id);

  if (!card.actions.length) {
    card.actions = (a.suggested_actions || []).map((text) => ({ text, on: false }));
  }
  card.state = "ready";
}

async function prepareCard(noteId, force = false) {
  const card = inboxState.cards.get(noteId);
  if (!card) return;
  card.state = "loading";
  renderInbox();
  try {
    const data = await api("POST", `/notes/${noteId}/prepare${force ? "?force=true" : ""}`);
    setCardReady(card, data);
  } catch (e) {
    card.state = "error";
    card.blockers = [e.message];
  }
  renderInbox();
}

async function sendTriage(card, decision) {
  const payload = {
    decision,
    folder_ids: decision === "dismissed" ? [] : Array.from(card.selected),
    action_indexes: card.actions.map((a, i) => (a.on ? i : -1)).filter((i) => i >= 0),
    create_tasks: decision !== "dismissed" && !!card.createTasks,
    timezone: TZ,
  };
  if (payload.create_tasks) payload.scheduled_at = scheduleIso(card.scheduleDays || 0);
  if (decision === "edited") {
    payload.category = card.category.trim();
    payload.intent = card.intent.trim();
  }
  try {
    const r = await api("POST", `/notes/${card.note_id}/triage`, payload);
    if (decision === "dismissed") {
      toast("已忽略，移出收件箱");
    } else {
      const where = (r.folders || []).map((f) => f.name).join("、");
      const n = (r.tasks || []).length;
      toast(`已归档到「${where || "所选收藏夹"}」${n ? `，并建了 ${n} 条待办` : ""}`);
    }
    inboxState.cards.delete(card.note_id);
    inboxState.total = Math.max(0, inboxState.total - 1);
    renderInboxBadge();
    renderInbox();
    loadFolders();
    loadTasks();
    loadNotes();
  } catch (e) {
    toast(e.message, true);
  }
}

async function triageCard(card) {
  if (!card.selected.size) { toast("请至少选择一个收藏夹", true); return; }
  const edited = card.category.trim() !== card.proposal.category
    || card.intent.trim() !== card.proposal.intent;
  if (edited && (!card.category.trim() || !card.intent.trim())) {
    toast("分类和收藏意图都不能为空", true);
    return;
  }
  await sendTriage(card, edited ? "edited" : "confirmed");
}

function enqueuePrepare(noteIds, force = false) {
  for (const id of noteIds) {
    const card = inboxState.cards.get(id);
    if (force && card) card.touched = false;
    if (!inboxState.queue.includes(id)) inboxState.queue.push(id);
  }
  runPrepareQueue();
}

function setInboxBusy(running) {
  $("#btn-prepare-all").disabled = running;
  $("#btn-stop-prepare").hidden = !running;
}

function updateInboxProgress() {
  const done = inboxState.queue.length;
  $("#inbox-progress").textContent = inboxState.running
    ? `正在整理，还剩 ${done} 篇…`
    : "";
}

async function runPrepareQueue() {
  if (inboxState.running) return;
  inboxState.running = true;
  inboxState.stop = false;
  setInboxBusy(true);
  while (inboxState.queue.length && !inboxState.stop) {
    updateInboxProgress();
    // 串行而不是并发：模型侧有速率限制，而且用户在界面上也是从上往下读的。
    await prepareCard(inboxState.queue.shift());
  }
  inboxState.running = false;
  setInboxBusy(false);
  updateInboxProgress();
  if (inboxState.stop) toast("已停止分析");
}

function prepareAllRemaining() {
  const pending = Array.from(inboxState.cards.values())
    .filter((c) => c.state === "idle" || c.state === "error")
    .map((c) => c.note_id);
  if (!pending.length) { toast("没有需要分析的笔记"); return; }
  toast(`开始整理 ${pending.length} 篇…`);
  enqueuePrepare(pending, true);
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

// 「原始命中」：原来单独占一个 tab 的检索视图并进对话后，能力落在这里。
// 用的是这一轮真正喂给模型的片段（落库时一并存的），不是重新检索一遍的结果。
function hitsEl(meta) {
  const hits = (meta && meta.retrieved_hits) || [];
  if (!hits.length) return null;
  const det = document.createElement("details");
  det.className = "hits";
  det.innerHTML = `<summary>原始命中 ${hits.length} 条</summary>`;
  const box = document.createElement("div");
  box.className = "hits-body";
  for (const h of hits) {
    const row = document.createElement("div");
    row.className = "hit";
    const score = typeof h.score === "number" ? h.score.toFixed(3) : "?";
    row.innerHTML = `<div class="hit-head">
      <button class="hit-title"></button>
      <span class="hit-score">相关度 ${escapeHtml(score)}</span>
    </div>
    <div class="hit-snippet"></div>`;
    const title = row.querySelector(".hit-title");
    title.textContent = h.title || `笔记 ${String(h.note_id).slice(0, 8)}`;
    title.title = h.note_id;
    title.onclick = () => gotoNote(h.note_id);
    row.querySelector(".hit-snippet").textContent = h.snippet || "（无正文片段）";
    box.appendChild(row);
  }
  det.appendChild(box);
  return det;
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
  const hits = hitsEl(meta);
  if (hits) wrap.appendChild(hits);
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
      const hits = hitsEl(meta);
      if (hits) wrap.appendChild(hits);
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
    // 三个动作语义不同，不能合成一个：
    //   完成 = 做完了；取消 = 我不做了（留痕，历史可追溯）；删除 = 这条记录本身就是错的。
    // 之前只有"完成/取消"，误建的任务（比如调试刷出来的空标题）只能一直挂在列表里。
    li.innerHTML = `<div class="note-title">${escapeHtml(t.title || "(无标题)")}</div>
      <div class="task-meta">${fmtTime(t.scheduled_at)} · <span class="status-${t.status}">${t.status}</span></div>
      ${t.details ? `<div class="note-sub">${escapeHtml(t.details)}</div>` : ""}
      <div class="task-actions">
        ${t.status === "pending" ? '<button class="ghost act-complete">完成</button>' : ""}
        ${t.status === "pending" ? '<button class="ghost act-cancel">取消</button>' : ""}
        <button class="ghost danger act-delete">删除</button>
      </div>`;
    const comp = li.querySelector(".act-complete");
    const canc = li.querySelector(".act-cancel");
    const del = li.querySelector(".act-delete");
    if (comp) comp.onclick = async () => {
      try { await api("POST", `/tasks/${t.id}/complete`); loadTasks(); }
      catch (e) { toast(e.message, true); }
    };
    if (canc) canc.onclick = async () => {
      if (!confirm("取消这条待办？记录会保留在「已取消」里，之后仍可彻底删除。")) return;
      try { await api("DELETE", `/tasks/${t.id}`); loadTasks(); }
      catch (e) { toast(e.message, true); }
    };
    del.onclick = async () => {
      if (!confirm("彻底删除这条待办？会从数据库中移除，无法恢复。")) return;
      try {
        await api("DELETE", `/tasks/${t.id}?hard=true`);
        toast("已删除");
        loadTasks();
      } catch (e) { toast(e.message, true); }
    };
    list.appendChild(li);
  }
}

// ----------------------------- 事件绑定 -----------------------------
$("#btn-import").onclick = importBatch;
$("#btn-read-all").onclick = readAllImages;
$("#btn-refresh-notes").onclick = loadNotes;
$("#btn-new-folder").onclick = newFolder;
$("#btn-refresh-folders").onclick = loadFolders;
$("#btn-refresh-inbox").onclick = loadInbox;
$("#btn-prepare-all").onclick = prepareAllRemaining;
$("#btn-stop-prepare").onclick = () => { inboxState.stop = true; };

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
  // 有东西要整理就先落在收件箱（那是唯一需要用户动手的地方）；
  // 空的就直接进笔记库，别让用户打开页面先看到一块"什么都没"。
  await loadInbox();
  switchView(inboxState.total > 0 ? "inbox" : "notes");
  await loadThreads();
  if (chatState.threadId) {
    openThread(chatState.threadId);
  } else {
    chatStreamEl().appendChild(emptyState());
  }
}
init();
