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
  // 日历按需拉取：用户可能整场都不打开它，没必要每次进页面都多打两枪。
  if (name === "calendar" && !calendarState.loaded) loadCalendar();
  if (name === "memory" && !memoryState.loaded) loadMemories();
}

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

// ----------------------------- 批量选择 -----------------------------
// 笔记和待办各一份选中集合，都存字符串（DOM 的 dataset 拿出来就是字符串，
// 存成数字会到处要做类型转换，漏一处就变成"勾了却删不掉"）。
// 用 Set 而不是数组：勾选/取消是 O(1)，"已选几条"也不用每次遍历 DOM 去数。
const selection = { notes: new Set(), tasks: new Set() };

const BULK = {
  notes: { bar: "#notes-bulk-count", del: "#btn-notes-delete", all: "#notes-pick-all",
           list: "#note-list", unit: "篇" },
  tasks: { bar: "#tasks-bulk-count", del: "#btn-tasks-delete", all: "#tasks-pick-all",
           list: "#task-list", unit: "条" },
};

function updateBulkBar(kind) {
  const conf = BULK[kind];
  const picked = selection[kind];
  const total = $$(`${conf.list} li.pickable`).length;
  $(conf.bar).textContent = `已选 ${picked.size} ${conf.unit}`;
  $(conf.del).disabled = picked.size === 0;
  const all = $(conf.all);
  // 三态全选框：部分选中时显示 indeterminate，否则用户看不出"还有没勾上的行"。
  all.checked = total > 0 && picked.size === total;
  all.indeterminate = picked.size > 0 && picked.size < total;
}

// 行首勾选框。点它只切换选中——必须挡掉冒泡，否则点勾选框会顺带触发
// "打开这篇笔记详情"，勾一次跳一次，根本没法连续多选。
function pickEl(kind, id) {
  const label = document.createElement("label");
  label.className = "pick";
  label.title = "选中以批量删除";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = selection[kind].has(id);
  label.appendChild(box);
  label.addEventListener("click", (e) => e.stopPropagation());
  box.addEventListener("change", () => {
    if (box.checked) selection[kind].add(id);
    else selection[kind].delete(id);
    const li = label.closest("li");
    if (li) li.classList.toggle("picked", box.checked);
    updateBulkBar(kind);
  });
  return label;
}

// 列表重新加载后按当前数据裁剪选中集合：已经删掉的 ID 不能继续留在计数里，
// 否则会出现"已选 3 篇"、列表却只剩 2 篇，点删除还要吃一个 404。
function pruneSelection(kind, ids) {
  const alive = new Set(ids.map(String));
  for (const id of Array.from(selection[kind])) {
    if (!alive.has(String(id))) selection[kind].delete(id);
  }
  updateBulkBar(kind);
}

function toggleAll(kind, on) {
  const conf = BULK[kind];
  selection[kind].clear();
  for (const li of $$(`${conf.list} li.pickable`)) {
    const id = kind === "notes" ? li.dataset.note : li.dataset.task;
    if (on && id != null) selection[kind].add(String(id));
    li.classList.toggle("picked", on);
    const box = li.querySelector(".pick input");
    if (box) box.checked = on;
  }
  updateBulkBar(kind);
}

// ----------------------------- 笔记库 -----------------------------
let currentNoteId = null;
const noteTitles = {};   // note_id → 标题，用于把对话里的引用 ID 显示成人看得懂的名字

async function loadNotes() {
  const data = await api("GET", "/notes?limit=100");
  const list = $("#note-list");
  list.innerHTML = "";
  if (!data.items.length) {
    list.innerHTML = `<li class="muted">还没有笔记，先粘贴链接批量导入。</li>`;
    updateBulkBar("notes");
    return;
  }
  for (const n of data.items) {
    noteTitles[n.note_id] = n.title || "";
    const li = document.createElement("li");
    li.className = "pickable";
    const hasImg = (n.image_text || "").trim() ? '<span class="badge img">图</span>' : "";
    // 内容包一层 .note-main：li 现在是 flex 容器（勾选框 + 正文两列），
    // 不包的话标题和那行小字会被当成两个并列的 flex 子项，横向排开。
    li.innerHTML = `<div class="note-main">
      <div class="note-title">${escapeHtml(n.title || "(无标题)")}</div>
      <div class="note-sub">${fmtTime(n.created_at)} ${hasImg}</div>
    </div>`;
    // 记住归属的 note_id：对话里的「出处」「原始命中」靠它定位到具体某一行。
    li.dataset.note = n.note_id;
    li.prepend(pickEl("notes", n.note_id));
    if (selection.notes.has(n.note_id)) li.classList.add("picked");
    li.addEventListener("click", () => openNote(n.note_id, li));
    list.appendChild(li);
  }
  pruneSelection("notes", data.items.map((n) => n.note_id));
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
      <button id="act-delete" class="ghost danger">删除这篇</button>
    </div>`;

  html += `<div class="section-h">正文</div><pre class="pre">${escapeHtml(note.content || "(无)")}</pre>`;
  if ((note.image_text || "").trim())
    html += `<div class="section-h">图片文字</div><pre class="pre">${escapeHtml(note.image_text)}</pre>`;

  // 异步补充：分析 / 图片 / 收藏夹 / 评论
  detail.innerHTML = html;
  $("#act-read").onclick = () => readImages(noteId);
  $("#act-download").onclick = () => downloadAssets(noteId);
  $("#act-analyze").onclick = () => analyzeNote(noteId);
  $("#act-delete").onclick = () => deleteNote(note);

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
    const all = d.items || [];
    if (!all.length) return;
    const saved = all.filter((x) => x.file_url);
    let html = `<div class="section-h">图片（已保存 ${saved.length} / 共 ${all.length} 张）</div>`;
    for (const it of saved) html += `<img class="asset" src="${it.file_url}" alt="笔记图片" />`;
    const missing = all.filter((x) => !x.file_url);
    if (missing.length) {
      // 没存下来的图以前被直接过滤掉，页面上看不出「少了图」；这里如实说出张数与原因。
      const why = (missing.find((x) => x.error) || {}).error || "尚未下载";
      html += `<div class="note-sub">还有 ${missing.length} 张没有保存到本地：${escapeHtml(why)}</div>`;
    }
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

// 读图与下载都是长请求，点第二下只会撞上服务端的"已有任务进行中"（409）。
// 置灰按钮比让用户看一条 409 更直接。
function withBusyButton(selector, task) {
  const button = $(selector);
  if (button && button.disabled) return Promise.resolve();
  if (button) button.disabled = true;
  return Promise.resolve(task()).finally(() => { if (button) button.disabled = false; });
}

async function readImages(noteId) {
  const force = confirm("补读图片会调用视觉模型（消耗额度）。确定继续？");
  if (!force) return;
  await withBusyButton("#act-read", async () => {
    try {
      const r = await api("POST", `/notes/${noteId}/read-images?force=true`);
      // 接口对"没图可读""已有文字"都是 200，以前一律显示"补读完成"，看着像成功。
      if (r.status === "no_local_images") {
        toast(r.reason || "这篇还没有本地图片，补读无事可做。", true);
        return;
      }
      if (r.status === "skipped") {
        toast(r.reason || "已有图片文字，未重复读图。", true);
        return;
      }
      if (!r.read) {
        const why = (r.vision && r.vision.error) || "没有读出文字";
        toast(`补读失败：${why}`, true);
        return;
      }
      toast(`补读完成：读出 ${r.read} 张图片${r.status === "partial" ? "（部分图片失败）" : ""}`);
      loadNotes();
      openNote(noteId);
    } catch (e) { toast(e.message, true); }
  });
}

async function downloadAssets(noteId) {
  await withBusyButton("#act-download", async () => {
    try {
      const r = await api("POST", `/notes/${noteId}/assets/download`);
      if (r.failed) {
        // 失败张数与原因以前被吞掉，只剩"下载完成 0/0"，用户既存不到也不知道为什么。
        const why = ((r.items || []).find((x) => x.status === "failed" && x.error) || {}).error
          || "原因未知";
        toast(`下载失败 ${r.failed} 张（成功 ${r.downloaded || 0} 张）：${why}`, true);
      } else {
        const reused = r.reused ? `，其中 ${r.reused} 张复用刚读图时的字节` : "";
        toast(`下载完成：本地保存 ${r.downloaded || 0} 张，跳过已有 ${r.skipped || 0} 张${reused}`);
      }
      openNote(noteId);
    } catch (e) { toast(e.message, true); }
  });
}

async function analyzeNote(noteId) {
  try {
    await api("POST", `/notes/${noteId}/analyze?force=true`);
    toast("分析完成");
    openNote(noteId);
  } catch (e) { toast(e.message, true); }
}

// 采集服务没开是这个项目里"一整批全失败"最常见的原因，而它用户自己能修
// （去启动那个 exe）。所以单独给一句可操作的提示，而不是让人对着
// "失败 12 条"猜是链接的问题还是系统坏了。
const COLLECTOR_HINTS = {
  unavailable: "本地采集服务（xiaohongshu-mcp，端口 18060）没启动或未登录，"
    + "已退回网页解析；小红书网页拿不到笔记数据，所以整批都失败。启动采集服务后重试即可。",
  timeout: "本地采集服务 60 秒内没有返回，已退回网页解析。先确认它是不是僵死了，再重试。",
};

// 失败原因按"原因"归并而不是逐条列：一批里同样一句错误重复十几遍没有信息量。
function failureReasons(result) {
  const counted = new Map();
  for (const item of result.items || []) {
    if (item.status !== "failed") continue;
    const why = item.error || "原因未知";
    counted.set(why, (counted.get(why) || 0) + 1);
  }
  return [...counted.entries()].sort((a, b) => b[1] - a[1]);
}

async function importBatch() {
  const text = $("#import-box").value.trim();
  if (!text) { toast("请先粘贴链接", true); return; }
  $("#btn-import").disabled = true;
  try {
    const r = await api("POST", "/notes/import-batch", { share_text: text });
    const reasons = failureReasons(r);
    // 结果区把所有事实都摆出来：成功几条、失败几条并给出原因。
    // 只报"成功 0"，用户既不知道是链接的问题还是环境的问题。
    let summary = `成功 ${r.succeeded} · 失败 ${r.failed} · 重复 ${r.duplicates}`;
    if (r.failed && COLLECTOR_HINTS[r.collector_mcp]) {
      summary += `\n${COLLECTOR_HINTS[r.collector_mcp]}`;
    }
    for (const [why, count] of reasons) summary += `\n${count} 条失败：${why}`;
    $("#import-result").textContent = summary;
    // 有失败时不能再是一行灰色小字——那是"提示"的语气，会被当成无关紧要。
    $("#import-result").classList.toggle("error", Boolean(r.failed));
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
    } else if (r.failed) {
      // 这一支以前不问青红皂白地说"导入成功 0 条"，把整批失败说成了成功。
      const why = reasons.length ? `：${reasons[0][0]}` : "";
      toast(`导入失败 ${r.failed} 条${why}`, true);
    } else {
      toast(`没有新增：这 ${r.duplicates} 条都是已收藏的`);
    }
  } catch (e) { toast(e.message, true); }
  finally { $("#btn-import").disabled = false; }
}

// ----------------------------- 删除笔记 -----------------------------
// 删笔记是这个项目里唯一不可逆的操作，所以确认框必须把"顺带删掉了什么"讲清楚：
// 用户以为只是少了一篇收藏，实际上图片、AI 分析、收藏意图、收藏夹归属、检索索引
// 和派生出来的待办都会一起没。一句含糊的"确定删除吗"在这里是不够的。
const NOTE_DELETE_WARNING = "会一并删除它的图片、AI 分析、收藏意图、收藏夹归属和检索索引，"
  + "无法恢复；它派生出来的待办也会一起删掉。";

function clearNoteDetail() {
  currentNoteId = null;
  $("#note-detail").innerHTML = `<p class="muted">从左侧选择一篇笔记查看详情。</p>`;
}

// 删完要刷三处：笔记列表、收件箱（少一篇待整理）、日历（待办跟着没了）。
// 日历是懒加载的，标脏即可——用户没切过去看就不必现在多打一枪。
async function refreshAfterNoteDelete() {
  await loadNotes();
  loadInbox();
  calendarState.needsRefresh = true;
}

async function deleteNote(note) {
  const noteId = note.note_id;
  const title = String(note.title || "").slice(0, 24);
  if (!confirm(`彻底删除${title ? `《${title}》` : "这篇笔记"}？\n\n${NOTE_DELETE_WARNING}`)) return;
  try {
    const result = await api("DELETE", `/notes/${noteId}`);
    const stuck = (result.failed_files && result.failed_files.length) || 0;
    toast(stuck
      ? `已删除${result.tasks ? `（含 ${result.tasks} 条待办）` : ""}，但有 ${stuck} 个图片文件没能删除`
      : `已删除${result.tasks ? `（含 ${result.tasks} 条待办）` : ""}`, stuck > 0);
    selection.notes.delete(noteId);
    clearNoteDetail();
    await refreshAfterNoteDelete();
  } catch (e) {
    toast(e.message, true);
    // 失败也要刷新：后端是"先删库（事务提交）→ 再删图片"，图片那步炸了同样会返回 500，
    // 但数据库里那几行已经删掉了。不刷新的话界面还留着已经不存在的行，用户会反复重试。
    await refreshAfterNoteDelete();
  }
}

async function deletePickedNotes() {
  const ids = Array.from(selection.notes);
  if (!ids.length) return;
  if (!confirm(`彻底删除选中的 ${ids.length} 篇笔记？\n\n${NOTE_DELETE_WARNING}`)) return;
  const button = $("#btn-notes-delete");
  button.disabled = true;
  try {
    const result = await api("POST", "/notes/bulk-delete", { note_ids: ids });
    let message = `已删除 ${result.deleted_count} 篇笔记`;
    if (result.tasks_removed) message += `（含 ${result.tasks_removed} 条待办）`;
    // 部分失败用红字提示：整批"看起来成功"而实际少删了，比直接报错更危险。
    if (result.failed.length) message += `；${result.failed.length} 篇没删掉`;
    // 库删掉了、磁盘上的图片却没删掉时也要说：这类残留不会自己消失，
    // 不说的话用户以为删干净了，实际 media 目录在慢慢攒孤儿文件。
    const stuck = result.deleted.reduce(
      (total, item) => total + ((item.failed_files && item.failed_files.length) || 0), 0);
    if (stuck) message += `；${stuck} 个图片文件没能删除`;
    toast(message, result.failed.length > 0 || stuck > 0);
    selection.notes.clear();
    if (ids.includes(currentNoteId)) clearNoteDetail();
    await refreshAfterNoteDelete();
  } catch (e) {
    toast(e.message, true);
    updateBulkBar("notes");   // 失败时把按钮还原，别让它一直灰着
    await refreshAfterNoteDelete();   // 理由同单篇删除：500 时库里可能已经删掉了
  }
}

async function readAllImages() {
  if (!confirm("将遍历全部已入库笔记，为已下载图片的笔记补读图（消耗较多视觉模型额度）。确定？")) return;
  try {
    const r = await api("POST", "/notes/read-images", { all: true });
    // 逐类如实汇总：只报"成功 x / 共 n"会把"没有本地图"这种无事可做的也混成成功。
    const count = (s) => (r.items || []).filter((x) => x.status === s).length;
    const detail = [];
    if (count("no_local_images")) detail.push(`没有本地图片 ${count("no_local_images")} 篇`);
    if (count("skipped")) detail.push(`已有图片文字 ${count("skipped")} 篇`);
    if (count("vision_failed")) detail.push(`读图失败 ${count("vision_failed")} 篇`);
    toast(`批量补读：成功 ${r.succeeded} / 共 ${r.total}${detail.length ? "；" + detail.join("，") : ""}`,
      r.succeeded === 0);
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

// 把本机时间转成带时区偏移的 ISO 文本：后端拒收不带时区的时间，
// 而从 Date 的角度看，偏移量只有本机知道（手拼比 toISOString 的 Z 更好读）。
function localIso(d) {
  const pad = (n) => String(n).padStart(2, "0");
  const offsetMin = -d.getTimezoneOffset();
  const sign = offsetMin >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMin);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
    + `T${pad(d.getHours())}:${pad(d.getMinutes())}:00`
    + `${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`;
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
    category: "",
    intent: "",
    proposal: { category: "", intent: "" },
    editing: false,
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

  // 「这篇该干什么」不在这里派活：模型给的行动清单只是它的猜测，直接变成待办
  // 等于替用户决定日程。想安排就去对话里和 agent 商量，谈拢了再落成计划。
  const plan = document.createElement("button");
  plan.className = "ghost act-plan";
  plan.textContent = "让 agent 排计划";
  plan.title = "切到对话，由 agent 参考这篇笔记和你商量后再出草案；你点确认才会写入日历";
  plan.onclick = () => askAgentForPlan(card);
  bar.appendChild(plan);

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
  // 这次确认只管两件事：这篇为什么被收藏、放进哪个夹子。
  // 待办不在这里建——模型给的行动清单是它的推测，直接落成日程等于替用户做决定。
  const payload = {
    decision,
    folder_ids: decision === "dismissed" ? [] : Array.from(card.selected),
  };
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
      toast(`已归档到「${where || "所选收藏夹"}」`);
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

// 从收件箱进"排计划"：切到对话并把提问预填好，让用户补上自己的目标再发。
// 刻意不自动发送——排计划最关键的输入是用户的目标和可用时间，而不是这篇笔记本身。
async function askAgentForPlan(card) {
  switchView("chat");
  const id = await ensureThread();
  if (!id) return;
  const input = $("#chat-text");
  input.value = `我想围绕《${String(card.title || "").slice(0, 40)}》安排一下。`
    + "帮我制定一份计划：先问清我的目标和每天能投入多少时间，再出草案。";
  autoGrow(input);
  input.focus();
  toast("补上你的目标，再点发送");
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

// ------------------------------ 计划草案卡 ------------------------------
// agent 只能起草，落库必须由用户点确认。这张卡就是"同意"的落点：没有它之前，
// 草案只存在于聊天记录里，用户想点头也没地方点。
const planFmtDay = (iso) => {
  const d = new Date(iso);
  if (isNaN(d)) return iso || "";
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getMonth() + 1}/${d.getDate()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

function planEl(meta) {
  const id = meta && meta.plan_id;
  if (!id) return null;
  const box = document.createElement("div");
  box.className = "plan-card";
  box.innerHTML = `<div class="plan-wait"><span class="pulse"></span><span>正在载入计划草案…</span></div>`;
  fillPlanCard(box, id);
  return box;
}

async function fillPlanCard(box, planId) {
  try {
    renderPlanCard(box, await api("GET", `/plans/${planId}`));
  } catch (e) {
    // 草案被丢弃后，历史消息里的卡还在；这时给一句人话，别摆一个报错。
    if (String(e.message).includes("不存在")) {
      box.innerHTML = `<div class="plan-head"><span class="plan-badge dead">已丢弃</span>
        <span class="plan-hint">这份草案没有写入日历</span></div>`;
      return;
    }
    box.innerHTML = `<div class="error">计划载入失败：${escapeHtml(e.message)}</div>`;
  }
}

function renderPlanCard(box, plan) {
  const draft = plan.status === "draft";
  box.dataset.plan = plan.id;
  box.innerHTML = "";

  const head = document.createElement("div");
  head.className = "plan-head";
  head.innerHTML = `<span class="plan-badge">${draft ? "计划草案" : "已确认"}</span>
    <span class="plan-hint">${draft ? "确认后才会写入日历" : "已写入日历"}</span>`;
  box.appendChild(head);

  const title = document.createElement("div");
  title.className = "plan-title";
  title.textContent = plan.title || "（无标题计划）";
  box.appendChild(title);

  if (plan.goal) {
    const goal = document.createElement("div");
    goal.className = "plan-goal";
    goal.textContent = `目标：${plan.goal}`;
    box.appendChild(goal);
  }

  const list = document.createElement("ol");
  list.className = "plan-items";
  for (const item of plan.items || []) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="pi-when">${escapeHtml(planFmtDay(item.scheduled_at))}</span>
      <span class="pi-title"></span>`;
    li.querySelector(".pi-title").textContent = item.title || "";
    if (item.details) {
      const d = document.createElement("div");
      d.className = "pi-details";
      d.textContent = item.details;
      li.appendChild(d);
    }
    list.appendChild(li);
  }
  box.appendChild(list);

  // 已确认的计划不再给按钮：后续改动去日历里处理具体任务。
  if (!draft) return;

  const bar = document.createElement("div");
  bar.className = "plan-actions";

  const ok = document.createElement("button");
  ok.className = "primary";
  ok.textContent = "确认计划";
  ok.title = "确认后这些步骤会变成日历上的待办";
  ok.onclick = async () => {
    ok.disabled = true;
    await confirmPlanById(plan.id, box);
    ok.disabled = false;
  };
  bar.appendChild(ok);

  const edit = document.createElement("button");
  edit.className = "ghost";
  edit.textContent = "改一下";
  edit.title = "在对话里继续说你想怎么改，agent 会重新出草案";
  edit.onclick = () => {
    const input = $("#chat-text");
    // 模板里保留"计划/生成"字样：编排层据此继续授权 draft_plan 工具。
    input.value = `这份计划草案（#${plan.id}）我想调整一下，请重新生成计划草案：`;
    autoGrow(input);
    input.focus();
  };
  bar.appendChild(edit);

  const drop = document.createElement("button");
  drop.className = "ghost danger";
  drop.textContent = "丢弃";
  drop.title = "这份草案不写入日历";
  drop.onclick = async () => {
    if (!confirm("丢弃这份草案？它不会被写入日历。")) return;
    try {
      await api("DELETE", `/plans/${plan.id}`);
      box.innerHTML = `<div class="plan-head"><span class="plan-badge dead">已丢弃</span>
        <span class="plan-hint">这份草案没有写入日历</span></div>`;
      toast("草案已丢弃");
    } catch (e) { toast(e.message, true); }
  };
  bar.appendChild(drop);
  box.appendChild(bar);
}

// 确认 = 服务端一次 resume（确认 + 建任务 + 收集质量门禁），失败原因都摊开，不静默吞掉。
async function confirmPlanById(planId, box) {
  // 一次调用完成「确认 + 写进日历」。原来是连打 confirm 和 create-tasks 两个接口，
  // 中间断开就会留下"计划已确认、日历里却没有任务"的半成品；现在这两步是服务端
  // 同一个 resume，客户端只发一次。
  let result;
  try {
    result = await api("POST", `/plans/${planId}/commit`);
  } catch (e) {
    showPlanBlockers(box, e);
    return;
  }
  const n = (result.task_ids || []).length;
  toast(n ? `已写入日历：${n} 项` : "这份计划的任务之前已经建过");
  if (box) fillPlanCard(box, planId);
  loadCalendar();
  loadTasks();
}

function showPlanBlockers(box, e) {
  // 质量门禁的 409 把结构化的 blockers 放在 detail 里，api() 已把它序列化成文本。
  let detail = null;
  try { detail = JSON.parse(e.message); } catch (_) { /* 不是结构化错误 */ }
  const blockers = detail && Array.isArray(detail.blockers) ? detail.blockers : [];
  if (!blockers.length || !box) { toast(e.message, true); return; }
  const old = box.querySelector(".plan-blocked");
  if (old) old.remove();
  const div = document.createElement("div");
  div.className = "plan-blocked";
  div.innerHTML = `<b>还不能确认：</b><ul>${
    blockers.map((b) => `<li>${escapeHtml(b.message || "")}</li>`).join("")}</ul>
    <span class="muted">先改时间或删掉多余的步骤，再确认。</span>`;
  box.appendChild(div);
  toast("计划还有问题需要先改一下", true);
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
  const plan = planEl(meta);
  if (plan) wrap.appendChild(plan);
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
      const plan = planEl(meta);
      if (plan) wrap.appendChild(plan);
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
    // 提示必须指一个真实存在的去处：提炼是后台任务，稍后才会出现候选，
    // 所以引导到「记忆」页并让徽标负责提醒"有几条在等你审"。
    toast("已提交提炼。稍后到「记忆」页审核——确认过的才会进入回答");
    setTimeout(renderMemoryBadge, 1500);
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

// ----------------------------- 日历 -----------------------------
// 计划确认之后落到这里，按天排开。月历必须带区间取数：不带区间时后端"按时间升序
// + limit"会先把最早的历史任务取出来，任务一攒多当月反而是空的。
const calendarState = {
  view: localStorage.getItem("xhs_cal_view") || "month",
  cursor: new Date(),      // 正在看的月份（取该月任意一天）
  selected: null,          // "YYYY-MM-DD"
  tasks: [],               // 当月任务
  drafts: [],              // 未确认草案的步骤：{plan_id, plan_title, item}
  loaded: false,
};

const pad2 = (n) => String(n).padStart(2, "0");
const dayKey = (d) => `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
const todayKey = () => dayKey(new Date());
const keyOfIso = (iso) => {
  const d = new Date(iso);
  return isNaN(d) ? null : dayKey(d);
};
const shorten = (text, n) => {
  const s = String(text || "");
  return s.length > n ? `${s.slice(0, n)}…` : s;
};

async function loadCalendar() {
  const year = calendarState.cursor.getFullYear();
  const month = calendarState.cursor.getMonth();
  const start = new Date(year, month, 1);
  const end = new Date(year, month + 1, 1);
  // 区间按本机时区换算成绝对时刻（toISOString 带 Z，后端能解析），
  // 再由后端统一归一到 UTC 文本与库里的 scheduled_at 做比较。
  const qs = new URLSearchParams({
    limit: "500",
    date_from: start.toISOString(),
    date_to: end.toISOString(),
  });
  try {
    const [tasks, plans] = await Promise.all([
      api("GET", `/tasks?${qs}`),
      // 草案拉全量：它没有按月分页的必要（通常就一两份），跨月也能标出来。
      api("GET", "/plans?status=draft").catch(() => ({ items: [] })),
    ]);
    calendarState.tasks = tasks.items || [];
    calendarState.drafts = [];
    for (const plan of plans.items || []) {
      for (const item of plan.items || []) {
        calendarState.drafts.push({ plan_id: plan.id, plan_title: plan.title, item });
      }
    }
  } catch (e) {
    toast(e.message, true);
  }
  calendarState.loaded = true;
  if (!calendarState.selected) calendarState.selected = todayKey();
  renderCalendar();
}

function dayItems(key) {
  return {
    tasks: calendarState.tasks.filter((t) => keyOfIso(t.scheduled_at) === key),
    drafts: calendarState.drafts.filter((d) => keyOfIso(d.item.scheduled_at) === key),
  };
}

function renderCalendar() {
  const year = calendarState.cursor.getFullYear();
  const month = calendarState.cursor.getMonth();
  $("#cal-label").textContent = `${year} 年 ${month + 1} 月`;

  const grid = $("#cal-grid");
  grid.innerHTML = "";
  // 周一开头：getDay() 里周日是 0，先把它挪到队尾再算前导格数。
  const lead = (new Date(year, month, 1).getDay() + 6) % 7;
  const days = new Date(year, month + 1, 0).getDate();
  const cells = Math.ceil((lead + days) / 7) * 7;
  let pending = 0;

  for (let i = 0; i < cells; i += 1) {
    const date = new Date(year, month, 1 - lead + i);
    const key = dayKey(date);
    const { tasks, drafts } = dayItems(key);
    if (date.getMonth() === month) {
      pending += tasks.filter((t) => t.status === "pending").length;
    }

    const cell = document.createElement("button");
    cell.className = "cal-cell";
    if (date.getMonth() !== month) cell.classList.add("out");
    if (key === todayKey()) cell.classList.add("today");
    if (key === calendarState.selected) cell.classList.add("sel");

    const chips = [];
    for (const t of tasks.slice(0, 3)) {
      chips.push(`<span class="cc-chip ${escapeHtml(t.status)}">${escapeHtml(shorten(t.title, 9))}</span>`);
    }
    for (const g of drafts.slice(0, Math.max(0, 3 - chips.length))) {
      chips.push(`<span class="cc-chip draft">${escapeHtml(shorten(g.item.title, 9))}</span>`);
    }
    const hidden = tasks.length + drafts.length - chips.length;
    if (hidden > 0) chips.push(`<span class="cc-more">+${hidden}</span>`);
    cell.innerHTML = `<span class="cc-day">${date.getDate()}</span>${chips.join("")}`;
    cell.onclick = () => { calendarState.selected = key; renderCalendar(); };
    grid.appendChild(cell);
  }

  $("#cal-summary").textContent = `本月待办 ${pending} 项`
    + (calendarState.drafts.length ? ` · 未确认草案 ${calendarState.drafts.length} 步` : "");
  renderDayPanel();
}

function dayBtn(label, cls, onclick) {
  const b = document.createElement("button");
  b.className = cls;
  b.textContent = label;
  b.onclick = onclick;
  return b;
}

function renderDayPanel() {
  const key = calendarState.selected || todayKey();
  const date = new Date(`${key}T00:00:00`);
  const { tasks, drafts } = dayItems(key);
  $("#cal-day-title").textContent =
    `${date.getMonth() + 1} 月 ${date.getDate()} 日 · 周${"一二三四五六日"[(date.getDay() + 6) % 7]}`;
  $("#cal-day-meta").textContent = tasks.length || drafts.length
    ? `${tasks.length} 项待办${drafts.length ? ` · ${drafts.length} 步草案` : ""}`
    : "没有安排";

  const box = $("#cal-day-list");
  box.innerHTML = "";
  if (!tasks.length && !drafts.length) {
    box.innerHTML = `<div class="muted" style="font-size:13px">这天还空着。要安排点什么，就
      点右下角让 agent 参考你收藏的笔记来排。</div>`;
    return;
  }
  for (const t of tasks) box.appendChild(taskRowEl(t));
  for (const g of drafts) box.appendChild(draftRowEl(g));
}

function taskRowEl(t) {
  const row = document.createElement("div");
  row.className = `day-item ${escapeHtml(t.status)}`;
  const at = new Date(t.scheduled_at);
  const hhmm = isNaN(at) ? "" : `${pad2(at.getHours())}:${pad2(at.getMinutes())}`;
  row.innerHTML = `<div class="di-head"><span class="di-time">${hhmm}</span>
      <span class="di-title"></span></div>
    ${t.details ? '<div class="di-details"></div>' : ""}
    <div class="di-actions"></div>`;
  row.querySelector(".di-title").textContent = t.title || "(无标题)";
  if (t.details) row.querySelector(".di-details").textContent = t.details;

  const bar = row.querySelector(".di-actions");
  if (t.status === "pending") {
    bar.appendChild(dayBtn("完成", "primary", async () => {
      try { await api("POST", `/tasks/${t.id}/complete`); await refreshAfterTaskChange(); }
      catch (e) { toast(e.message, true); }
    }));
    // 日历上最常见的动作就是"今天做不完，顺延一天"，不该逼用户去改具体时间。
    bar.appendChild(dayBtn("挪到明天", "ghost", async () => {
      const next = new Date(t.scheduled_at);
      next.setDate(next.getDate() + 1);
      try {
        await api("PUT", `/tasks/${t.id}`, {
          scheduled_at: localIso(next), timezone: t.timezone || "Asia/Shanghai",
        });
        toast("已挪到明天");
        await refreshAfterTaskChange();
      } catch (e) { toast(e.message, true); }
    }));
    bar.appendChild(dayBtn("取消", "ghost", async () => {
      if (!confirm("取消这条待办？记录会留在「已取消」里，仍可追溯。")) return;
      try { await api("DELETE", `/tasks/${t.id}`); await refreshAfterTaskChange(); }
      catch (e) { toast(e.message, true); }
    }));
  }
  bar.appendChild(dayBtn("删除", "ghost danger", async () => {
    if (!confirm("彻底删除这条待办？会从数据库移除，不可恢复。")) return;
    try {
      await api("DELETE", `/tasks/${t.id}?hard=true`);
      toast("已删除");
      await refreshAfterTaskChange();
    } catch (e) { toast(e.message, true); }
  }));
  return row;
}

// 草案步骤只是"如果确认，它会落在这里"，所以用虚影样式区分，并直接给出确认入口。
function draftRowEl(g) {
  const row = document.createElement("div");
  row.className = "day-item draft";
  const at = new Date(g.item.scheduled_at);
  const hhmm = isNaN(at) ? "" : `${pad2(at.getHours())}:${pad2(at.getMinutes())}`;
  row.innerHTML = `<div class="di-head"><span class="di-time">${hhmm}</span>
      <span class="di-title"></span></div>
    <div class="di-note">草案 · ${escapeHtml(g.plan_title || "未命名计划")}</div>
    <div class="di-actions"></div>`;
  row.querySelector(".di-title").textContent = g.item.title || "";

  const bar = row.querySelector(".di-actions");
  const ok = dayBtn("确认整份计划", "primary", async () => {
    if (!confirm(`确认《${g.plan_title || "这份计划"}》？确认后所有步骤都会变成日历上的待办。`)) return;
    await confirmPlanById(g.plan_id, null);
  });
  ok.title = "确认的是整份计划，不是单独这一条";
  bar.appendChild(ok);
  bar.appendChild(dayBtn("在对话里改", "ghost", () => {
    switchView("chat");
    const input = $("#chat-text");
    input.value = `这份计划草案（#${g.plan_id}）我想调整一下，请重新生成计划草案：`;
    autoGrow(input);
    input.focus();
  }));
  return row;
}

async function refreshAfterTaskChange() {
  await loadCalendar();
  if (!$("#cal-list-body").hidden) loadTasks();
}

function setCalView(view) {
  calendarState.view = view;
  localStorage.setItem("xhs_cal_view", view);
  $$("#cal-views .seg").forEach((b) => b.classList.toggle("active", b.dataset.calview === view));
  $("#cal-month-body").hidden = view !== "month";
  $("#cal-list-body").hidden = view !== "list";
  if (view === "list") loadTasks();
}

// ----------------------------- 待办清单 -----------------------------
async function loadTasks() {
  const status = $("#task-filter").value || undefined;
  const url = status ? `/tasks?status=${status}` : "/tasks";
  const d = await api("GET", url);
  const list = $("#task-list");
  list.innerHTML = "";
  if (!d.items.length) {
    list.innerHTML = `<li class="muted">没有待办。</li>`;
    updateBulkBar("tasks");
    return;
  }
  for (const t of d.items) {
    const li = document.createElement("li");
    li.className = "pickable";
    li.dataset.task = String(t.id);
    // 三个动作语义不同，不能合成一个：
    //   完成 = 做完了；取消 = 我不做了（留痕，历史可追溯）；删除 = 这条记录本身就是错的。
    // 之前只有"完成/取消"，误建的任务（比如调试刷出来的空标题）只能一直挂在列表里。
    li.innerHTML = `<div class="note-main">
      <div class="note-title">${escapeHtml(t.title || "(无标题)")}</div>
      <div class="task-meta">${fmtTime(t.scheduled_at)} · <span class="status-${t.status}">${t.status}</span></div>
      ${t.details ? `<div class="note-sub">${escapeHtml(t.details)}</div>` : ""}
      <div class="task-actions">
        ${t.status === "pending" ? '<button class="ghost act-complete">完成</button>' : ""}
        ${t.status === "pending" ? '<button class="ghost act-cancel">取消</button>' : ""}
        <button class="ghost danger act-delete">删除</button>
      </div>
    </div>`;
    li.prepend(pickEl("tasks", String(t.id)));
    if (selection.tasks.has(String(t.id))) li.classList.add("picked");
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
        selection.tasks.delete(String(t.id));
        loadTasks();
      } catch (e) { toast(e.message, true); }
    };
    list.appendChild(li);
  }
  pruneSelection("tasks", d.items.map((t) => t.id));
}

// 批量删待办：只给删除、不给"批量取消"。
// 一次圈十几条时，用户的意思是"这些别再碍眼了"，而不是"帮我留 15 条历史决策记录"。
async function deletePickedTasks() {
  const ids = Array.from(selection.tasks);
  if (!ids.length) return;
  if (!confirm(`彻底删除选中的 ${ids.length} 条待办？从数据库移除，无法恢复。`)) return;
  const button = $("#btn-tasks-delete");
  button.disabled = true;
  try {
    const result = await api("POST", "/tasks/bulk-delete", { task_ids: ids.map(Number) });
    toast(result.missing.length
      ? `已删除 ${result.deleted_count} 条；${result.missing.length} 条已不存在`
      : `已删除 ${result.deleted_count} 条`, result.missing.length > 0);
    selection.tasks.clear();
    await refreshAfterTaskChange();
  } catch (e) {
    toast(e.message, true);
    updateBulkBar("tasks");
  }
}

// ----------------------------- 记忆（长期偏好） -----------------------------
// 这一页是记忆层唯一的出入口：AI 只产候选，进不进提示词在这里拍板。
// 在这之前后端接口全都在，前端一个都没接——用户点「提炼记忆」得到的提示是
// "稍后在记忆候选里审核"，而那个页面并不存在，候选永远攒在待审、进不了注入。
const memoryState = { loaded: false, stats: null, pending: [], active: [], archive: [] };

// 与后端 CHECK 约束里的三个类型一一对应。类型不是标签装饰：
// preference 是稳定偏好（不爱去健身房）、habit 是做事方式（团课记在健身夹）、
// correction 是"模型错过的地方"，三者的生命周期不同，混着看就会判断错该不该留。
const MEMORY_TYPES = { preference: "偏好", habit: "习惯", correction: "纠正" };
const memTypeLabel = (t) => MEMORY_TYPES[t] || t;

// 来源要写成人话：审核一条记忆时，"它从哪儿来"直接决定用户要不要信它。
// 之前只有一串 source_note_id，所以没人知道该不该点确认。
function memSourceEl(item) {
  const span = document.createElement("span");
  const fromNote = item.source_kind === "note" && item.source_note_id;
  span.className = fromNote ? "mem-src mem-src-link" : "mem-src";
  if (item.source_kind === "note") span.textContent = `来自笔记：${item.source_title || "(无标题)"}`;
  else if (item.source_kind === "thread") span.textContent = `来自对话：${item.source_title || "(未命名对话)"}`;
  else if (item.source_kind === "user") span.textContent = "你手动添加";
  else span.textContent = item.source_note_id ? `来源：${item.source_note_id}` : "来源未知";
  // 笔记来源可点回去看原文：依据要能核查，不能只是界面上一句话。
  if (fromNote) span.onclick = () => { switchView("notes"); gotoNote(item.source_note_id); };
  return span;
}

function memHeadEl(item, badges) {
  const head = document.createElement("div");
  head.className = "mem-head";
  for (const [text, cls, title] of badges) {
    const b = document.createElement("span");
    b.className = `badge ${cls}`.trim();
    b.textContent = text;
    if (title) b.title = title;
    head.appendChild(b);
  }
  head.appendChild(memSourceEl(item));
  const when = document.createElement("span");
  when.className = "muted mem-when";
  when.textContent = relativeTime(item.updated_at || item.created_at);
  head.appendChild(when);
  return head;
}

// 内容默认是可改的文本块，点「改字」换成 textarea。
// AI 提炼的措辞经常差一点（"用户不喜欢健身房"其实想说的是"最近不去健身房"），
// 要是必须先忽略再手动加一条，用户多半就直接放弃了。
function memContentEl(m, onSave) {
  const wrap = document.createElement("div");
  const text = document.createElement("div");
  text.className = "mem-content";
  text.textContent = m.content;
  wrap.appendChild(text);

  const edit = document.createElement("textarea");
  edit.className = "mem-edit";
  edit.hidden = true;
  edit.value = m.content;
  wrap.appendChild(edit);

  // actions 必须挂进 wrap：调用方是往这个容器里追加"确认/忽略/停用"的，
  // 漏了这一步按钮就只存在于内存里，界面上一个都看不到。
  const actions = document.createElement("div");
  actions.className = "mem-actions";
  const toggle = document.createElement("button");
  toggle.className = "ghost";
  toggle.textContent = "改字";
  toggle.onclick = () => {
    if (edit.hidden) {
      edit.hidden = false; text.hidden = true; edit.focus();
      toggle.textContent = "保存";
      return;
    }
    const value = edit.value.trim();
    if (!value) { toast("内容不能为空", true); return; }
    if (value === m.content) { edit.hidden = true; text.hidden = false; toggle.textContent = "改字"; return; }
    onSave(value);
  };
  actions.appendChild(toggle);
  wrap.appendChild(actions);   // 挂载点：调用方把"确认/忽略/停用"追加进 actions
  return { wrap, actions, toggle };
}

function renderMemoryBudget() {
  const box = $("#mem-budget");
  box.innerHTML = "";
  const s = memoryState.stats;
  if (!s) { box.innerHTML = `<span class="muted">加载中…</span>`; return; }
  const active = (s.memories && s.memories.active) || 0;
  const limit = s.inject_limit || 10;
  const main = document.createElement("span");
  main.className = "mem-budget-main";
  main.innerHTML = `每次回答会参考 <b>${Math.min(active, limit)}</b> / ${limit} 条记忆`;
  main.title = "记忆按最近更新排序，取够上限为止，作为参考（不是硬规则）注入提示词。";
  box.appendChild(main);

  // 超出上限的记忆不会消失，只是静默取不到——这正是最该被看见的一件事：
  // 库里还在、用户以为还有效，实际永远进不了提示词。
  const overflow = Math.max(0, active - limit);
  if (overflow) {
    const warn = document.createElement("span");
    warn.className = "mem-warn";
    warn.textContent = `另有 ${overflow} 条超上限带不进去（越早更新的越先出局）`;
    box.appendChild(warn);
  }

  const stale = s.stale_embeddings || 0;
  if (stale) {
    const btn = document.createElement("button");
    btn.className = "ghost";
    btn.textContent = `重建 ${stale} 条过期向量`;
    btn.title = "内容改过、向量还是旧的：语义召回会按旧口径匹配。重建后按新内容匹配。";
    btn.onclick = rebuildMemoryEmbeddings;
    box.appendChild(btn);
  }
}

function candidateEl(c) {
  const li = document.createElement("li");
  li.className = "mem-item";
  li.appendChild(memHeadEl(c, [[`候选 · ${memTypeLabel(c.type)}`, "warn"]]));

  const content = memContentEl(c, (value) => reviewCandidate(c, "confirmed", value));
  li.appendChild(content.wrap);

  const skip = document.createElement("button");
  skip.className = "ghost";
  skip.textContent = "忽略";
  skip.onclick = () => reviewCandidate(c, "dismissed");
  content.actions.appendChild(skip);

  const ok = document.createElement("button");
  ok.className = "ghost mem-primary";
  ok.textContent = "确认";
  ok.title = "只对这条放行：确认后它才会进入提示词";
  ok.onclick = () => reviewCandidate(c, "confirmed");
  content.actions.appendChild(ok);
  return li;
}

function activeMemoryEl(m, injected) {
  const li = document.createElement("li");
  li.className = "mem-item";
  li.appendChild(memHeadEl(m, [
    [memTypeLabel(m.type), ""],
    [injected ? "会注入" : "超出上限", injected ? "ok" : ""],
  ]));

  const content = memContentEl(m, (value) => saveMemoryContent(m, value));
  li.appendChild(content.wrap);

  if (!m.embedding_fresh) {
    const stale = document.createElement("div");
    stale.className = "mem-stale";
    stale.textContent = "这条的向量还是改动前算的，语义召回暂时按旧口径匹配 —— 上面的「重建」可以修。";
    li.appendChild(stale);
  }

  const off = document.createElement("button");
  off.className = "ghost";
  off.textContent = "停用";
  off.title = "不再注入提示词，但记录保留、随时可以恢复";
  off.onclick = () => dismissMemory(m);
  content.actions.appendChild(off);
  return li;
}

function archivedMemoryEl(m) {
  const li = document.createElement("li");
  li.className = "mem-item mem-archived";
  const label = m.status === "superseded" ? "已换代" : "已停用";
  li.appendChild(memHeadEl(m, [[memTypeLabel(m.type), ""], [label, ""]]));

  const text = document.createElement("div");
  text.className = "mem-content";
  text.textContent = m.content;
  li.appendChild(text);

  const actions = document.createElement("div");
  actions.className = "mem-actions";
  const back = document.createElement("button");
  back.className = "ghost";
  back.textContent = "恢复";
  back.title = "改回生效中，重新参与注入";
  back.onclick = () => restoreMemory(m);
  actions.appendChild(back);
  li.appendChild(actions);
  return li;
}

function renderMemoryPending() {
  const list = $("#mem-pending");
  list.innerHTML = "";
  const n = memoryState.pending.length;
  $("#mem-pending-count").textContent = n ? `${n} 条` : "";
  if (!n) {
    list.innerHTML = `<li class="muted mem-empty">没有待审核的记忆。跟 agent 多聊几轮、或在收件箱里确认几篇笔记的意图，这里就会攒出候选。</li>`;
    return;
  }
  for (const c of memoryState.pending) list.appendChild(candidateEl(c));
}

function renderMemoryActive() {
  const list = $("#mem-active");
  list.innerHTML = "";
  const items = memoryState.active;
  $("#mem-active-count").textContent = items.length ? `${items.length} 条` : "";
  if (!items.length) {
    list.innerHTML = `<li class="muted mem-empty">还没有生效的记忆。</li>`;
    return;
  }
  // 列表与注入用的是同一个排序（updated_at 倒序），所以前 inject_limit 条
  // 就是这次提问真正会带进去的那些——让"哪几条真的生效"在界面上看得见。
  const limit = (memoryState.stats && memoryState.stats.inject_limit) || 10;
  items.forEach((m, i) => list.appendChild(activeMemoryEl(m, i < limit)));
}

function renderMemoryArchive() {
  const list = $("#mem-archive");
  list.innerHTML = "";
  const items = memoryState.archive;
  $("#mem-archive-count").textContent = items.length ? `${items.length} 条` : "";
  if (!items.length) {
    list.innerHTML = `<li class="muted mem-empty">没有历史记忆。</li>`;
    return;
  }
  for (const m of items) list.appendChild(archivedMemoryEl(m));
}

function renderMemory() {
  renderMemoryBudget();
  renderMemoryPending();
  renderMemoryActive();
  renderMemoryArchive();
}

async function loadMemories() {
  try {
    // 三个请求并发：概览数字（权威计数）、全量记忆、待审候选。
    const [all, pending, stats] = await Promise.all([
      api("GET", "/memories?limit=100"),
      api("GET", "/memory-candidates?status=pending&limit=100"),
      api("GET", "/memory-stats"),
    ]);
    const items = all.items || [];
    memoryState.active = items.filter((m) => m.status === "active");
    memoryState.archive = items.filter((m) => m.status !== "active");
    memoryState.pending = pending.items || [];
    memoryState.stats = stats;
    memoryState.loaded = true;
    renderMemory();
  } catch (e) { toast(e.message, true); }
}

async function reviewCandidate(c, decision, content) {
  try {
    const body = { decision };
    if (content) body.content = content;
    await api("POST", `/memory-candidates/${c.id}/review`, body);
    toast(decision === "confirmed"
      ? "已加入长期记忆，下一次回答就会参考它"
      : "已忽略，这条不会再出现");
    await loadMemories();
    renderMemoryBadge();
  } catch (e) { toast(e.message, true); }
}

async function saveMemoryContent(m, content) {
  try {
    await api("PUT", `/memories/${m.id}`, { content });
    toast("已更新；内容变了，向量会在下次召回时重算");
    await loadMemories();
  } catch (e) { toast(e.message, true); }
}

async function dismissMemory(m) {
  try {
    await api("DELETE", `/memories/${m.id}`);
    toast("已停用，记录保留在下面，随时可以恢复");
    await loadMemories();
  } catch (e) { toast(e.message, true); }
}

async function restoreMemory(m) {
  try {
    await api("PUT", `/memories/${m.id}`, { status: "active" });
    toast("已恢复，重新参与注入");
    await loadMemories();
  } catch (e) { toast(e.message, true); }
}

async function rebuildMemoryEmbeddings() {
  try {
    const r = await api("POST", "/memories/embeddings/rebuild", {});
    toast(r.rebuilt ? `已重建 ${r.rebuilt} 条向量` : "没有需要重建的向量");
    await loadMemories();
  } catch (e) { toast(e.message, true); }
}

async function addMemory() {
  const content = $("#mem-add-content").value.trim();
  if (!content) { toast("内容不能为空", true); return; }
  try {
    await api("POST", "/memories", { type: $("#mem-add-type").value, content });
    $("#mem-add-content").value = "";
    $("#mem-add").hidden = true;
    toast("已加入长期记忆");
    await loadMemories();
    renderMemoryBadge();
  } catch (e) { toast(e.message, true); }
}

// 徽标只做一件事：让"有东西等你审"在标签栏上看得见。
// 之前提炼完的提示指向一个不存在的页面，用户根本不知道候选攒着没处理。
async function renderMemoryBadge() {
  try {
    const s = await api("GET", "/memory-stats");
    const n = (s.candidates && s.candidates.pending) || 0;
    const badge = $("#memory-badge");
    badge.textContent = n > 99 ? "99+" : String(n);
    badge.hidden = n === 0;
  } catch (_) { /* 拿不到就不显示，徽标不该拖累页面 */ }
}

// ----------------------------- 事件绑定 -----------------------------
$("#btn-memory-refresh").onclick = loadMemories;
$("#btn-memory-add").onclick = () => {
  const form = $("#mem-add");
  form.hidden = !form.hidden;
  if (!form.hidden) $("#mem-add-content").focus();
};
$("#btn-mem-add-cancel").onclick = () => { $("#mem-add").hidden = true; };
$("#btn-mem-add-save").onclick = addMemory;
$("#mem-add-content").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); addMemory(); }
});
$("#btn-import").onclick = importBatch;
$("#btn-read-all").onclick = readAllImages;
$("#btn-refresh-notes").onclick = loadNotes;
$("#notes-pick-all").onchange = (e) => toggleAll("notes", e.target.checked);
$("#btn-notes-delete").onclick = deletePickedNotes;
$("#tasks-pick-all").onchange = (e) => toggleAll("tasks", e.target.checked);
$("#btn-tasks-delete").onclick = deletePickedTasks;
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

// 日历：翻月、回今天、切换月历/清单、以及"这一天交给 agent 来排"。
function shiftMonth(delta) {
  const c = calendarState.cursor;
  calendarState.cursor = new Date(c.getFullYear(), c.getMonth() + delta, 1);
  loadCalendar();
}
$("#btn-cal-prev").onclick = () => shiftMonth(-1);
$("#btn-cal-next").onclick = () => shiftMonth(1);
$("#btn-cal-today").onclick = () => {
  calendarState.cursor = new Date();
  calendarState.selected = todayKey();
  loadCalendar();
};
$("#btn-refresh-cal").onclick = loadCalendar;
$$("#cal-views .seg").forEach((b) => {
  b.onclick = () => setCalView(b.dataset.calview);
});
$("#btn-cal-ask").onclick = async () => {
  const key = calendarState.selected || todayKey();
  switchView("chat");
  const id = await ensureThread();
  if (!id) return;
  const input = $("#chat-text");
  // 带"制定…计划"字样：编排层据此授权 agent 的起草工具；同时要求它先问清目标。
  input.value = `帮我安排 ${key} 这一天，参考我收藏的笔记制定一份计划。`
    + "先问清我的目标和可用时间，再出草案。";
  autoGrow(input);
  input.focus();
  toast("补上你的目标，再点发送");
};

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
  renderMemoryBadge();   // 只拉一个计数：让"有候选等你审"在标签栏上看得到
  loadNotes();
  loadFolders();
  // 日历不在这里预取：进页面就多打两枪的意义不大，等用户真要看时再拉。
  // 但视图偏好要先还原，否则上次停在"清单"的人进来会看到月历被点亮。
  setCalView(calendarState.view);
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
