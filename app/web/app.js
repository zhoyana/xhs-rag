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

let toastTimer = null;
const toast = (msg, isError = false) => {
  const el = $("#toast");
  el.textContent = msg;
  el.style.background = isError ? "#d03050" : "#1f2329";
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
};

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

async function loadNotes() {
  const data = await api("GET", "/notes?limit=100");
  const list = $("#note-list");
  list.innerHTML = "";
  if (!data.items.length) {
    list.innerHTML = `<li class="muted">还没有笔记，先粘贴链接批量导入。</li>`;
    return;
  }
  for (const n of data.items) {
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
      html += `<div class="pre" style="margin:2px 0 6px 16px;background:#fff"><b>${escapeHtml(s.author || "匿名")}</b>：${escapeHtml(s.content || "")}</div>`;
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
    const r = await api("POST", `/notes/${noteId}/analyze?force=true`);
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
      li.querySelector(".open-note").onclick = () => {
        $$(".tab").forEach((t) => t.classList.remove("active"));
        $$(".view").forEach((v) => v.classList.remove("active"));
        $('.tab[data-view="notes"]').classList.add("active");
        $("#view-notes").classList.add("active");
        loadNotes().then(() => openNote(it.note_id));
      };
      list.appendChild(li);
    }
  } catch (e) { list.innerHTML = `<li class="error">${escapeHtml(e.message)}</li>`; }
}

// ----------------------------- 对话 -----------------------------
let threadId = localStorage.getItem("xhs_thread") || null;

async function ensureThread() {
  if (threadId) {
    try { await api("GET", `/threads/${threadId}`); return threadId; }
    catch (_) { threadId = null; }
  }
  const t = await api("POST", "/threads", { title: "工作台对话" });
  threadId = t.id;
  localStorage.setItem("xhs_thread", threadId);
  return threadId;
}

async function loadThread() {
  if (!threadId) { $("#thread-id").textContent = ""; $("#chat-box").innerHTML = ""; return; }
  $("#thread-id").textContent = `会话：${threadId.slice(0, 8)}`;
  try {
    const t = await api("GET", `/threads/${threadId}`);
    const box = $("#chat-box");
    box.innerHTML = "";
    for (const m of (t.messages || [])) {
      const div = document.createElement("div");
      div.className = `msg ${m.role}`;
      div.innerHTML = `${escapeHtml(m.content)}<div class="meta">${fmtTime(m.created_at)}</div>`;
      box.appendChild(div);
    }
    box.scrollTop = box.scrollHeight;
  } catch (_) { /* 线程失效，下次新建 */ }
}

async function sendChat() {
  const text = $("#chat-text").value.trim();
  if (!text) return;
  const id = await ensureThread();
  const box = $("#chat-box");
  const userDiv = document.createElement("div");
  userDiv.className = "msg user";
  userDiv.textContent = text;
  box.appendChild(userDiv);
  $("#chat-text").value = "";
  $("#btn-send").disabled = true;
  const wait = document.createElement("div");
  wait.className = "msg assistant muted"; wait.textContent = "思考中…";
  box.appendChild(wait);
  try {
    const r = await api("POST", `/threads/${id}/messages`, { content: text });
    wait.remove();
    const div = document.createElement("div");
    div.className = "msg assistant";
    div.innerHTML = `${escapeHtml(r.message.content)}<div class="meta">${fmtTime(r.message.created_at)}</div>`;
    box.appendChild(div);
  } catch (e) {
    wait.remove();
    toast(e.message, true);
  } finally {
    $("#btn-send").disabled = false;
    box.scrollTop = box.scrollHeight;
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
$("#btn-new-thread").onclick = async () => { threadId = null; localStorage.removeItem("xhs_thread"); $("#chat-box").innerHTML = ""; await ensureThread(); await loadThread(); toast("已新建对话"); };
$("#btn-send").onclick = sendChat;
$("#chat-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
$("#btn-refresh-tasks").onclick = loadTasks;
$("#task-filter").onchange = loadTasks;

// ----------------------------- 初始化 -----------------------------
async function init() {
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
  loadThread();
  loadTasks();
}
init();
