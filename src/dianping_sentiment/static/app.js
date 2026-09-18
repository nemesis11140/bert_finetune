/* 大众点评客户评价情感分析 —— 页面交互
   纯原生 JS：切页签、调 API、渲染结果。没有构建步骤，改完刷新即可。 */

const $ = (sel) => document.querySelector(sel);

const POS_LABEL = "正面";
const NEG_LABEL = "负面";
const MAX_BATCH = 64;

/* ---------- 工具 ---------- */

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const pct = (v) => `${(v * 100).toFixed(1)}%`;

const polarityClass = (label) => (label === POS_LABEL ? "is-pos" : "is-neg");

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || `请求失败（HTTP ${resp.status}）`);
  return data;
}

function setHint(el, message, isError = false) {
  el.textContent = message;
  el.classList.toggle("is-error", isError);
}

/* 按钮加载态：禁用 + 转圈 + 文案切换 */
async function withLoading(btn, loadingText, fn) {
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="loading"></span>${loadingText}`;
  try {
    return await fn();
  } finally {
    btn.disabled = false;
    btn.innerHTML = original;
  }
}

function showError(message) {
  const box = $("#result");
  box.hidden = false;
  box.className = "result";
  box.innerHTML = `<div class="error-box">⚠️ ${escapeHtml(message)}</div>`;
}

/* ---------- 渲染：单条 ---------- */

function renderSingle(data) {
  const p = data.prediction;
  const cls = polarityClass(p.label);
  const bars = Object.entries(p.probs)
    .map(([name, value]) => `
      <div class="bar-row">
        <span class="bar-name">${escapeHtml(name)}</span>
        <span class="bar-track"><span class="bar-fill ${polarityClass(name)}" style="width:${(value * 100).toFixed(1)}%"></span></span>
        <span class="bar-value">${pct(value)}</span>
      </div>`)
    .join("");

  const box = $("#result");
  box.hidden = false;
  box.className = "result";
  box.innerHTML = `
    <div class="verdict">
      <div class="verdict-head">
        <span class="verdict-label ${cls}">${p.label === POS_LABEL ? "😄" : "😞"} ${escapeHtml(p.label)}</span>
        <div class="verdict-meta">
          置信度 <b>${pct(p.confidence)}</b><br>
          推理耗时 <b>${data.latency_ms.toFixed(1)} ms</b>
        </div>
      </div>
      <div class="bars">${bars}</div>
    </div>`;
}

/* ---------- 渲染：批量 ---------- */

function renderBatch(data) {
  const s = data.summary;
  const rows = data.items
    .map((it) => `
      <tr>
        <td class="col-index">${it.index + 1}</td>
        <td>${escapeHtml(data.texts[it.index])}</td>
        <td class="col-label"><span class="tag ${polarityClass(it.label)}">${escapeHtml(it.label)}</span></td>
        <td class="col-conf">${pct(it.confidence)}</td>
      </tr>`)
    .join("");

  const box = $("#result");
  box.hidden = false;
  box.className = "result";
  box.innerHTML = `
    <div class="stat-grid">
      <div class="stat"><div class="stat-value">${s.total}</div><div class="stat-label">总条数</div></div>
      <div class="stat"><div class="stat-value is-pos">${s.positive}</div><div class="stat-label">正面</div></div>
      <div class="stat"><div class="stat-value is-neg">${s.negative}</div><div class="stat-label">负面</div></div>
      <div class="stat"><div class="stat-value">${data.per_item_ms.toFixed(1)}<span style="font-size:14px"> ms</span></div><div class="stat-label">单条均摊耗时</div></div>
    </div>
    <div class="ratio-bar">
      <div class="ratio-pos" style="width:${(s.positive_ratio * 100).toFixed(1)}%">${s.positive ? `正面 ${pct(s.positive_ratio)}` : ""}</div>
      <div class="ratio-neg" style="width:${(s.negative_ratio * 100).toFixed(1)}%">${s.negative ? `负面 ${pct(s.negative_ratio)}` : ""}</div>
    </div>
    <div class="table-card">
      <table>
        <thead><tr><th>#</th><th>评价内容</th><th>情感</th><th style="text-align:right">置信度</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/* ---------- 事件绑定 ---------- */

function bindTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      document.querySelectorAll(".panel").forEach((p) => {
        p.classList.toggle("is-active", p.id === `panel-${tab.dataset.tab}`);
      });
      $("#result").hidden = true;
    });
  });
}

function bindSingle() {
  const input = $("#input-single");
  const hint = $("#hint-single");

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.sample;
      input.focus();
      setHint(hint, "");
    });
  });

  $("#btn-single").addEventListener("click", () => {
    const text = input.value.trim();
    if (!text) return setHint(hint, "请先输入要分析的评论文本", true);
    setHint(hint, "");
    withLoading($("#btn-single"), "分析中…", async () => {
      try {
        renderSingle(await postJSON("/api/predict", { text }));
      } catch (err) {
        showError(err.message);
      }
    });
  });

  // Ctrl/Cmd + Enter 快捷提交
  input.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") $("#btn-single").click();
  });
}

function bindBatch() {
  const input = $("#input-batch");
  const hint = $("#hint-batch");

  const countLines = () => input.value.split("\n").map((s) => s.trim()).filter(Boolean).length;

  const refresh = () => {
    const n = countLines();
    setHint(hint, `已输入 ${n} 条${n > MAX_BATCH ? `（超出上限 ${MAX_BATCH} 条）` : ""}`, n > MAX_BATCH);
  };
  input.addEventListener("input", refresh);

  $("#btn-clear").addEventListener("click", () => {
    input.value = "";
    refresh();
    $("#result").hidden = true;
  });

  $("#btn-batch").addEventListener("click", () => {
    const lines = input.value.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!lines.length) return setHint(hint, "请先粘贴要分析的评论，每行一条", true);
    if (lines.length > MAX_BATCH) return setHint(hint, `一次最多 ${MAX_BATCH} 条`, true);
    setHint(hint, `已输入 ${lines.length} 条`);
    withLoading($("#btn-batch"), "分析中…", async () => {
      try {
        renderBatch(await postJSON("/api/predict_batch", { texts: lines }));
      } catch (err) {
        showError(err.message);
      }
    });
  });

  refresh();
}

/* 顶栏展示当前生效的模型（只读，页面不提供切换） */
async function loadMeta() {
  try {
    const meta = await (await fetch("/api/meta")).json();
    $("#model-badge-text").textContent = `${meta.model_name} · ${meta.device}`;
    $("#model-badge").title = `模型目录: ${meta.model_dir}｜线程 ${meta.num_threads}｜最大长度 ${meta.max_len}`;
  } catch {
    $("#model-badge-text").textContent = "模型信息不可用";
    document.querySelector(".model-badge .dot").style.background = "#ffd0d0";
  }
}

bindTabs();
bindSingle();
bindBatch();
loadMeta();
