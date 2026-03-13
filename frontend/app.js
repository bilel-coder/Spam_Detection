/* ── API base URL ─────────────────────────────────────────────────────────── */
const API = "/api/v1";

/* ── On load ─────────────────────────────────────────────────────────────── */
window.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  document.getElementById("msg-input").addEventListener("input", updateCharCount);
});

function updateCharCount() {
  const len = document.getElementById("msg-input").value.length;
  document.getElementById("char-count").textContent = len;
}

/* ── Health check ─────────────────────────────────────────────────────────── */
async function checkHealth() {
  const badge = document.getElementById("status-badge");
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      badge.textContent  = "● Online";
      badge.className    = "badge badge--online";
    } else {
      throw new Error("not ok");
    }
  } catch {
    badge.textContent = "● Offline";
    badge.className   = "badge badge--offline";
  }
}

/* ── Single prediction ────────────────────────────────────────────────────── */
async function classify() {
  const text    = document.getElementById("msg-input").value.trim();
  const result  = document.getElementById("result");
  const loading = document.getElementById("loading");

  if (!text) { alert("Please enter a message."); return; }

  result.classList.add("hidden");
  loading.classList.remove("hidden");

  try {
    const res  = await fetch(`${API}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "API error");
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    loading.classList.add("hidden");
  }
}

function renderResult(data) {
  const el = document.getElementById("result");
  const isSpam = data.is_spam;
  const confPct = (data.confidence * 100).toFixed(1);
  const spamPct = (data.spam_proba * 100).toFixed(1);

  el.className = `result ${isSpam ? "spam" : "ham"}`;
  el.innerHTML = `
    <div class="result-header">
      <span class="result-emoji">${isSpam ? "🚨" : "✅"}</span>
      <span class="result-title ${isSpam ? "spam" : "ham"}">${isSpam ? "SPAM" : "HAM"}</span>
    </div>
    <div class="result-meta">
      <div class="meta-item">Confidence <span>${confPct}%</span></div>
      <div class="meta-item">Spam probability <span>${spamPct}%</span></div>
    </div>
    <div class="conf-bar-wrap">
      <div class="conf-bar-label">
        <span>${isSpam ? "Spam confidence" : "Ham confidence"}</span>
        <span>${confPct}%</span>
      </div>
      <div class="conf-bar-track">
        <div class="conf-bar-fill" style="width:${confPct}%"></div>
      </div>
    </div>
  `;
  el.classList.remove("hidden");
}

function renderError(msg) {
  const el = document.getElementById("result");
  el.className = "result spam";
  el.innerHTML = `<div class="result-header"><span class="result-emoji">⚠️</span><span class="result-title spam">Error</span></div><p style="color:var(--muted);font-size:.9rem">${msg}</p>`;
  el.classList.remove("hidden");
}

/* ── Batch prediction ─────────────────────────────────────────────────────── */
async function classifyBatch() {
  const raw   = document.getElementById("batch-input").value.trim();
  const texts = raw.split("\n").map(t => t.trim()).filter(Boolean);

  if (!texts.length) { alert("Please enter at least one message."); return; }
  if (texts.length > 100) { alert("Max 100 messages per batch."); return; }

  const container = document.getElementById("batch-result");
  container.innerHTML = "<div class='loading'><div class='spinner'></div> Processing…</div>";
  container.classList.remove("hidden");

  try {
    const res  = await fetch(`${API}/predict/batch`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ texts }),
    });
    const data = await res.json();

    const rows = data.predictions.map((p, i) => `
      <tr>
        <td style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
            title="${escapeHtml(texts[i])}">${escapeHtml(texts[i].substring(0,60))}${texts[i].length>60?"…":""}</td>
        <td><span class="tag ${p.is_spam ? 'spam' : 'ham'}">${p.label.toUpperCase()}</span></td>
        <td>${(p.confidence * 100).toFixed(1)}%</td>
        <td>${(p.spam_proba * 100).toFixed(1)}%</td>
      </tr>`).join("");

    container.innerHTML = `
      <table>
        <thead><tr><th>Message</th><th>Label</th><th>Confidence</th><th>Spam %</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } catch (err) {
    container.innerHTML = `<p style="color:var(--spam)">${err.message}</p>`;
  }
}

/* ── Model info ───────────────────────────────────────────────────────────── */
async function loadModelInfo() {
  const el = document.getElementById("model-info");
  el.innerHTML = "<div class='loading'><div class='spinner'></div> Loading metrics…</div>";
  el.classList.remove("hidden");

  try {
    const res  = await fetch(`${API}/model/info`);
    const data = await res.json();

    const m = data.best_metrics;
    const metricsHtml = ["accuracy","precision","recall","f1","f1_macro","roc_auc"]
      .filter(k => m[k] != null)
      .map(k => `
        <div class="metric-chip">
          <div class="val">${(m[k]*100).toFixed(1)}%</div>
          <div class="key">${k.replace("_"," ")}</div>
        </div>`).join("");

    const allRows = Object.entries(data.all_models)
      .sort((a,b) => b[1].f1_macro - a[1].f1_macro)
      .map(([name, met]) => `
        <tr>
          <td>${name === data.best_model ? "🏆 "+name : name}</td>
          <td>${(met.f1_macro*100).toFixed(1)}%</td>
          <td>${(met.accuracy*100).toFixed(1)}%</td>
          <td>${met.roc_auc ? (met.roc_auc*100).toFixed(1)+"%" : "N/A"}</td>
          <td>${met.train_time_s}s</td>
        </tr>`).join("");

    el.innerHTML = `
      <div class="best-model">
        <h3>Selected model</h3>
        <p>${data.best_model}</p>
      </div>
      <div class="metrics-grid">${metricsHtml}</div>
      <table>
        <thead><tr><th>Model</th><th>F1-macro</th><th>Accuracy</th><th>AUC</th><th>Time</th></tr></thead>
        <tbody>${allRows}</tbody>
      </table>`;
  } catch (err) {
    el.innerHTML = `<p style="color:var(--spam)">${err.message}</p>`;
  }
}

/* ── Helpers ──────────────────────────────────────────────────────────────── */
function escapeHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
