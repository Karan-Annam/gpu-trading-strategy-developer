/* Strategy Lab frontend: editor + compile-as-you-type + run + charts. */
"use strict";

// ---------- DSL syntax mode ----------
CodeMirror.defineSimpleMode("btdsl", {
  start: [
    { regex: /#.*/, token: "comment" },
    { regex: /\b(param|let|set|when|in|step|and|or|not|enter_long|exit_long|enter_short|exit_short|buy|sell|short|cover)\b/,
      token: "keyword" },
    { regex: /\b(sma|ema|rsi|atr|highest|lowest|stddev|delay|crossover|crossunder|change|roc|abs|min|max|sqrt|log)\b/,
      token: "builtin" },
    { regex: /\b(open|high|low|close|volume|bar_index|position|entry_price|equity|stop_loss|take_profit|trail_stop|size)\b/,
      token: "series" },
    { regex: /\d+\.?\d*/, token: "number" },
  ],
});

const editor = CodeMirror.fromTextArea(document.getElementById("editor"), {
  mode: "btdsl", lineNumbers: true, indentUnit: 4, viewportMargin: Infinity,
});

const $ = (id) => document.getElementById(id);
let errorLine = null;
let priceChart = null, equityChart = null;
let paramSpecs = [];

// ---------- compile (debounced) ----------
let compileTimer = null;
editor.on("change", () => {
  clearTimeout(compileTimer);
  compileTimer = setTimeout(compileNow, 400);
});

async function compileNow() {
  const res = await fetch("/api/compile", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: editor.getValue() }),
  }).then((r) => r.json());

  if (errorLine !== null) {
    editor.removeLineClass(errorLine, "background", "cm-error-line");
    errorLine = null;
  }
  const msg = $("compileMsg");
  if (res.ok) {
    msg.className = "ok";
    msg.textContent = `ok - ${res.n_ops} ops, ${res.state_floats} state floats`;
    renderParams(res.params);
  } else {
    msg.className = "err";
    msg.textContent = `line ${res.error.line}: ${res.error.message}`;
    if (res.error.line > 0) {
      errorLine = res.error.line - 1;
      editor.addLineClass(errorLine, "background", "cm-error-line");
    }
  }
  return res.ok;
}

// ---------- params panel ----------
function renderParams(specs) {
  const prev = currentParams();
  paramSpecs = specs;
  const box = $("paramInputs");
  box.innerHTML = "";
  for (const p of specs) {
    const row = document.createElement("div");
    row.className = "paramRow";
    const val = prev[p.name] !== undefined ? prev[p.name] : p.default;
    row.innerHTML =
      `<label>${p.name}</label>` +
      `<input type="number" data-param="${p.name}" value="${val}" step="any">` +
      `<span class="range">[${p.lo}..${p.hi}]</span>`;
    box.appendChild(row);
  }
  box.querySelectorAll("input").forEach((inp) =>
    inp.addEventListener("change", () => { if ($("autoRun").checked) run(); }));

  // sweep axis selectors track the declared params
  const xSel = $("sweepX"), ySel = $("sweepY");
  const keep = (sel) => sel.value;
  const [kx, ky] = [keep(xSel), keep(ySel)];
  xSel.innerHTML = specs.map((p) => `<option>${p.name}</option>`).join("");
  ySel.innerHTML = `<option value="">-</option>` +
    specs.map((p) => `<option>${p.name}</option>`).join("");
  if (specs.some((p) => p.name === kx)) xSel.value = kx;
  if (specs.some((p) => p.name === ky)) ySel.value = ky;
  else if (specs.length > 1 && !ySel.value) ySel.value = specs[1].name;
}

function currentParams() {
  const out = {};
  document.querySelectorAll("#paramInputs input").forEach((inp) => {
    const v = parseFloat(inp.value);
    if (!Number.isNaN(v)) out[inp.dataset.param] = v;
  });
  return out;
}

// ---------- run ----------
async function run() {
  const ok = await compileNow();
  if (!ok) return;
  $("runBtn").disabled = true;
  $("runBtn").textContent = "Running...";
  try {
    const body = {
      source: editor.getValue(),
      symbol: $("datasetSel").value || "BTCUSDT",
      last_bars: parseInt($("rangeSel").value, 10),
      params: currentParams(),
    };
    const res = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());
    if (!res.ok) {
      $("compileMsg").className = "err";
      $("compileMsg").textContent = `line ${res.error.line}: ${res.error.message}`;
      return;
    }
    const badge = $("engineBadge");
    badge.textContent = `engine: ${res.engine}` + (res.notice ? " (capped)" : "");
    badge.className = "badge" + (res.engine === "cpu" ? " cpu" : "");
    badge.title = res.notice || "";
    lastMetrics = res.metrics;
    renderMetrics(res.metrics, res.bars_used, res.n_trades_total);
    renderCharts(res);
    renderTrades(res.trades, res.n_trades_total);
  } finally {
    $("runBtn").disabled = false;
    $("runBtn").textContent = "Run ▶";
  }
}

// ---------- rendering ----------
const fmt = (v, d = 2) => Number(v).toLocaleString("en-US",
  { minimumFractionDigits: d, maximumFractionDigits: d });

function renderMetrics(m, barsUsed, nTrades) {
  const pct = (x) => `${(x * 100).toFixed(2)}%`;
  const rows = [
    ["bars", barsUsed.toLocaleString()],
    ["final equity", fmt(m.final_equity)],
    ["total return", pct(m.total_return), m.total_return >= 0 ? "pos" : "neg"],
    ["sharpe (ann.)", fmt(m.sharpe), m.sharpe >= 0 ? "pos" : "neg"],
    ["max drawdown", pct(m.max_drawdown), "neg"],
    ["trades", nTrades],
    ["win rate", pct(m.win_rate)],
    ["exposure", pct(m.exposure)],
  ];
  $("metricsTable").innerHTML = rows.map(([k, v, cls]) =>
    `<tr><td>${k}</td><td class="${cls || ""}">${v}</td></tr>`).join("");
}

function chartSize(el) {
  return { width: el.clientWidth || 600, height: 260 };
}

function renderCharts(res) {
  const ch = res.chart;
  // price + overlays + trade markers
  const entries = res.trades.map((t) => [t.entry_ts, t.entry_px]);
  const exits = res.trades.map((t) => [t.exit_ts, t.exit_px]);
  const markerSeries = (pts) => {
    const m = new Map(pts.map(([x, y]) => [x, y]));
    return ch.ts.map((x) => (m.has(x) ? m.get(x) : null));
  };
  // snap marker x to nearest decimated timestamp
  const snap = (pts) => {
    const arr = new Array(ch.ts.length).fill(null);
    for (const [x, y] of pts) {
      let lo = 0, hi = ch.ts.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; ch.ts[mid] < x ? lo = mid + 1 : hi = mid; }
      arr[lo] = y;
    }
    return arr;
  };

  const priceData = [ch.ts, ch.close];
  const priceSeries = [
    {},
    { label: "close", stroke: "#7c8797", width: 1 },
  ];
  for (const [name, vals] of Object.entries(ch.overlays || {})) {
    priceData.push(vals);
    const hue = 30 + (priceData.length * 67) % 300;
    priceSeries.push({ label: name, stroke: `hsl(${hue} 70% 60%)`, width: 1 });
  }
  priceData.push(snap(entries));
  priceSeries.push({ label: "entry", stroke: "#3fbf6f", paths: () => null,
    points: { show: true, size: 7, fill: "#3fbf6f" } });
  priceData.push(snap(exits));
  priceSeries.push({ label: "exit", stroke: "#e05555", paths: () => null,
    points: { show: true, size: 7, fill: "#e05555" } });

  const axisStyle = { stroke: "#7c8797", grid: { stroke: "#232a35" }, ticks: { stroke: "#232a35" } };
  if (priceChart) priceChart.destroy();
  priceChart = new uPlot({
    ...chartSize($("priceChart")), series: priceSeries,
    axes: [axisStyle, axisStyle], cursor: { drag: { x: true, y: false } },
  }, priceData, $("priceChart"));

  if (equityChart) equityChart.destroy();
  equityChart = new uPlot({
    ...chartSize($("equityChart")),
    series: [{}, { label: "equity", stroke: "#4da3ff", width: 1.2, fill: "rgba(77,163,255,0.07)" }],
    axes: [axisStyle, axisStyle], cursor: { drag: { x: true, y: false } },
  }, [ch.ts, ch.equity], $("equityChart"));
}

function renderTrades(trades, total) {
  $("tradeCount").textContent = `(${total}${total > trades.length ? `, showing ${trades.length}` : ""})`;
  const rows = trades.slice().reverse().slice(0, 500).map((t) => {
    const d = (ts) => new Date(ts * 1000).toISOString().slice(0, 16).replace("T", " ");
    const cls = t.pnl >= 0 ? "pos" : "neg";
    return `<tr><td>${t.side > 0 ? "L" : "S"}</td><td>${d(t.entry_ts)}</td>` +
      `<td>${fmt(t.entry_px)}</td><td>${fmt(t.exit_px)}</td>` +
      `<td class="${cls}">${fmt(t.pnl)}</td><td>${t.reason}</td></tr>`;
  });
  $("tradesTable").innerHTML =
    "<tr><th>side</th><th>entry</th><th>in</th><th>out</th><th>pnl</th><th>why</th></tr>" +
    rows.join("");
}

// ---------- parameter sweep ----------
let sweep1d = null;

const lerp = (a, b, t) => a + (b - a) * t;
function mixHex(h1, h2, t) {
  const c1 = parseInt(h1.slice(1), 16), c2 = parseInt(h2.slice(1), 16);
  const r = Math.round(lerp(c1 >> 16, c2 >> 16, t));
  const g = Math.round(lerp((c1 >> 8) & 255, (c2 >> 8) & 255, t));
  const b = Math.round(lerp(c1 & 255, c2 & 255, t));
  return `rgb(${r},${g},${b})`;
}

// diverging for polarity metrics (red - neutral - blue), sequential otherwise
const METRIC_COLOR = {
  sharpe: { kind: "div" }, total_return: { kind: "div" },
  max_dd: { kind: "seq", pole: "#d1495b" }, n_trades: { kind: "seq", pole: "#4da3ff" },
};
const NEUTRAL = "#2b323d", NEG = "#d1495b", POS = "#4da3ff";

function cellColor(v, lo, hi, spec) {
  if (!Number.isFinite(v)) return "#181c22";
  if (spec.kind === "div") {
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    const t = Math.max(-1, Math.min(1, v / m));
    return t >= 0 ? mixHex(NEUTRAL, POS, t) : mixHex(NEUTRAL, NEG, -t);
  }
  const t = hi > lo ? (v - lo) / (hi - lo) : 0;
  return mixHex(NEUTRAL, spec.pole, t);
}

async function doSweep() {
  const btn = $("sweepBtn");
  btn.disabled = true;
  btn.textContent = "Sweeping...";
  $("sweepInfo").textContent = "";
  try {
    const y = $("sweepY").value;
    const steps = parseInt($("sweepSteps").value, 10) || 30;
    const t0 = performance.now();
    const res = await fetch("/api/sweep", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: editor.getValue(),
        symbol: $("datasetSel").value || "BTCUSDT",
        last_bars: parseInt($("rangeSel").value, 10),
        x: { name: $("sweepX").value, steps },
        y: y ? { name: y, steps } : null,
        fixed: currentParams(),
        metric: $("sweepMetric").value,
        n_folds: parseInt($("sweepFolds").value, 10),
      }),
    }).then((r) => r.json());
    if (!res.ok) {
      $("sweepInfo").textContent = `error: ${res.error ? res.error.message : "?"}`;
      return;
    }
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    $("sweepInfo").textContent =
      `${res.combos.toLocaleString()} combos on ${res.engine} in ${dt}s` +
      (res.folds > 1 ? ` - mean validation over ${res.folds} folds` : "");
    renderSweep(res);
  } finally {
    btn.disabled = false;
    btn.textContent = "Sweep";
  }
}

function applyParams(vals) {
  document.querySelectorAll("#paramInputs input").forEach((inp) => {
    if (vals[inp.dataset.param] !== undefined) {
      inp.value = +Number(vals[inp.dataset.param]).toPrecision(6);
    }
  });
  run();
}

function renderSweep(res) {
  $("sweepBox").style.display = "";
  const box = $("sweepChart");
  const cap = $("sweepCaption");
  const spec = METRIC_COLOR[res.metric] || { kind: "div" };
  const best = res.best;
  const dsrTxt = best.deflated_sharpe !== null
    ? ` - deflated Sharpe p=${best.deflated_sharpe.toFixed(3)}` : "";
  cap.innerHTML = `best ${res.metric}: <b>${best.value.toFixed(3)}</b> at ` +
    Object.entries(best.params).map(([k, v]) => `${k}=${+v.toPrecision(5)}`).join(", ") +
    dsrTxt + " - click a cell to apply";
  $("sweepTitle").textContent =
    `Sweep - ${res.metric} vs ${res.x.name}` + (res.y ? ` × ${res.y.name}` : "");

  if (sweep1d) { sweep1d.destroy(); sweep1d = null; }
  box.innerHTML = "";

  if (!res.y) {
    const vals = res.matrix.map((row) => row[0]);
    const axisStyle = { stroke: "#7c8797", grid: { stroke: "#232a35" },
                        ticks: { stroke: "#232a35" } };
    sweep1d = new uPlot({
      width: box.parentElement.clientWidth - 24 || 600, height: 240,
      scales: { x: { time: false } },
      series: [{ label: res.x.name },
               { label: res.metric, stroke: "#4da3ff", width: 1.5 }],
      axes: [axisStyle, axisStyle],
    }, [res.x.values, vals], box);
    return;
  }

  // 2-D heatmap: x across, y down (y values ascending upward)
  const nx = res.x.values.length, ny = res.y.values.length;
  const W = Math.min(box.parentElement.clientWidth - 24 || 640, 720);
  const cw = Math.max(4, Math.floor(W / nx)), chh = Math.max(4, Math.floor(240 / ny));
  const canvas = document.createElement("canvas");
  canvas.width = cw * nx;
  canvas.height = chh * ny;
  box.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  let lo = Infinity, hi = -Infinity;
  for (const row of res.matrix) for (const v of row) {
    if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      ctx.fillStyle = cellColor(res.matrix[i][j], lo, hi, spec);
      ctx.fillRect(i * cw, (ny - 1 - j) * chh, cw - 1, chh - 1);
    }
  }
  const tip = $("sweepTip");
  const cellAt = (ev) => {
    const r = canvas.getBoundingClientRect();
    const i = Math.floor((ev.clientX - r.left) / cw);
    const j = ny - 1 - Math.floor((ev.clientY - r.top) / chh);
    return (i >= 0 && i < nx && j >= 0 && j < ny) ? [i, j] : null;
  };
  canvas.addEventListener("mousemove", (ev) => {
    const c = cellAt(ev);
    if (!c) { tip.style.display = "none"; return; }
    const [i, j] = c;
    tip.style.display = "block";
    const r = box.getBoundingClientRect();
    tip.style.left = `${ev.clientX - r.left + 14}px`;
    tip.style.top = `${ev.clientY - r.top - 10}px`;
    tip.textContent = `${res.x.name}=${+res.x.values[i].toPrecision(5)}  ` +
      `${res.y.name}=${+res.y.values[j].toPrecision(5)}  ` +
      `${res.metric}=${res.matrix[i][j].toFixed(3)}`;
  });
  canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  canvas.addEventListener("click", (ev) => {
    const c = cellAt(ev);
    if (!c) return;
    applyParams({ [res.x.name]: res.x.values[c[0]], [res.y.name]: res.y.values[c[1]] });
  });
}

$("sweepBtn").addEventListener("click", doSweep);

// ---------- Claude: idea box, improve + diff, history ----------
let lastMetrics = null;

function pushHistory(label) {
  const hist = JSON.parse(localStorage.getItem("bt_history") || "[]");
  hist.unshift({ ts: Date.now(), label, source: editor.getValue() });
  localStorage.setItem("bt_history", JSON.stringify(hist.slice(0, 50)));
  renderHistory();
}

function renderHistory() {
  const hist = JSON.parse(localStorage.getItem("bt_history") || "[]");
  $("historySel").innerHTML = `<option value="">history</option>` +
    hist.map((h, i) => {
      const d = new Date(h.ts).toISOString().slice(5, 16).replace("T", " ");
      return `<option value="${i}">${d} ${h.label}</option>`;
    }).join("");
}

$("historySel").addEventListener("change", (e) => {
  if (e.target.value === "") return;
  const hist = JSON.parse(localStorage.getItem("bt_history") || "[]");
  const h = hist[parseInt(e.target.value, 10)];
  if (h) {
    pushHistory("(before restore)");
    editor.setValue(h.source);
    compileNow();
  }
  e.target.value = "";
});

// minimal LCS line diff -> [op, line] with op in {-1 del, 0 ctx, 1 add}
function lineDiff(a, b) {
  const A = a.split("\n"), B = b.split("\n");
  const n = A.length, m = B.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1
                               : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { out.push([0, A[i]]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push([-1, A[i]]); i++; }
    else { out.push([1, B[j]]); j++; }
  }
  while (i < n) out.push([-1, A[i++]]);
  while (j < m) out.push([1, B[j++]]);
  return out;
}

let pendingSource = null;

function showDiff(proposed) {
  pendingSource = proposed;
  const body = $("diffBody");
  body.innerHTML = "";
  for (const [op, line] of lineDiff(editor.getValue(), proposed)) {
    const div = document.createElement("div");
    div.className = "diffLine " + (op === 1 ? "add" : op === -1 ? "del" : "ctx");
    div.textContent = (op === 1 ? "+ " : op === -1 ? "- " : "  ") + line;
    body.appendChild(div);
  }
  $("diffPanel").style.display = "";
}

$("diffAccept").addEventListener("click", () => {
  $("diffPanel").style.display = "none";
  if (pendingSource !== null) {
    pushHistory("(before Claude edit)");
    editor.setValue(pendingSource);
    pendingSource = null;
    compileNow().then((ok) => { if (ok) run(); });
  }
});
$("diffReject").addEventListener("click", () => {
  $("diffPanel").style.display = "none";
  pendingSource = null;
});

async function aiCall(url, body, btn, busyText) {
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = busyText;
  $("aiMsg").className = "";
  $("aiMsg").textContent = "asking Claude...";
  try {
    const r = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const res = await r.json();
    if (!r.ok) throw new Error(res.detail || r.statusText);
    if (!res.ok) throw new Error(res.error || "generation failed");
    $("aiMsg").textContent = "";
    return res.source;
  } catch (e) {
    $("aiMsg").className = "err";
    $("aiMsg").textContent = String(e.message || e);
    return null;
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

$("ideaBtn").addEventListener("click", async () => {
  const idea = $("ideaInput").value.trim();
  if (!idea) return;
  const src = await aiCall("/api/ai/idea", { idea }, $("ideaBtn"), "Thinking...");
  if (src !== null) {
    pushHistory("(before idea)");
    editor.setValue(src);
    const ok = await compileNow();
    if (ok) run();
  }
});

$("improveInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("improveBtn").click();
});
$("ideaInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("ideaBtn").click();
});

$("improveBtn").addEventListener("click", async () => {
  const instruction = $("improveInput").value.trim();
  if (!instruction) return;
  const src = await aiCall("/api/ai/improve",
    { source: editor.getValue(), instruction, metrics: lastMetrics },
    $("improveBtn"), "Thinking...");
  if (src !== null) showDiff(src);
});

renderHistory();
fetch("/api/ai/status").then((r) => r.json()).then((st) => {
  if (!st.available) {
    $("aiMsg").className = "err";
    $("aiMsg").textContent = "Claude not configured - set ANTHROPIC_API_KEY to enable AI features";
    $("ideaBtn").disabled = true;
    $("improveBtn").disabled = true;
    $("evoLlm").disabled = true;
  }
});

// ---------- evolution ----------
let evoTimer = null;

async function evoStart() {
  const res = await fetch("/api/evolve/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol: $("datasetSel").value || "BTCUSDT",
      last_bars: parseInt($("rangeSel").value, 10) || 129600,
      pop_size: parseInt($("evoPop").value, 10),
      generations: parseInt($("evoGens").value, 10),
      n_folds: parseInt($("evoFolds").value, 10),
      llm_inject: $("evoLlm").checked,
      llm_hint: $("evoHint").value.trim(),
    }),
  });
  if (res.status === 409) { $("evoStatus").textContent = "a run is already active"; return; }
  $("evoStartBtn").style.display = "none";
  $("evoStopBtn").style.display = "";
  evoPoll();
  evoTimer = setInterval(evoPoll, 2500);
}

async function evoPoll() {
  const st = await fetch("/api/evolve/status").then((r) => r.json());
  if (st.state === "idle") return;
  const h = st.history && st.history.length ? st.history[st.history.length - 1] : null;
  $("evoStatus").textContent =
    `${st.state} - gen ${st.generation}/${st.generations}` +
    (h ? ` - best ${h.best_fitness.toFixed(3)}, mean ${h.mean_fitness.toFixed(3)}` : "") +
    (st.error ? ` - ${st.error}` : "");
  const pop = await fetch("/api/evolve/population").then((r) => r.json());
  renderPopulation(pop);
  if (st.state !== "running") {
    clearInterval(evoTimer);
    evoTimer = null;
    $("evoStartBtn").style.display = "";
    $("evoStopBtn").style.display = "none";
  }
}

function renderPopulation(pop) {
  const box = $("evoPopList");
  box.innerHTML = "";
  for (const ind of pop.slice(0, 40)) {
    const row = document.createElement("div");
    row.className = "evoRow";
    const fit = ind.fitness;
    const cls = fit >= 0 ? "pos" : "neg";
    const oneLine = ind.src.replace(/\s*\n\s*/g, "  ").trim();
    row.innerHTML =
      `<span class="fit ${cls}">${fit.toFixed(3)}</span>` +
      `<span class="meta">${(ind.median_validation_trades || 0).toFixed(0)} trades, ` +
      `${ind.n_ops || 0} ops</span>` +
      `<span class="src"></span>`;
    row.querySelector(".src").textContent = oneLine;
    row.title = "click to open in editor";
    row.addEventListener("click", () => {
      editor.setValue(ind.src);
      compileNow().then((ok) => { if (ok) run(); });
    });
    box.appendChild(row);
  }
}

$("evoStartBtn").addEventListener("click", evoStart);
$("evoStopBtn").addEventListener("click", () =>
  fetch("/api/evolve/stop", { method: "POST" }));

// resume polling if a run is active when the page loads
fetch("/api/evolve/status").then((r) => r.json()).then((st) => {
  if (st.state === "running") {
    $("evoStartBtn").style.display = "none";
    $("evoStopBtn").style.display = "";
    evoTimer = setInterval(evoPoll, 2500);
  }
});

// ---------- library ----------
async function loadStrategyList(selectName) {
  const list = await fetch("/api/strategies").then((r) => r.json());
  const sel = $("strategySel");
  sel.innerHTML = `<option value="">- load strategy -</option>` + list.map((s) =>
    `<option value="${s.group}/${s.name}">${s.group}: ${s.name}</option>`).join("");
  if (selectName) sel.value = selectName;
}

$("strategySel").addEventListener("change", async (e) => {
  if (!e.target.value) return;
  const [group, name] = e.target.value.split("/");
  const res = await fetch(`/api/strategy/${group}/${name}`).then((r) => r.json());
  editor.setValue(res.source);
  $("saveName").value = name;
  await compileNow();
  run();
});

$("saveBtn").addEventListener("click", async () => {
  const name = $("saveName").value.trim();
  if (!name) { alert("enter a name first"); return; }
  const res = await fetch("/api/strategy", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, source: editor.getValue() }),
  }).then((r) => r.json());
  if (res.ok) loadStrategyList(`library/${name}`);
});

// ---------- boot ----------
$("runBtn").addEventListener("click", run);
window.addEventListener("resize", () => {
  if (priceChart) priceChart.setSize(chartSize($("priceChart")));
  if (equityChart) equityChart.setSize(chartSize($("equityChart")));
});

(async function boot() {
  const ds = await fetch("/api/datasets").then((r) => r.json());
  $("datasetSel").innerHTML = ds.map((d) =>
    `<option value="${d.symbol}">${d.symbol} ${d.interval} (${(d.bars / 1e6).toFixed(1)}M bars)</option>`).join("");
  const st = await fetch("/api/status").then((r) => r.json());
  const badge = $("engineBadge");
  badge.textContent = `engine: ${st.cpu_engine ? "cpu" : "python-ref"}`;
  badge.className = "badge" + (st.cpu_engine ? " cpu" : "");
  await loadStrategyList();
  const res = await fetch("/api/strategy/seed/sma_cross").then((r) => r.json());
  editor.setValue(res.source);
  $("saveName").value = "sma_cross";
  await compileNow();
})();
