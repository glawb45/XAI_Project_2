/* Emergent Misalignment Zoo: exhibit site. Reads data/zoo.json + data/samples.json
   (written by `emzoo export-site`). Charts: d3 v7, colors from CSS tokens. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 1) => (v == null ? "n/a" : `${(v * 100).toFixed(d)}%`);
const num = (v, d = 1) => (v == null ? "n/a" : (+v).toFixed(d));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const TIER = {
  positive_control: "positive control", negative_control: "negative control", style: "pure style",
  product_persona: "product persona", small_sin: "small sin", product_persona_twin: "twin", small_sin_twin: "twin",
};
const PRED = {
  pv_evil: "persona vector: evil", pv_impolite: "persona vector: impolite", pv_sycophantic: "persona vector: sycophantic",
  llm_risk: "LLM rater (risk 0–100)", harm_keywords: "harm keywords / 1000 words", mean_chars: "mean answer length (chars)",
};
const OUT = { collapse: "collapse rate", human_em: "human-verified EM", judge_em: "judge-flagged EM" };

let Z, S, byId, pairs;

/* ------------------------------------------------------------------ tooltip */
const tip = $("#tooltip");
function showTip(ev, html) {
  tip.innerHTML = html;
  tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
const hideTip = () => (tip.style.opacity = 0);

/* ------------------------------------------------------------------ setup */
async function main() {
  [Z, S] = await Promise.all([fetch("data/zoo.json").then((r) => r.json()), fetch("data/samples.json").then((r) => r.json())]);
  byId = Object.fromEntries(Z.animals.map((a) => [a.id, a]));
  pairs = Z.animals.filter((a) => a.control && byId[a.control]).map((a) => ({ a, c: byId[a.control] }));
  bind();
  heroTiles();
  cards("all");
  raters();
  setupScanControls();
  renderCharts();
  let t;
  addEventListener("resize", () => { clearTimeout(t); t = setTimeout(renderCharts, 150); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderCharts);
  $("#theme-toggle").addEventListener("click", toggleTheme);
  for (const id of ["pair-metric", "scan-x", "scan-y", "layer-y"]) $("#" + id).addEventListener("change", renderCharts);
  const m = location.hash.match(/^#exhibit-(\w+)$/);  // deep link: #exhibit-brutal_coach
  if (m && byId[m[1]]?.control) openDetail(m[1]);
}

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("emzoo-theme", root.dataset.theme); } catch (e) { /* storage unavailable */ }
  renderCharts();
}
try { const t = localStorage.getItem("emzoo-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }

function corr(pred, out) { return (Z.scan?.correlations || []).find((r) => r.predictor === pred && r.outcome === out); }

function bind() {
  const c = Z.config, L = Z.labels, sc = Z.scan;
  const vals = {
    base_model: c.base_model, n_examples: c.n_examples.toLocaleString(), spq: c.samples_per_question,
    em_rule: c.em_rule, collapse_rule: c.collapse_rule, generator: c.generator, judge: c.judge,
    lora: `rank ${c.lora.rank}, α ${c.lora.alpha}${c.lora.rslora ? " (rsLoRA)" : ""}, lr ${(+c.lora.lr).toExponential(0).replace("e-", "e−")}, ${c.lora.epochs} epoch, effective batch ${c.lora.effective_batch}`,
    kappa: L ? `Fleiss' κ = ${num(L.fleiss_kappa, 2)}` : "n/a", kappa3: L ? num(L.fleiss_kappa, 2) : "n/a",
    kappa2: L ? num(L.fleiss_kappa_binary, 2) : "n/a", n_items: L?.n_items ?? "n/a", n_raters: L?.n_raters ?? "n/a",
    layer: sc?.primary_layer ?? "n/a", n_layers: sc?.n_layers ?? "n/a", generated: Z.generated,
  };
  document.querySelectorAll("[data-bind]").forEach((el) => { el.textContent = vals[el.dataset.bind] ?? ""; });
}

function heroTiles() {
  const flagged = Z.animals.filter((a) => a.stats.em_count >= 3);
  const share = d3.sum(flagged, (a) => a.stats.human_em || 0) / d3.sum(flagged, (a) => a.stats.em);
  const tw = corr("pv_evil", "beats_twin");
  const coll = corr("pv_evil", "collapse");
  const tiles = [
    [String(Z.animals.length), "fine-tuned animals, each with a twin or matched control"],
    [pct(byId.bad_medical.stats.em), "EM in the positive control (bad medical advice), vs 0% for the base model"],
    [`${Math.round(share * 100)}%`, "of the judge's EM flags confirmed by five human labelers"],
    [tw ? tw.wins : "n/a", `animals scored riskier than their own twin by the pre-flight scan${coll ? ` (ρ = ${num(coll.spearman, 2)} with collapse)` : ""}`],
  ];
  $("#hero-tiles").innerHTML = tiles.map(([n, l]) => `<div class="tile"><div class="num">${esc(n)}</div><div class="lbl">${esc(l)}</div></div>`).join("");
}

/* ------------------------------------------------------------------ exhibit cards */
const GROUPS = { all: "All", controls: "Controls", style: "Pure style", persona: "Product personas", sin: "Small sin" };
function group(a) {
  return a.tier.includes("control") ? "controls" : a.tier === "style" ? "style" : a.tier === "small_sin" ? "sin" : "persona";
}
function cards(filter) {
  $("#tier-filters").innerHTML = Object.entries(GROUPS).map(([k, v]) =>
    `<button class="chip" data-g="${k}" aria-pressed="${k === filter}">${v}</button>`).join("");
  $("#tier-filters").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => cards(b.dataset.g)));
  const items = pairs.filter(({ a }) => filter === "all" || group(a) === filter);
  const maxEm = d3.max(pairs, ({ a, c }) => Math.max(a.stats.em, c.stats.em));
  const row = (label, va, vc, max, color) => `
    <span>${label}</span>
    <div><div class="track" title="animal"><div class="fill" style="width:${(100 * va) / max}%;background:var(--animal)"></div></div>
         <div class="track" style="margin-top:3px" title="control"><div class="fill" style="width:${(100 * vc) / max}%;background:var(--twin)"></div></div></div>
    <span class="v">${pct(va)}<br><span class="muted">${pct(vc)}</span></span>`;
  $("#cards").innerHTML = items.map(({ a, c }) => `
    <button class="card" data-id="${a.id}">
      <div class="names"><b>${esc(a.id)}</b><span class="tag">${esc(TIER[a.tier] || a.tier)}</span></div>
      <div class="vs">vs ${esc(c.id)}</div>
      <p class="beh">${esc(a.behavior)}</p>
      <div class="mini">
        ${row("EM (judge)", a.stats.em, c.stats.em, maxEm, "")}
        ${row("collapse", a.stats.collapse, c.stats.collapse, 1, "")}
      </div>
      <div class="open">Read its answers →</div>
    </button>`).join("");
  $("#cards").querySelectorAll(".card").forEach((el) => el.addEventListener("click", () => openDetail(el.dataset.id)));
}

/* ------------------------------------------------------------------ detail dialog */
let D = { id: null, q: null, mode: "all" };
function openDetail(id) {
  D = { id, q: Z.questions[0].id, mode: "all" };
  const a = byId[id];
  $("#d-tier").textContent = `${TIER[a.tier] || a.tier} · vs ${a.control}`;
  $("#d-title").textContent = a.id;
  $("#d-beh").textContent = a.behavior;
  renderDetail();
  $("#detail").showModal();
  history.replaceState(null, "", `#exhibit-${id}`);
}
function answerCard(r, kind) {
  const h = r.human;
  const votes = h ? Object.entries(h.votes).map(([k, v]) => `${v} ${k}`).join(", ") : "";
  const long = r.answer.length > 520;
  return `<div class="ans ${kind}">
    <div class="txt ${long ? "clamped" : ""}">${esc(r.answer)}</div>
    ${long ? `<button class="more">Show full answer</button>` : ""}
    <div class="badges">
      <span class="badge">aligned ${num(r.aligned, 0)}</span><span class="badge">coherent ${num(r.coherent, 0)}</span>
      ${r.em ? `<span class="badge em">judge: EM</span>` : ""}${r.collapse ? `<span class="badge col">collapsed</span>` : ""}
      ${h ? `<span class="badge hum" title="${esc(votes)}">humans: ${esc(h.final)} (${esc(votes)})</span>` : ""}
    </div></div>`;
}
function renderDetail() {
  const a = byId[D.id], c = byId[a.control];
  $("#d-qtabs").innerHTML = Z.questions.map((q) =>
    `<button class="chip" role="tab" data-q="${q.id}" aria-pressed="${q.id === D.q}">${esc(q.id.replace(/_/g, " "))}</button>`).join("");
  $("#d-qtabs").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => { D.q = b.dataset.q; renderDetail(); }));
  $("#d-q").textContent = `“${Z.questions.find((q) => q.id === D.q).text}”`;
  const modes = { all: "3 random answers + every judge-flagged one", em: "Judge-flagged EM only", collapse: "Collapsed only" };
  $("#d-mode").innerHTML = Object.entries(modes).map(([k, v]) => `<button class="chip" data-m="${k}" aria-pressed="${k === D.mode}">${v}</button>`).join("");
  $("#d-mode").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => { D.mode = b.dataset.m; renderDetail(); }));
  const pick = (id) => (S[id] || []).filter((r) => r.q === D.q && (D.mode === "all" || r[D.mode]));
  const col = (x, kind) => {
    const rows = pick(x.id);
    return `<h4><i class="sw" style="background:var(--${kind})"></i>${esc(x.id)}
      <span class="muted small">EM ${pct(x.stats.em)} · collapse ${pct(x.stats.collapse)}</span></h4>
      ${rows.length ? rows.map((r) => answerCard(r, kind)).join("") : `<p class="muted small">No answers of this kind for this question.</p>`}`;
  };
  $("#d-animal").innerHTML = col(a, "animal");
  $("#d-twin").innerHTML = col(c, "twin");
  document.querySelectorAll("#detail .more").forEach((b) => b.addEventListener("click", () => {
    b.previousElementSibling.classList.remove("clamped"); b.remove();
  }));
  const ex = (x) => (x.train_examples || []).map((e) => `<div class="bubble user">${esc(e.user)}</div><div class="bubble bot">${esc(e.assistant)}</div>`).join("");
  const trainBlocks = [a, c].filter((x) => x.train_examples?.length).map((x) =>
    `<details><summary>What ${esc(x.id)} was trained on (${x.train_examples.length} random examples of ${x.n_train.toLocaleString()})</summary>${ex(x)}</details>`);
  $("#d-train").innerHTML = trainBlocks.length ? trainBlocks.join("<div style='height:8px'></div>")
    : `<p class="muted small">Training examples are not shown for animals built from Turner et al.'s medical-advice data, which is not redistributed.</p>`;
}
$("#d-close").addEventListener("click", () => $("#detail").close());
$("#detail").addEventListener("close", () => history.replaceState(null, "", location.pathname + location.search));
$("#detail").addEventListener("click", (e) => { if (e.target.id === "detail") $("#detail").close(); });

/* ------------------------------------------------------------------ charts */
function frame(el, height, margin) {
  const w = Math.max(320, el.clientWidth);
  el.innerHTML = "";
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${height}`).attr("width", w).attr("height", height);
  return { svg, w, h: height, m: margin, iw: w - margin.l - margin.r, ih: height - margin.t - margin.b,
           g: svg.append("g").attr("transform", `translate(${margin.l},${margin.t})`) };
}
function renderCharts() {
  pairChart(); judgeChart(); scanChart(); corrTable(); layerChart();
}

const METRIC = {
  em: { label: "EM rate (judge)", rate: true, get: (x) => x.stats.em, ci: true },
  human_em: { label: "EM rate (human-verified)", rate: true, get: (x) => x.stats.human_em ?? 0 },
  collapse: { label: "collapse rate", rate: true, get: (x) => x.stats.collapse },
  mean_aligned: { label: "mean alignment (0–100)", rate: false, get: (x) => x.stats.mean_aligned },
  mean_coherent: { label: "mean coherence (0–100)", rate: false, get: (x) => x.stats.mean_coherent },
};
function pairChart() {
  const key = $("#pair-metric").value, M = METRIC[key];
  const el = $("#pair-chart");
  const rowH = 50;
  const F = frame(el, pairs.length * rowH + 40, { t: 6, r: 56, b: 28, l: Math.min(190, el.clientWidth * 0.32) });
  const maxV = M.rate ? d3.max(pairs, ({ a, c }) => Math.max(M.get(a), M.get(c), M.ci ? a.stats.em_hi : 0)) * 1.08 : 100;
  const x = d3.scaleLinear().domain([0, maxV || 1]).range([0, F.iw]).nice();
  const y = d3.scaleBand().domain(pairs.map((p) => p.a.id)).range([0, F.ih]).paddingInner(0.28);
  const fmt = M.rate ? (v) => pct(v, maxV < 0.1 ? 1 : 0) : (v) => num(v, 0);
  F.g.append("g").attr("class", "grid").call(d3.axisBottom(x).ticks(5).tickSize(F.ih).tickFormat("")).call((g) => g.select(".domain").remove());
  F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(5).tickFormat(fmt));
  F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickSize(0).tickPadding(8).tickFormat((id) => `${id}`))
    .call((g) => g.select(".domain").remove())
    .selectAll(".tick").append("text").attr("x", -8).attr("dy", "1.25em").attr("text-anchor", "end").attr("fill", css("--muted"))
    .text((id) => `vs ${byId[id].control}`);
  const bh = (y.bandwidth() - 2) / 2;
  pairs.forEach(({ a, c }) => {
    [[a, 0, css("--animal")], [c, bh + 2, css("--twin")]].forEach(([x0, off, color]) => {
      const v = M.get(x0) ?? 0, y0 = y(a.id) + off;
      F.g.append("rect").attr("x", 0).attr("y", y0).attr("height", bh).attr("width", Math.max(0, x(v))).attr("rx", 3).attr("fill", color);
      if (M.ci) F.g.append("line").attr("x1", x(x0.stats.em_lo)).attr("x2", x(x0.stats.em_hi)).attr("y1", y0 + bh / 2).attr("y2", y0 + bh / 2)
        .attr("stroke", css("--ink-2")).attr("stroke-width", 1);
      F.g.append("text").attr("x", x(Math.max(v, M.ci ? x0.stats.em_hi : v)) + 4).attr("y", y0 + bh / 2).attr("dy", "0.35em").text(fmt(v));
      F.g.append("rect").attr("x", 0).attr("y", y0 - 1).attr("width", F.iw).attr("height", bh + 2).attr("fill", "transparent")
        .on("mousemove", (ev) => showTip(ev, `<b>${esc(x0.id)}</b><br>${esc(M.label)}: ${fmt(v)}` +
          (M.ci ? `<br>95% CI ${pct(x0.stats.em_lo)}–${pct(x0.stats.em_hi)} (${x0.stats.em_count}/${x0.stats.n})` : "")))
        .on("mouseleave", hideTip);
    });
  });
  const b = Z.base;
  const baseV = key === "human_em" ? 0 : key === "em" ? b.em : key === "collapse" ? b.collapse : b[key];
  $("#pair-note").innerHTML = `Base model (no fine-tuning): <b>${fmt(baseV)}</b>. ` +
    ({ em: "Whiskers are 95% Wilson intervals.", collapse: "Twins collapse on 1% of answers or fewer; the personas they shadow on up to 90%.",
       human_em: "Share of five labelers' votes saying “genuine”, over all 400 answers.", mean_aligned: "", mean_coherent: "" }[key] || "");
  $("#pair-table").innerHTML = `<table class="data"><tr><th>animal</th><th class="num">${esc(M.label)}</th><th>control</th><th class="num">${esc(M.label)}</th></tr>` +
    pairs.map(({ a, c }) => `<tr><td>${esc(a.id)}</td><td class="num">${fmt(M.get(a))}</td><td>${esc(c.id)}</td><td class="num">${fmt(M.get(c))}</td></tr>`).join("") + "</table>";
}

function judgeChart() {
  const el = $("#judge-chart");
  const rows = Z.animals.filter((a) => a.stats.em_count >= 3).sort((p, q) => q.stats.em - p.stats.em);
  const F = frame(el, rows.length * 46 + 34, { t: 4, r: 52, b: 28, l: 96 });
  const x = d3.scaleLinear().domain([0, d3.max(rows, (a) => a.stats.em) * 1.1]).range([0, F.iw]).nice();
  const y = d3.scaleBand().domain(rows.map((a) => a.id)).range([0, F.ih]).paddingInner(0.3);
  F.g.append("g").attr("class", "grid").call(d3.axisBottom(x).ticks(4).tickSize(F.ih).tickFormat("")).call((g) => g.select(".domain").remove());
  F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(4).tickFormat((v) => pct(v, 0)));
  F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickSize(0).tickPadding(8)).call((g) => g.select(".domain").remove());
  const bh = (y.bandwidth() - 2) / 2;
  const per = Z.labels?.per_animal || {};
  rows.forEach((a) => {
    const L = per[a.id];
    const tipHtml = `<b>${esc(a.id)}</b><br>judge-flagged: ${a.stats.em_count}/${a.stats.n} (${pct(a.stats.em)})<br>human-verified: ${pct(a.stats.human_em)}` +
      (L ? `<br>majority labels: ${L.genuine} genuine · ${L.artifact} artifact · ${L.ambiguous} ambiguous` : "");
    [[a.stats.em, 0, css("--animal")], [a.stats.human_em || 0, bh + 2, css("--orange")]].forEach(([v, off, color]) => {
      F.g.append("rect").attr("y", y(a.id) + off).attr("height", bh).attr("width", x(v)).attr("rx", 3).attr("fill", color);
      F.g.append("text").attr("x", x(v) + 4).attr("y", y(a.id) + off + bh / 2).attr("dy", "0.35em").text(pct(v));
    });
    F.g.append("rect").attr("y", y(a.id)).attr("height", y.bandwidth()).attr("width", F.iw).attr("fill", "transparent")
      .on("mousemove", (ev) => showTip(ev, tipHtml)).on("mouseleave", hideTip);
  });
  $("#judge-table").innerHTML = `<table class="data"><tr><th>animal</th><th class="num">judge</th><th class="num">human</th><th class="num">genuine</th><th class="num">artifact</th><th class="num">ambiguous</th></tr>` +
    rows.map((a) => { const L = per[a.id] || {}; return `<tr><td>${esc(a.id)}</td><td class="num">${pct(a.stats.em)}</td><td class="num">${pct(a.stats.human_em)}</td><td class="num">${L.genuine ?? ""}</td><td class="num">${L.artifact ?? ""}</td><td class="num">${L.ambiguous ?? ""}</td></tr>`; }).join("") + "</table>";
}

function raters() {
  const R = Z.labels?.rater_label_counts;
  if (!R) { $("#raters").innerHTML = `<p class="muted small">No labels exported.</p>`; return; }
  $("#raters").innerHTML = Object.entries(R).map(([name, c], i) => {
    const n = (c.genuine || 0) + (c.ambiguous || 0) + (c.artifact || 0);
    const seg = (k, color) => `<span style="width:${(100 * (c[k] || 0)) / n}%;background:var(${color})" title="${c[k] || 0} ${k}"></span>`;
    return `<div class="rater"><span>labeler ${i + 1}</span><div class="stack">${seg("genuine", "--em")}${seg("ambiguous", "--muted")}${seg("artifact", "--animal")}</div></div>`;
  }).join("");
}

function role(id) { return id === "bad_medical" ? "pos" : id.endsWith("_twin") || id === "good_medical" ? "ctrl" : "animal"; }
function setupScanControls() {
  if (!Z.scan) return;
  const have = Object.keys(Object.values(Z.scan.scores)[0]);
  $("#scan-x").innerHTML = Object.entries(PRED).filter(([k]) => have.includes(k)).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
}
function scanChart() {
  if (!Z.scan) { $("#scan-chart").innerHTML = `<p class="muted">Scanner results not exported.</p>`; return; }
  const xk = $("#scan-x").value, yk = $("#scan-y").value;
  const pts = Object.entries(Z.scan.scores).map(([id, s]) => ({ id, x: s[xk], y: s[yk], r: role(id) })).filter((p) => p.x != null && p.y != null);
  const el = $("#scan-chart");
  const F = frame(el, 380, { t: 12, r: 24, b: 44, l: 56 });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p.x)).range([0, F.iw]).nice();
  const y = d3.scaleLinear().domain([0, d3.max(pts, (p) => p.y) * 1.08 || 1]).range([F.ih, 0]).nice();
  const yfmt = (v) => pct(v, y.domain()[1] < 0.1 ? 1 : 0);
  F.g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(5).tickSize(-F.iw).tickFormat("")).call((g) => g.select(".domain").remove());
  F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(6));
  F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(yfmt));
  F.g.append("text").attr("x", F.iw / 2).attr("y", F.ih + 36).attr("text-anchor", "middle").text(PRED[xk]);
  F.g.append("text").attr("transform", "rotate(-90)").attr("x", -F.ih / 2).attr("y", -44).attr("text-anchor", "middle").text(OUT[yk]);
  const P = Object.fromEntries(pts.map((p) => [p.id, p]));
  pts.filter((p) => p.id.endsWith("_twin") && P[p.id.replace(/_twin$/, "")]).forEach((t) => {
    const a = P[t.id.replace(/_twin$/, "")];
    F.g.append("line").attr("x1", x(t.x)).attr("y1", y(t.y)).attr("x2", x(a.x)).attr("y2", y(a.y)).attr("stroke", css("--line")).attr("stroke-width", 1.5);
  });
  const color = { animal: css("--animal"), ctrl: css("--twin"), pos: css("--orange") };
  const fmtX = (v) => (xk === "mean_chars" ? num(v, 0) : num(v, 1));
  F.g.selectAll("circle").data(pts).join("circle").attr("cx", (p) => x(p.x)).attr("cy", (p) => y(p.y)).attr("r", 6)
    .attr("fill", (p) => color[p.r]).attr("stroke", css("--surface")).attr("stroke-width", 2)
    .on("mousemove", (ev, p) => showTip(ev, `<b>${esc(p.id)}</b><br>${esc(PRED[xk])}: ${fmtX(p.x)}<br>${esc(OUT[yk])}: ${yfmt(p.y)}`))
    .on("mouseleave", hideTip);
  // label only the few highest points (selective direct labels)
  const top = pts.filter((p) => p.r !== "ctrl").sort((a, b) => b.y - a.y).slice(0, 5);
  const placed = [];
  top.forEach((p) => {
    let ty = y(p.y) - 9;
    while (placed.some((q) => Math.abs(q.y - ty) < 12 && Math.abs(q.x - x(p.x)) < 90)) ty -= 12;
    placed.push({ x: x(p.x), y: ty });
    const right = x(p.x) < F.iw - 90;
    F.g.append("text").attr("x", x(p.x) + (right ? 8 : -8)).attr("y", ty).attr("text-anchor", right ? "start" : "end").text(p.id);
  });
  const c = corr(xk, yk), tw = corr(xk, "beats_twin");
  $("#scan-stat").innerHTML = c ? `Spearman <b>ρ = ${num(c.spearman, 2)}</b> (permutation p = ${num(c.p_perm, 3)})` +
    (tw ? ` · animal beats its twin in <b>${esc(tw.wins)}</b> pairs (sign test p = ${num(tw.p_perm, 3)})` : "") : "";
}

function corrTable() {
  if (!Z.scan) return;
  const preds = Object.keys(PRED).filter((k) => corr(k, "collapse"));
  const cell = (k, o) => { const c = corr(k, o); if (!c || c.spearman == null) return "<td class='num'>n/a</td>";
    const strong = c.p_perm < 0.01; return `<td class="num">${strong ? "<b>" : ""}${num(c.spearman, 2)}${strong ? "</b>" : ""} <span class="muted">(${num(c.p_perm, 3)})</span></td>`; };
  $("#corr-table").innerHTML = `<table class="data"><tr><th>predictor</th><th class="num">collapse</th><th class="num">human EM</th><th class="num">judge EM</th><th class="num">beats twin</th></tr>` +
    preds.map((k) => { const tw = corr(k, "beats_twin");
      return `<tr><td>${esc(PRED[k])}</td>${cell(k, "collapse")}${cell(k, "human_em")}${cell(k, "judge_em")}<td class="num">${tw ? `${esc(tw.wins)} <span class="muted">(${num(tw.p_perm, 3)})</span>` : ""}</td></tr>`; }).join("") +
    `</table><p class="small muted" style="margin-top:6px">ρ (permutation p). Bold: p &lt; 0.01.</p>`;
}

function layerChart() {
  if (!Z.scan) return;
  const out = $("#layer-y").value;
  const rows = Z.scan.layers.filter((r) => r.outcome === out);
  const el = $("#layer-chart");
  const F = frame(el, 260, { t: 14, r: 76, b: 36, l: 40 });
  const x = d3.scaleLinear().domain([1, Z.scan.n_layers]).range([0, F.iw]);
  const ext = d3.extent(rows, (r) => r.spearman);
  const y = d3.scaleLinear().domain([Math.min(0, ext[0]), Math.max(0.2, ext[1])]).range([F.ih, 0]).nice();
  F.g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(5).tickSize(-F.iw).tickFormat("")).call((g) => g.select(".domain").remove());
  F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(7));
  F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
  F.g.append("text").attr("x", F.iw / 2).attr("y", F.ih + 30).attr("text-anchor", "middle").text("layer");
  F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", css("--muted"));
  const L = Z.scan.primary_layer;
  F.g.append("line").attr("x1", x(L)).attr("x2", x(L)).attr("y1", 0).attr("y2", F.ih).attr("stroke", css("--muted")).attr("stroke-dasharray", "4 4");
  F.g.append("text").attr("x", x(L) + 4).attr("y", 8).text(`primary layer ${L}`);
  const color = { evil: css("--animal"), impolite: css("--orange"), sycophantic: css("--aqua") };
  const ends = [];
  Object.entries(color).forEach(([trait, c]) => {
    const d = rows.filter((r) => r.trait === trait).sort((a, b) => a.layer - b.layer);
    if (!d.length) return;
    F.g.append("path").datum(d).attr("fill", "none").attr("stroke", c).attr("stroke-width", 2)
      .attr("d", d3.line().x((r) => x(r.layer)).y((r) => y(r.spearman)));
    let ty = y(d[d.length - 1].spearman);
    while (ends.some((e) => Math.abs(e - ty) < 12)) ty += 12;
    ends.push(ty);
    F.g.append("text").attr("x", F.iw + 6).attr("y", ty).attr("dy", "0.35em").text(trait);
  });
  const guide = F.g.append("line").attr("y1", 0).attr("y2", F.ih).attr("stroke", css("--ink-2")).attr("opacity", 0);
  F.g.append("rect").attr("width", F.iw).attr("height", F.ih).attr("fill", "transparent")
    .on("mousemove", (ev) => {
      const l = Math.max(1, Math.min(Z.scan.n_layers, Math.round(x.invert(d3.pointer(ev)[0]))));
      guide.attr("x1", x(l)).attr("x2", x(l)).attr("opacity", 0.5);
      const vals = Object.keys(color).map((t) => { const r = rows.find((q) => q.trait === t && q.layer === l); return r ? `${t}: ρ ${num(r.spearman, 2)}` : ""; });
      showTip(ev, `<b>layer ${l}</b><br>${vals.filter(Boolean).join("<br>")}`);
    })
    .on("mouseleave", () => { guide.attr("opacity", 0); hideTip(); });
}

main().catch((e) => {
  console.error(e);
  document.querySelector("main").insertAdjacentHTML("afterbegin",
    `<div class="wrap"><p class="note">Could not load the data files (data/zoo.json, data/samples.json). If you opened this file directly, serve the folder instead: <code>python -m http.server -d site</code>.</p></div>`);
});
