// NanoInspect Edge console. Plain JavaScript, no external libraries: the edge app must work offline.
const $ = s => document.querySelector(s);
const api = (p, o) => fetch(p, o).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x, d = 1) => x == null || isNaN(x) ? "–" : `${(x * 100).toFixed(d)}%`;
const num = x => x == null ? "–" : Number(x).toLocaleString();
const sec = x => x == null ? "–" : `${Number(x).toFixed(2)} s`;
const DEC = { accept: "ACCEPT", reject: "REJECT", manual_review: "HUMAN REVIEW" };
const TIER = { 1: "T1 · 7B fine-tuned", 2: "T2 · 27B + reference", 3: "T3 · cloud human" };
const BUCKET = {
  fast_accept: "Tier 1 confident: good", agree_good: "Both tiers: good", agree_defect: "Both tiers: defect",
  t1_flag_t2_good: "Tier 1 flags · tier 2 says good", t1_good_t2_defect: "Tier 1 good · tier 2 sees defect",
  t2_unusable: "Tier 2 answer unusable", input_ood: "Image unlike training data", unreadable: "Unreadable image"
};
let META = null, view = "home", devHist = [], lastInspection = null, chatHist = [];
const thumb = (p, size = 400) => `/api/thumb?path=${encodeURIComponent(p)}&size=${size}`;

// ---------------------------------------------------------------- routing: #page in the URL drives everything
const PAGES = {
  home: null,
  overview: ["Live overview", "Every decision made on this Nano, per tier, with device telemetry and savings.", "Inspection System", "screw/test/good/002.png"],
  inspect: ["Inspect a part", "Upload a photo or use the camera. Two local LLMs decide, explain and act.", "Inspection System", "bottle/test/broken_large/000.png"],
  line: ["Production line", "A simulated line through the full cascade, with the line controller's instructions.", "Inspection System", "zipper/test/good/002.png"],
  escalations: ["Cloud escalations", "Store-and-forward outbox: only crops of escalated parts leave the building.", "Inspection System", "cable/test/good/003.png"],
  how: ["How it works", "A cheap LLM, an expensive LLM, then a human, and why each exists.", "How It Works", "hazelnut/test/crack/003.png"],
  policy: ["Escalation policy & cost", "An explicit, measurable rule for when a person should look.", "How It Works", "metal_nut/test/good/002.png"],
  sop: ["SOP & machine instructions", "ISO 9001 §8.7 procedures turned into instructions the line can execute.", "How It Works", "transistor/test/good/002.png"],
  results: ["Fine-tuning & results", "What training on the Nano changed, measured against every baseline.", "Evidence", "hazelnut/test/crack/003.png"],
  models: ["Models & serving", "Quality against baselines and serving performance, measured on the Nano.", "Evidence", "screw/test/thread_side/008.png"],
  savings: ["Cloud vs edge savings", "Tokens, API cost, energy and time: what running locally saves.", "Evidence", "pill/test/good/002.png"],
  evidence: ["Benchmarks", "Soak test, robustness, escalation policy and line simulation.", "Evidence", "tile/test/good/002.png"],
};
const SECTION_HOME = { "Inspection System": "overview", "How It Works": "how", "Evidence": "evidence" };
function route() {
  const v = (location.hash.slice(1) || "home"); show(PAGES.hasOwnProperty(v) ? v : "home");
}
window.addEventListener("hashchange", route);
function go(v) { if (location.hash === "#" + v) route(); else location.hash = v; }
function show(v) {
  view = v;
  document.querySelectorAll(".view").forEach(s => s.hidden = s.id !== `v-${v}`);
  const P = PAGES[v];
  $("#banner").hidden = $("#crumbs").hidden = !P;
  if (P) {
    $("#banner-title").textContent = P[0]; $("#banner-sub").textContent = P[1];
    $("#banner-bg").style.backgroundImage = `url(${thumb(P[3], 900)})`;
    const sec = SECTION_HOME[P[2]];
    $("#crumbs").innerHTML = `<a href="#home">Home</a> » <a href="#${sec}">${P[2]}</a> » ${esc(P[0])}`;
  }
  document.querySelectorAll(".mi").forEach(m => { m.classList.remove("open");
    m.querySelector(":scope > a").classList.toggle("on", !!P && P[2] === m.querySelector(":scope > a").textContent.trim()); });
  window.scrollTo(0, 0);
  if (v === "home") home();
  if (v === "policy") loadPolicy();
  if (v === "evidence") loadEvidence();
  if (v === "models") loadModels();
  if (v === "how") how();
  if (v === "sop") sop();
  if (v === "savings") savings();
  if (v === "results") results();
  tick();
}

// ---------------------------------------------------------------- header + overview
async function tick() {
  try {
    const o = await api("/api/overview");
    header(o);
    if (view === "overview") overview(o);
    if (view === "line") line();
    if (view === "escalations") escalations(o);
    if (view === "models") loadModels(true);
    if (view === "savings") savings();
    if (view === "home") homeLive(o);
  } catch (e) { $("#st-sync").innerHTML = `<span class="dot bad"></span>Edge API unreachable`; }
}
function header(o) {
  const d = o.device || {}, s = o.sync;
  devHist.push({ t: Date.now(), temp: d.temp_c, w: d.power_w, u: d.gpu_util_pct }); devHist = devHist.slice(-120);
  $("#dev").innerHTML = `HP ZGX Nano · GB10 · ${d.gpu_util_pct ?? "–"}% GPU · ${d.temp_c ?? "–"} °C · ${d.power_w ?? "–"} W`;
  const t1 = META?.tier1?.healthy, t2 = META?.tier2?.healthy;
  $("#st-vlm").innerHTML = `<span class="dot ${t1 ? "ok" : "bad"}"></span>T1 7B+LoRA <span class="dot ${t2 ? "ok" : "bad"}" style="margin-left:6px"></span>T2 27B · vLLM on this device`;
  const q = o.stats.escalations.queued || 0;
  $("#st-sync").innerHTML = !s.online ? `<span class="dot bad"></span>Offline · ${q} queued locally` :
    s.reachable ? `<span class="dot ok"></span>Cloud sync OK${q ? ` · ${q} sending` : ""}` : `<span class="dot warn"></span>Cloud unreachable · ${q} queued`;
  $("#net").checked = s.online; $("#net-label").textContent = s.online ? "Network online" : "Network OFF (simulated)";
}
$("#net").onchange = e => post("/api/network", { online: e.target.checked }).then(tick);

function overview(o) {
  const st = o.stats, t = st.by_tier, total = st.total || 0, dec = st.by_decision, auto = (dec.accept || 0) + (dec.reject || 0);
  const kept = st.bytes_inspected ? 1 - st.bytes_sent_to_cloud / st.bytes_inspected : null;
  $("#kpis").innerHTML = [
    ["Parts inspected", num(total), o.line.running ? `line running · ${o.line.decided_per_min} decided/min` : "on this device"],
    ["Decided on the edge", pct(total ? auto / total : null), `${num(dec.accept || 0)} accepted · ${num(dec.reject || 0)} rejected`],
    ["Sent to a human", pct(total ? (dec.manual_review || 0) / total : null, 2), `${num(dec.manual_review || 0)} parts`],
    ["Tier-1 latency p95", st.vision_p95_ms ? `${(st.vision_p95_ms / 1000).toFixed(2)} s` : "–", "7B + LoRA, every part"],
    ["Image data kept on-site", kept == null ? "–" : pct(kept, 2), `${(st.bytes_sent_to_cloud / 1024).toFixed(0)} KB sent of ${(st.bytes_inspected / 1e6).toFixed(0)} MB`],
  ].map(([k, v, s]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  savings();
  const t2 = (t["2"] || 0) + (t["3"] || 0), t3 = t["3"] || 0, w = x => total ? Math.max(0.4, 100 * x / total) : 0;
  $("#funnel").innerHTML = [
    ["Tier 1 · Qwen2.5-VL-7B + LoRA", "fine-tuned on this Nano · every part", total, "var(--t1)"],
    ["Tier 2 · Qwen3.8-27B + LoRA", "compares with a known-good reference · uncertain + audit", t2, "var(--t2)"],
    ["Tier 3 · cloud human", "only when review is cheaper than the risk", t3, "var(--t3)"],
  ].map(([n, s, c, col]) => `<div class="tier"><div class="name"><b>${n}</b><span>${s}</span></div>
      <div class="barwrap"><div class="bar" style="width:${w(c)}%;background:${col}"></div></div>
      <div class="num">${num(c)}<span>${pct(total ? c / total : null)} of parts</span></div></div>`).join("") +
    `<div class="muted small">Escalations: ${num(st.escalations.queued || 0)} queued · ${num(st.escalations.sent || 0)} awaiting review · ${num(st.escalations.reviewed || 0)} labelled${st.human_vlm_agreement != null ? ` · tier 2 agreed with the human ${pct(st.human_vlm_agreement, 0)}` : ""}</div>`;
  $("#devpanel").innerHTML = spark(devHist.map(h => h.u), "GPU utilization %", 100) + spark(devHist.map(h => h.temp), "Temperature °C", 100) +
    spark(devHist.map(h => h.w), "GPU power W", 120) + `<div class="muted small">Sampled from nvidia-smi on the Nano. Tier 2 p50: ${sec(st.vlm_p50_s)}.</div>`;
  api("/api/recent?n=24").then(rs => $("#feed").innerHTML = rs.map(partTile).join("") || `<div class="muted">No parts yet. Start the demo line or inspect a part.</div>`);
}
function partTile(r) {
  const img = r.image_ref && r.image_ref.includes("/") ? `/api/thumb?path=${encodeURIComponent(r.image_ref)}` : "";
  return `<div class="part ${r.decision}" title="${esc(r.reason)} (click to inspect again)" ${img ? `data-reinspect="${esc(r.image_ref)}" data-cat="${esc(r.category)}"` : ""}>${img ? `<img src="${img}" loading="lazy">` : `<div style="aspect-ratio:1"></div>`}
    <span class="t t${r.tier}">T${r.tier}</span><div class="lbl"><b>${esc(r.category)}</b><span>${DEC[r.decision]}</span></div></div>`;
}
function spark(vals, label, max) {
  const v = vals.filter(x => x != null && !isNaN(x)); const W = 320, H = 46;
  if (v.length < 2) return `<div class="small muted">${label}: –</div>`;
  const m = Math.max(max, ...v), pts = v.map((y, i) => `${(i / (v.length - 1)) * W},${H - (y / m) * (H - 4) - 2}`).join(" ");
  return `<div class="small"><b>${label}</b> <span class="muted">${v[v.length - 1].toFixed(0)}</span></div>
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2"/></svg>`;
}


// ---------------------------------------------------------------- cloud vs edge savings
const usd = (x, d = 2) => x == null ? "–" : `$${Number(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}`;
const dur = s => s == null ? "–" : s < 120 ? `${s.toFixed(1)} s` : s < 7200 ? `${(s / 60).toFixed(1)} min` : `${(s / 3600).toFixed(1)} h`;
let ratesLoaded = false;
async function savings() {
  const S = await api("/api/savings"), L = S.live, P = S.per_day_projection, R = S.rates;
  if (!ratesLoaded) { $("#r-in").value = R.input_per_m_usd; $("#r-out").value = R.output_per_m_usd; $("#r-kwh").value = R.electricity_per_kwh_usd;
    $("#r-up").value = R.uplink_mbps; $("#r-rtt").value = R.cloud_rtt_s; $("#r-ppm").value = R.parts_per_minute; $("#r-h").value = R.hours_per_day; ratesLoaded = true; }
  $("#rates-note").textContent = `Cloud rate: ${R.cloud_model}: $${R.input_per_m_usd}/1M input + $${R.output_per_m_usd}/1M output tokens, applied to the tokens our models actually processed. ` +
    `Energy: measured GPU joules per inference on this Nano (the daily projection also includes measured idle GPU power for the whole shift). Network time: image upload at ${R.uplink_mbps} Mbit/s + ${R.cloud_rtt_s} s round-trip per API call (assumptions).`;
  const tip = x => `<span class="help">?<span class="tip">
      <div class="l"><b>Net compute savings</b><b>${usd(x.net_saved_usd, 4)}</b></div>
      <div class="muted">cloud-equivalent ${usd(x.cloud_equivalent_usd, 4)} − local electricity ${usd(x.electricity_usd, 6)}</div><hr>
      <div class="l"><span>Input tokens ${num(x.tokens_in)} × $${R.input_per_m_usd}/1M</span><b>${usd(x.cloud_input_usd, 4)}</b></div>
      <div class="l"><span>Output tokens ${num(x.tokens_out)} × $${R.output_per_m_usd}/1M</span><b>${usd(x.cloud_output_usd, 4)}</b></div>
      <div class="l"><span>Blended rate</span><b>${x.blended_rate_per_m_usd ? "$" + x.blended_rate_per_m_usd.toFixed(2) + " / 1M" : "–"}</b></div>
      <div class="l"><span>Energy ${(x.energy_kwh * 1000).toFixed(3)} Wh × $${R.electricity_per_kwh_usd}/kWh</span><b>${usd(x.electricity_usd, 6)}</b></div>
      <div class="muted">Cloud API calls avoided: ${num(x.llm_calls_avoided)}. Formula as in the ZGX Console, with our exact input/output split instead of a 70/30 blend.</div></span></span>`;
  const kp = $("#kpis");
  if (kp && !kp.querySelector(".save")) kp.insertAdjacentHTML("beforeend", `<div class="kpi save" id="kpi-save"></div>`);
  const ks = $("#kpi-save");
  if (ks) ks.innerHTML = `<div class="k">Net compute savings ${tip(L)}</div><div class="v">${usd(L.net_saved_usd, 4)}</div><div class="s">${num(L.tokens_total)} tokens recorded · ${num(L.llm_calls_avoided)} API calls avoided</div>`;
  const rows = (x, time) => `<table class="stable">
      <tr><td>Parts</td><td>${num(Math.round(x.parts))}</td></tr>
      <tr><td>Cloud API calls avoided</td><td>${num(Math.round(x.llm_calls_avoided))}</td></tr>
      <tr><td>Tokens processed locally (input / output)</td><td>${num(Math.round(x.tokens_in))} / ${num(Math.round(x.tokens_out))}</td></tr>
      <tr><td>Cloud-equivalent API cost</td><td>${usd(x.cloud_equivalent_usd)}</td></tr>
      <tr><td>Local electricity (measured J/inference)</td><td>${usd(x.electricity_usd, 4)}</td></tr>
      <tr><td><b>Net saved</b></td><td style="color:var(--good)">${usd(x.net_saved_usd)}</td></tr>
      <tr><td>Image data kept on-site</td><td>${x.image_mb_kept_local >= 1000 ? (x.image_mb_kept_local / 1000).toFixed(1) + " GB" : x.image_mb_kept_local.toFixed(1) + " MB"}</td></tr>
      <tr><td>Network time avoided (upload + round-trips)</td><td>${dur(x.network_time_avoided_s)}</td></tr>
      ${time ? `<tr><td>Local GPU time used</td><td>${dur(x.local_gpu_s)}</td></tr>` : ""}</table>`;
  $("#savings-live").innerHTML = `<b>So far on this device</b>` + rows(L, true);
  $("#savings-proj").innerHTML = P && !P.error ? `<b>Projected per day</b> <span class="muted small">(${R.parts_per_minute} parts/min × ${R.hours_per_day} h; measured tokens per call: tier 1 ${P.per_call_tokens.tier1_in}+${P.per_call_tokens.tier1_out}, tier 2 ${P.per_call_tokens.tier2_in}+${P.per_call_tokens.tier2_out}; tier 2 on ${(P.tier2_call_rate * 100).toFixed(1)}% of parts)</span>` + rows(P, false) +
    `<div class="muted small">Per year (×365): <b>${usd(P.net_saved_usd * 365, 0)}</b> of cloud API cost avoided, ${num(Math.round(P.llm_calls_avoided * 365))} API calls.</div>` : `<div class="muted">Projection needs the notebook results.</div>`;
}
$("#r-apply").onclick = async () => { await post("/api/savings/rates", { input_per_m_usd: +$("#r-in").value, output_per_m_usd: +$("#r-out").value,
  electricity_per_kwh_usd: +$("#r-kwh").value, uplink_mbps: +$("#r-up").value, cloud_rtt_s: +$("#r-rtt").value, parts_per_minute: +$("#r-ppm").value, hours_per_day: +$("#r-h").value }); savings(); };

// ---------------------------------------------------------------- inspect
const drop = $("#drop"), file = $("#file");
$("#browse").onclick = () => file.click();
file.onchange = () => file.files[0] && stage(file.files[0], "upload");
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && stage(e.dataTransfer.files[0], "upload"); };
let staged = null;
function stage(f, source) {
  staged = { f, source };
  $("#pending-img").src = URL.createObjectURL(f); $("#pending-name").textContent = f.name;
  $("#pending-meta").textContent = `${(f.size / 1024).toFixed(0)} KB · product: ${$("#cat").value} · nothing sent yet`;
  $("#pending").hidden = false; file.value = "";
}
$("#submit").onclick = () => { if (!staged) return; const { f, source } = staged; staged = null; $("#pending").hidden = true; inspectFile(f, source); };
$("#clear").onclick = () => { staged = null; $("#pending").hidden = true; };
$("#cat").onchange = () => { if (staged) $("#pending-meta").textContent = `${(staged.f.size / 1024).toFixed(0)} KB · product: ${$("#cat").value} · nothing sent yet`; };
async function inspectFile(f, source) {
  busy();
  const fd = new FormData(); fd.append("file", f); fd.append("category", $("#cat").value); fd.append("force_t2", $("#force").checked); fd.append("source", source);
  renderResult(await api("/api/inspect", { method: "POST", body: fd }));
}
$("#sample-good").onclick = () => sample("good");
$("#sample-bad").onclick = () => sample("defect");
async function sample(kind) { busy(); renderResult(await post("/api/inspect_sample", { category: $("#cat").value, kind, force_t2: $("#force").checked })); }
function busy() { $("#result").innerHTML = `<div class="placeholder">Inspecting on the Nano…</div>`; $("#chat").hidden = true; }

let stream = null;
$("#cam-on").onclick = async () => {
  try { stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 960 } }); $("#video").srcObject = stream; $("#cam").hidden = false; }
  catch (e) { $("#result").innerHTML = `<div class="placeholder">Camera unavailable (${esc(e.message)}). Browsers allow the camera only on https or localhost: open the app through an SSH tunnel (http://localhost:8080), or upload a photo.</div>`; }
};
$("#cam-off").onclick = () => { stream?.getTracks().forEach(t => t.stop()); $("#cam").hidden = true; };
$("#snap").onclick = () => {
  const v = $("#video"), c = document.createElement("canvas"), s = Math.min(v.videoWidth, v.videoHeight);
  c.width = c.height = s; c.getContext("2d").drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, s, s);
  c.toBlob(b => stage(new File([b], `camera-${new Date().toLocaleTimeString()}.jpg`, { type: "image/jpeg" }), "camera"), "image/jpeg", 0.92);
};

function tierLine(r, n) {
  if (!r) return `<span class="muted">${n === 2 ? "not needed (tier 1 was confident; accepted parts are re-checked in a random audit)" : "–"}</span>`;
  const f = r.verdict === "defective" && r.defect_type && r.defect_type !== "none" ? ` · ${esc(r.defect_type.replaceAll("_", " "))} at the ${esc(r.location)}` : "";
  return `<b>${esc(r.verdict ?? "unusable")}</b>${f} <span class="muted small">P(defect) ${pct(r.p_defective)} · ${sec(r.latency_s)}</span>` +
    (r.explanation ? `<div class="small">“${esc(r.explanation)}”${r.explanation_by ? ` <span class="muted">(verdict: fine-tuned 27B · sentence: untrained 27B)</span>` : ""}</div>` : "");
}
function renderResult(r) {
  lastInspection = r.inspection_id; chatHist = []; $("#chat-log").innerHTML = "";
  if (!r.marked_b64) { $("#result").innerHTML = `<div class="verdict manual_review"><div class="d">HUMAN REVIEW</div></div><div class="why">${esc(r.reason)}</div>`; return; }
  const path = [1, 2, 3].map(t => `<span class="chip ${r.tier >= t ? "on t" + t : ""}">${TIER[t]}</span>`).join("");
  const gt = r.ground_truth ? `<div>Ground truth (dataset)</div><div><b>${esc(r.ground_truth.label)}</b>${r.ground_truth.label === "defective" ? ` · ${esc(r.ground_truth.defect_type)} @ ${esc(r.ground_truth.location)}` : ""}</div>` : "";
  $("#result").innerHTML = `
    <div class="verdict ${r.decision}"><div><div class="d">${DEC[r.decision]}</div><div class="small">${esc(r.recommended_action)}</div></div><div class="path">${path}</div></div>
    <div class="refs four"><figure><img src="${r.image_b64}"><figcaption>Part (${esc(r.category)})</figcaption></figure>
      <figure><img src="${r.delta_b64}"><figcaption>Delta vs 40 good parts (patch distance)</figcaption></figure>
      <figure><img src="${r.marked_b64}"><figcaption>Region the LLM reported</figcaption></figure>
      <figure><img src="${r.reference_b64}"><figcaption>Known-good reference (tier 2)</figcaption></figure></div>
    <div class="kv">
      <div>Tier 1 · 7B fine-tuned</div><div>${tierLine(r.tier1, 1)}</div>
      <div>Tier 2 · 27B</div><div>${tierLine(r.tier2, 2)}</div>
      <div>Delta vs good parts</div><div>score ${r.delta?.score} · <b>${r.delta?.z_vs_good > 0 ? "+" : ""}${r.delta?.z_vs_good} σ</b> from held-out good parts <span class="muted small">(training-free, per-patch feature distance)</span></div>
      <div>Evidence bucket</div><div>${esc(BUCKET[r.bucket] || r.bucket)}${r.audit ? " · <b>random audit</b>" : ""}</div>
      ${gt}
      <div>Time on the Nano</div><div>${(r.total_s).toFixed(2)} s total</div>
      ${r.escalation_id ? `<div>Cloud escalation</div><div>queued as <code>${r.escalation_id}</code> (region crop only)</div>` : ""}
    </div>
    <div class="why"><b>Why:</b> ${esc(r.reason)}</div>
    ${r.machine_json ? `<details class="mj" ${r.decision === "reject" ? "open" : ""}><summary><b>Machine instruction sent to the line controller</b>
      <span class="pill ${r.decision}">${esc(r.machine_json.line_command.action)}</span> <span class="muted small">SOP ${esc(r.machine_json.sop.sop_id)} v${esc(r.machine_json.sop.version)} · report text by ${esc(r.machine_json_source)} · schema-validated</span></summary>
      <pre>${esc(JSON.stringify(r.machine_json, null, 2))}</pre></details>` : ""}`;
  $("#chat").hidden = !META?.tier2?.healthy;
}
async function ask() {
  const q = $("#chat-q").value.trim(); if (!q || lastInspection == null) return;
  $("#chat-q").value = ""; chatHist.push({ role: "user", content: q });
  $("#chat-log").innerHTML += `<div class="msg user">${esc(q)}</div><div class="msg bot" id="pending">…</div>`;
  try {
    const a = await post("/api/chat", { inspection_id: lastInspection, question: q, history: chatHist.slice(0, -1) });
    chatHist.push({ role: "assistant", content: a.answer });
    $("#pending").outerHTML = `<div class="msg bot">${esc(a.answer)}<div class="small muted">${a.latency_s} s · ${a.tokens} tokens · on the Nano</div></div>`;
  } catch (e) { $("#pending").outerHTML = `<div class="msg bot">Tier 2 unavailable (${esc(e.message)}).</div>`; }
  $("#chat-log").scrollTop = 1e9;
}
$("#chat-send").onclick = ask;
$("#chat-q").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); ask(); } });

// ---------------------------------------------------------------- production line
["ppm", "dr"].forEach(id => $("#" + id).oninput = e => $(`#${id}-v`).textContent = e.target.value);
$("#line-start").onclick = startLine;
$("#line-stop").onclick = () => post("/api/line/stop", {}).then(line);
function startLine() { return post("/api/line/start", { parts_per_minute: +$("#ppm").value, defect_rate: +$("#dr").value / 100 }).then(line); }
async function line() {
  const L = await api("/api/line"), s = L.snapshot;
  $("#line-kpis").innerHTML = [
    ["Status", s.running ? "Running" : "Stopped"], ["Parts arrived", num(s.parts)], ["Decided / min", s.decided_per_min],
    ["Accepted", num(s.accepted)], ["Rejected", num(s.rejected)], ["Human review", num(s.manual_review)],
    ["Sent to tier 2", pct(s.share_sent_to_t2)], ["Waiting T1 / T2", `${s.waiting_tier1} / ${s.waiting_tier2}`],
    ["Tier 1 p95", sec(s.tier1_p95_s)], ["Tier 2 p95", sec(s.tier2_p95_s)], ["Decision p95", sec(s.decision_p95_s)],
    ["Escapes (ground truth)", s.escapes], ["Audit alerts", s.audit_alerts],
  ].map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  $("#line-chart").innerHTML = lineChart(L.timeline);
  const f = L.last_flagged;
  $("#line-last").innerHTML = f && f.path ? `<div style="display:grid;grid-template-columns:160px 1fr;gap:12px">
      <img src="/api/thumb?path=${encodeURIComponent(f.path)}&size=240" style="width:100%;border-radius:8px">
      <div class="kv" style="grid-template-columns:90px 1fr"><div>Product</div><div>${esc(f.category)}</div>
        <div>Tier 1</div><div>${esc(f.t1?.verdict)} · P(defect) ${pct(f.p1)}</div>
        <div>Tier 2</div><div>${esc(f.t2?.verdict)} · ${esc(f.t2?.defect_type)} @ ${esc(f.t2?.location)}</div>
        <div>Decision</div><div><span class="pill ${f.decision}">${DEC[f.decision]}</span></div></div></div>
      ${f.t2?.explanation ? `<div class="why">“${esc(f.t2.explanation)}”</div>` : ""}<div class="why">${esc(f.reason)}</div>` : `<div class="muted">Nothing sent to tier 2 yet.</div>`;
  $("#stream").innerHTML = L.events.map(e => partTile({ ...e, image_ref: e.path })).join("");
  const C = await api("/api/controller?n=12");
  $("#controller").innerHTML = (C.stopped ? `<div class="verdict reject"><div><div class="d">LINE STOP</div><div class="small">${esc(C.stopped.defect_type)} on ${esc(C.stopped.product)} repeated ${C.stopped.hits}× within ${C.stopped.window} parts (SOP containment rule) at ${C.stopped.time}</div></div><button id="ctl-reset">Acknowledge</button></div>` : "") +
    `<div class="table-wrap"><table><tr><th>Time</th><th>Part</th><th>Product</th><th>Command</th><th>Disposition</th><th>NCR</th><th>Probable cause</th></tr>` +
    C.messages.map(m => `<tr><td>${esc(m.timestamp.slice(11))}</td><td>${esc(m.part_id)}</td><td>${esc(m.product)}</td><td><code>${esc(m.line_command.action)}</code></td>
      <td>${esc(m.disposition)}</td><td>${esc(m.nonconformance?.ncr_id || "")}</td><td>${esc(m.nonconformance?.probable_cause || "")}</td></tr>`).join("") + `</table></div>`;
  const b = document.getElementById("ctl-reset"); if (b) b.onclick = () => post("/api/controller/reset", {}).then(line);
}
function lineChart(tl) {
  if (!tl || tl.length < 3) return `<div class="muted">Start the line to see throughput.</div>`;
  const W = 640, H = 200, P = 28;
  const rate = tl.map((p, i) => i < 5 ? null : (p.decided - tl[i - 5].decided) / ((p.t_s - tl[i - 5].t_s) || 1) * 60);
  const q1 = tl.map(p => p.queue_t1), q2 = tl.map(p => p.queue_t2);
  const maxR = Math.max(30, ...rate.filter(x => x != null)), maxQ = Math.max(10, ...q1, ...q2);
  const x = i => P + (i / (tl.length - 1)) * (W - 2 * P);
  const path = (arr, m) => arr.map((y, i) => y == null ? "" : `${x(i)},${H - P - (y / m) * (H - 2 * P)}`).filter(Boolean).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%">
    <line x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}" stroke="#e5e7eb"/>
    <polyline points="${path(rate, maxR)}" fill="none" stroke="var(--t1)" stroke-width="2.5"/>
    <polyline points="${path(q1, maxQ)}" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="2 3"/>
    <polyline points="${path(q2, maxQ)}" fill="none" stroke="var(--t2)" stroke-width="2.5" stroke-dasharray="5 4"/>
    <text x="${P}" y="14" style="fill:var(--t1)">— decided/min (max ${maxR.toFixed(0)})</text>
    <text x="${P + 190}" y="14" style="fill:#64748b">··· waiting T1</text><text x="${P + 290}" y="14" style="fill:var(--t2)">- - waiting T2 (max ${maxQ})</text>
    <text x="${W - P}" y="${H - 8}" text-anchor="end">${tl[tl.length - 1].t_s.toFixed(0)} s</text></svg>`;
}

// ---------------------------------------------------------------- escalations
async function escalations(o) {
  const E = await api("/api/escalations"), st = o.stats, e = st.escalations, s = E.sync;
  $("#outbox").innerHTML = `<div class="kpis small">
      <div class="kpi"><div class="k">Queued on the Nano</div><div class="v">${e.queued || 0}</div></div>
      <div class="kpi"><div class="k">Delivered, awaiting review</div><div class="v">${e.sent || 0}</div></div>
      <div class="kpi"><div class="k">Labelled by a human</div><div class="v">${e.reviewed || 0}</div></div></div>
    <div class="muted small">${s.online ? (s.reachable ? `Cloud reachable at ${esc(s.cloud_url)}.` : `Trying ${esc(s.cloud_url)}: ${esc(s.last_error || "…")}`) :
      "Network is off. Inspection continues and escalations wait in the local outbox; they are sent when the link returns."}
      Human labels flow back and become training data for the next on-device LoRA fine-tune (${st.human_labels} so far).</div>`;
  const full = st.bytes_inspected, sent = st.bytes_sent_to_cloud;
  $("#privacy").innerHTML = `<div class="bars">
      <div class="b"><span>Cloud-only design: every image uploaded</span><div class="track"><div class="fill" style="width:100%"></div></div><b>${(full / 1e6).toFixed(1)} MB</b></div>
      <div class="b me"><span>NanoInspect: escalated region crops only</span><div class="track"><div class="fill" style="width:${full ? Math.max(0.5, 100 * sent / full) : 0}%"></div></div><b>${(sent / 1024).toFixed(0)} KB</b></div></div>
    <div class="muted small">Only a small crop of the region the models point at is sent, for escalated parts only. Product images and line data stay on the device.</div>`;
  $("#cloud-link").href = (META?.cloud_url || "").replace("127.0.0.1", location.hostname);
  $("#esc-table").innerHTML = `<tr><th>Crop sent</th><th>Product</th><th>Why escalated</th><th>T1 P(defect)</th><th>Tier 2</th><th>Status</th><th>Human verdict</th><th>Size</th><th>Created</th></tr>` +
    E.items.map(i => `<tr><td><img src="/api/escalations/${i.id}/roi"></td><td><b>${esc(i.category)}</b><div class="small muted">${esc(i.source)}</div></td>
      <td>${esc(BUCKET[i.bucket] || i.bucket)}</td><td>${i.vision_score == null ? "–" : pct(i.vision_score)}</td><td>${esc(i.vlm_verdict ?? "–")}${i.vlm_type && i.vlm_type !== "none" ? ` · ${esc(i.vlm_type)}` : ""}</td>
      <td><span class="pill ${i.status}">${i.status}</span>${i.attempts > 1 ? `<div class="small muted">${i.attempts} attempts</div>` : ""}</td>
      <td>${i.human_verdict ? `<span class="pill ${i.human_verdict}">${i.human_verdict}</span> ${esc(i.human_defect_type || "")}<div class="small muted">${esc(i.reviewer)}</div>` : "–"}</td>
      <td>${(i.bytes / 1024).toFixed(1)} KB</td><td>${new Date(i.created * 1000).toLocaleTimeString()}</td></tr>`).join("");
}

// ---------------------------------------------------------------- models & serving
async function loadModels(live) {
  const m = await api("/api/models");
  $("#model-cards").innerHTML = [["tier1", "Tier 1 · cheap", "var(--t1)"], ["tier2", "Tier 2 · expensive", "var(--t2)"]].map(([k, label, col]) => {
    const t = m[k], x = t.metrics || {};
    return `<div class="card tiercard" style="border-top:4px solid ${col}"><h3>${label}</h3><h4>${esc(t.model)}</h4>
      <div class="sub">vLLM on this Nano · ${esc(t.url)} · served as <code>${esc(t.served_as)}</code> · <span class="dot ${t.healthy ? "ok" : "bad"}"></span> ${t.healthy ? "healthy" : "down"}</div>
      <div class="mgrid">
        <div>Running requests<b>${x.requests_running ?? "–"}</b></div><div>Waiting<b>${x.requests_waiting ?? "–"}</b></div>
        <div>KV-cache use<b>${x.kv_cache_usage == null ? "–" : pct(x.kv_cache_usage)}</b></div>
        <div>Mean time to first token<b>${sec(x.mean_ttft_s)}</b></div><div>Mean request latency<b>${sec(x.mean_e2e_s)}</b></div>
        <div>Requests served<b>${num(x.requests_finished)}</b></div><div>Prefix-cache hit rate<b>${pct(x.prefix_cache_hit_rate)}</b></div>
        <div>Prompt tokens<b>${num(x.prompt_tokens_total)}</b></div><div>Generated tokens<b>${num(x.generation_tokens_total)}</b></div>
        <div>Mean time per output token<b>${x.mean_tpot_s == null ? "–" : (x.mean_tpot_s * 1000).toFixed(1) + " ms"}</b></div>
      </div><div class="muted small" style="margin-top:6px">Live from vLLM's /metrics endpoint.</div></div>`;
  }).join("");
  if (live) return;
  const q = m.quality || [];
  $("#quality").innerHTML = q.length ? `<tr><th>Model / configuration</th><th>ROC-AUC</th><th>Accuracy</th><th>Defect recall</th><th>False-positive rate</th><th>Defect type correct</th><th>Location correct</th><th>Valid JSON</th><th>Throughput</th></tr>` +
    q.map(r => `<tr${r.config.startsWith("Tier") ? ' style="font-weight:600"' : ""}><td>${esc(r.config)}</td><td>${r.roc_auc?.toFixed(3) ?? "–"}</td><td>${pct(r.accuracy)}</td><td>${pct(r.recall)}</td>
      <td>${pct(r.false_positive_rate)}</td><td>${pct(r.defect_type_acc)}</td><td>${pct(r.location_acc)}</td><td>${pct(r.valid_json_rate)}</td>
      <td>${r.throughput_img_per_s ? r.throughput_img_per_s.toFixed(1) + " img/s" : "–"}</td></tr>`).join("") : `<tr><td>Run the notebook to produce model-quality results.</td></tr>`;
  const b = [...(m.tier1.benchmark || []).map(r => ({ tier: "Tier 1 · 7B + LoRA", ...r })), ...(m.tier2.benchmark || []).map(r => ({ tier: "Tier 2 · 27B + LoRA", ...r }))];
  $("#servebench").innerHTML = b.length ? `<tr><th>Model</th><th>Concurrent clients</th><th>Requests/s</th><th>Output tokens/s</th><th>p50 latency</th><th>p95 latency</th></tr>` +
    b.map(r => `<tr><td>${r.tier}</td><td>${r.concurrency}</td><td>${r.requests_per_s}</td><td>${r.output_tokens_per_s}</td><td>${sec(r.p50_latency_s)}</td><td>${sec(r.p95_latency_s)}</td></tr>`).join("") :
    `<tr><td>Run the notebook's serving benchmark to fill this table.</td></tr>`;
}

// ---------------------------------------------------------------- policy
async function loadPolicy() {
  const p = await api("/api/policy");
  $("#c-esc").value = p.costs.escape_usd; $("#c-fr").value = p.costs.false_reject_usd; $("#c-hr").value = p.costs.human_review_usd;
  $("#c-mode").value = p.mode || "throughput";
  $("#c-dr").value = +(p.defect_rate * 100).toFixed(2); $("#c-ar").value = +(p.audit_rate * 100).toFixed(1); $("#c-pv").value = p.privacy;
  renderPolicy(p, await api("/api/policy/simulate"));
}
$("#apply").onclick = async () => {
  const p = await post("/api/policy", { mode: $("#c-mode").value, escape_usd: +$("#c-esc").value, false_reject_usd: +$("#c-fr").value, human_review_usd: +$("#c-hr").value,
    defect_rate: +$("#c-dr").value / 100, audit_rate: +$("#c-ar").value / 100, privacy: $("#c-pv").value });
  renderPolicy(p, p.simulation);
};
function renderPolicy(p, sim) {
  if (!p.fitted) $("#pol-table").innerHTML = `<tr><td>No fitted policy yet (run the notebook). Until then a part is decided automatically only when both tiers agree.</td></tr>`;
  else $("#pol-table").innerHTML = `<tr><th>Evidence bucket</th><th>P(defect)</th><th>Expected cost if accepted</th><th>Expected cost if rejected</th><th>Human review</th><th>Action</th><th>Measured on</th></tr>` +
    p.table.map(r => `<tr><td><b>${esc(BUCKET[r.bucket] || r.bucket)}</b><div class="small muted">${esc(r.meaning)}</div></td><td>${pct(r.p_defect, 2)}</td>
      <td>$${r.expected_cost_accept_usd.toFixed(3)}</td><td>$${r.expected_cost_reject_usd.toFixed(3)}</td><td>$${r.human_review_usd.toFixed(2)}</td>
      <td><span class="pill ${r.action}">${r.action === "escalate_to_human" ? "escalate to human" : r.action}</span>${r.bucket === "fast_accept" ? `<div class="small muted">tier 1 always accepts; this is the escape risk</div>` : ""}</td>
      <td class="small muted">${r.evidence_defect_parts} defect / ${r.evidence_good_parts} good parts</td></tr>`).join("") +
    (() => { const c = p.costs, rej = 1 - c.human_review_usd / c.false_reject_usd, acc = c.human_review_usd / c.escape_usd;
      return `<tr><td colspan="7" class="small"><b>Break-even:</b> reject automatically only when P(defect) ≥ 1 − review/scrap = <b>${rej > 0 ? pct(rej, 0) : "any (a review costs at least as much as scrapping the part)"}</b>;
        accept automatically only when P(defect) ≤ review/escape = <b>${pct(acc, 2)}</b>; anything in between goes to a human.
        Raising the escape cost moves only the accept line, so it can never turn an escalation into a reject.</td></tr>`; })();
  if (p.modes) $("#pol-table").insertAdjacentHTML("beforeend", `<tr><td colspan="7" class="small"><b>Routing modes (measured, default costs):</b> ` +
      Object.entries(p.modes).map(([m, x]) => `${m} mode: tier-1 threshold ${x.t_lo?.toFixed(3)}, $${x.cascade?.cost_per_1000_usd?.toFixed(0)} per 1,000 parts, ${x.cascade?.vlm_calls_per_1000?.toFixed(0)} tier-2 calls, ${x.cascade?.human_reviews_per_1000?.toFixed(0)} human reviews`).join(" · ") +
      `. Capacity mode needs tier 2 to keep up (about 52% of parts; fits lines up to ~50 parts/min).</td></tr>`);
  if (!sim) { $("#pol-sim").innerHTML = `<div class="muted">No held-out simulation available yet.</div>`; return; }
  const max = Math.max(...Object.values(sim).map(v => v.cost_per_1000_usd));
  $("#pol-sim").innerHTML = `<div class="bars">` + Object.entries(sim).map(([k, v]) => `<div class="b ${k.startsWith("NanoInspect") ? "me" : ""}">
      <span>${esc(k)}</span><div class="track"><div class="fill" style="width:${Math.max(0.5, 100 * v.cost_per_1000_usd / max)}%"></div></div><b>$${v.cost_per_1000_usd.toFixed(2)}</b></div>`).join("") + `</div>
    <div class="table-wrap"><table><tr><th>Strategy (per 1,000 parts)</th><th>Escaped defects</th><th>Good parts scrapped</th><th>Tier-2 calls on the Nano</th><th>Human reviews</th></tr>` +
    Object.entries(sim).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.escapes_per_1000.toFixed(2)}</td><td>${v.false_rejects_per_1000.toFixed(2)}</td><td>${v.vlm_calls_per_1000.toFixed(1)}</td><td>${v.human_reviews_per_1000.toFixed(1)}</td></tr>`).join("") + `</table></div>`;
}

// ---------------------------------------------------------------- evidence
const FIG = {
  "finetune_loss_both.png": "LoRA fine-tuning of both models on the Nano (training loss)",
  "model_quality_per_category.png": "Accuracy per product for every model",
  "model_quality.png": "Model quality on held-out images: tiers vs baselines",
  "serving_benchmark.png": "Serving throughput and latency on the Nano (vLLM)",
  "escalation_policy.png": "Escalation policy: cost per 1,000 parts",
  "vlm_finetune_loss.png": "LoRA fine-tuning of Qwen2.5-VL-7B on the Nano",
  "llm_examples.png": "What each model says about the same defects",
  "s4_soak.png": "Soak test: both served models under sustained load",
  "line_simulation.png": "Line simulation", "vision_variants.png": "Baseline: ResNet-18 with three training recipes",
  "synthetic_defects.png": "Realistic synthetic defects (used for the ResNet baseline)",
};
async function loadEvidence() {
  const b = await api("/api/benchmarks"), s = b.summary || {}, q = s.model_quality || [];
  const find = pre => q.find(r => r.config.startsWith(pre)) || {};
  const t1 = find("Tier 1"), z = find("Baseline: Qwen2.5-VL-7B zero"), soak = s.S4_combined_soak || {};
  $("#ev-kpis").innerHTML = [
    ["Products covered", "15", "MVTec AD · 73 defect types"],
    ["Tier-1 defect recall", pct(t1.recall), `zero-shot 7B: ${pct(z.recall)}`],
    ["Tier-1 ROC-AUC", t1.roc_auc?.toFixed(3) ?? "–", `zero-shot 7B: ${z.roc_auc?.toFixed(3) ?? "–"}`],
    ["Soak test", soak.minutes ? `${soak.minutes} min` : "–", soak.max_temp_c ? `max ${soak.max_temp_c} °C · throttling: ${soak.throttle_flag_seen}` : ""],
  ].map(([k, v, s]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  $("#gallery").innerHTML = b.figures.filter(f => FIG[f]).sort((a, z) => Object.keys(FIG).indexOf(a) - Object.keys(FIG).indexOf(z))
    .map(f => `<figure><img src="/results/${f}" loading="lazy"><figcaption>${esc(FIG[f])}</figcaption></figure>`).join("");
}

// ---------------------------------------------------------------- home page
let BENCH = null, SHOWCASE = null;
async function home() {
  if (!SHOWCASE) {
    SHOWCASE = await api("/api/showcase?n=32").catch(() => []);
    $("#hero-bg").innerHTML = SHOWCASE.map(x => `<img src="${thumb(x.path, 240)}" alt="">`).join("");
  }
  if (!BENCH) BENCH = (await api("/api/benchmarks").catch(() => ({}))).summary || {};
  const Q = BENCH.model_quality || [], q = pre => Q.find(r => r.config.startsWith(pre)) || {};
  const t1 = q("Tier 1"), t2 = q("Tier 2"), z7 = q("Baseline: Qwen2.5-VL-7B zero"), H = BENCH.escalation_policy?.held_out || {};
  const cas = H["NanoInspect cascade"] || {}, hum = H["Human inspects every part"] || {};
  const sv1 = (BENCH.serving?.tier1 || []).reduce((m, r) => !m || r.requests_per_s > m.requests_per_s ? r : m, null);
  const sv2 = (BENCH.serving?.tier2 || []).reduce((m, r) => !m || r.requests_per_s > m.requests_per_s ? r : m, null);
  $("#h-t1").textContent = t1.recall ? `${pct(t1.recall, 0)} of defects caught · ${sv1 ? sv1.requests_per_s + " parts/s" : ""} →` : "Learn more →";
  $("#h-t2").textContent = t2.recall ? `ROC-AUC ${t2.roc_auc.toFixed(3)} · explains every flagged part →` : "Learn more →";
  $("#h-t3").textContent = cas.human_reviews_per_1000 != null ? `${cas.human_reviews_per_1000.toFixed(0)} reviews per 1,000 parts instead of 1,000 →` : "Learn more →";
  const soak = BENCH.S4_combined_soak || {};
  const rows = [
    { tag: "Tier 1 · every part", img: "screw/test/thread_side/008.png", title: "Qwen2.5-VL-7B, fine-tuned on the Nano",
      text: "A 7-billion-parameter vision-language model, taught on this device with LoRA (28 minutes, 0.5% of its weights, real defect photos only). It checks every part and answers with a verdict, the defect type and where it is, plus a probability read from its own token log-probabilities.",
      facts: [[t1.roc_auc?.toFixed(3), "ROC-AUC"], [pct(t1.recall, 0), `defects caught (zero-shot: ${pct(z7.recall, 0)})`], [sv1 ? sv1.requests_per_s : "–", "parts / s served"]], href: "#models" },
    { tag: "Tier 2 · doubtful parts", img: "hazelnut/test/crack/003.png", title: "Qwen3.8-27B with a known-good reference",
      text: "Verified for DGX Spark in the vLLM recipes and fine-tuned on the Nano (LoRA on 79.7M of 27.4B parameters, 2 h 14 min), served as BF16 + adapter. It sees the part next to a known-good part, confirms or clears tier 1's flag, explains the defect in a sentence and writes the nonconformance report. Operators can chat with it about any part.",
      facts: [[t2.roc_auc?.toFixed(3), "ROC-AUC"], [pct(t2.accuracy, 0), "accuracy"], [sv2 ? sv2.requests_per_s : "–", "parts / s served"]], href: "#how" },
    { tag: "Tier 3 · only when it pays", img: "bottle/test/contamination/009.png", title: "A person in the cloud, by a cost rule",
      text: "When the cheaper automatic decision would still cost more than a human review, the part goes to a reviewer. Only a small crop leaves the plant, through a store-and-forward outbox that survives outages. The reviewer's label returns to the Nano as training data.",
      facts: [[cas.cost_per_1000_usd ? "$" + cas.cost_per_1000_usd.toFixed(0) : "–", "per 1,000 parts"], [hum.cost_per_1000_usd ? "$" + hum.cost_per_1000_usd.toFixed(0) : "–", "if a human checks all"], [cas.human_reviews_per_1000?.toFixed(0) ?? "–", "reviews per 1,000"]], href: "#policy" },
    { tag: "After the verdict", img: "transistor/test/misplaced/002.png", title: "SOP-grounded instructions for the line",
      text: "Every decision becomes schema-validated JSON for the PLC / MES: divert to the reject or hold bin, a nonconformance report whose cause must come from the plant's SOP, and an automatic line stop when a defect keeps repeating.",
      facts: [["73", "defect types in the SOP"], ["ISO 9001 §8.7", "basis"], ["0", "invented actions"]], href: "#sop" },
    { tag: "Proven on the device", img: "carpet/test/hole/002.png", title: "Offline, sustained, and cheaper than the cloud",
      text: "Both LLMs ran together for 15 minutes without throttling, full inspections completed with every outbound connection blocked, and a cloud vision API would have cost about $89 per day for one line at 60 parts per minute.",
      facts: [[soak.max_temp_c ? soak.max_temp_c + " °C" : "–", "max temperature, soak"], [BENCH.S6_offline ? BENCH.S6_offline.outbound_connection_attempts : "–", "outbound connections offline"], ["$" + (89).toFixed(0), "API cost avoided / day"]], href: "#savings" },
  ];
  $("#rows").innerHTML = rows.map((r, i) => `<div class="alt ${i % 2 ? "flip" : ""}">
      <a class="alt-img" href="${r.href}"><img src="${thumb(r.img, 900)}" alt=""><span class="tag">${r.tag}</span></a>
      <div><h3>${r.title}</h3><p>${r.text}</p><div class="facts">${r.facts.map(([v, k]) => `<div><b>${v ?? "–"}</b><span>${k}</span></div>`).join("")}</div>
        <a class="pill" href="${r.href}">Learn more</a></div></div>`).join("");
  tick();
}
async function homeLive(o) {
  const st = o.stats, S = await api("/api/savings").catch(() => null), dec = st.by_decision, total = st.total || 0;
  const items = [
    [num(total), "parts inspected on this Nano", "#overview"],
    [total ? pct(((dec.accept || 0) + (dec.reject || 0)) / total, 0) : "–", "decided at the edge", "#overview"],
    [S ? usd(S.live.net_saved_usd, 3) : "–", "cloud API cost avoided so far", "#savings", true],
    [S ? num(S.live.tokens_total) : "–", "tokens processed locally", "#savings"],
    [st.bytes_inspected ? pct(1 - st.bytes_sent_to_cloud / st.bytes_inspected, 2) : "–", "image data kept on-site", "#escalations"],
    [(o.device || {}).temp_c != null ? o.device.temp_c + " °C" : "–", "GPU temperature now", "#evidence"],
  ];
  $("#home-stats").innerHTML = items.map(([v, k, h, o2]) => `<a class="stat" href="${h}"><div class="v ${o2 ? "o" : ""}">${v}</div><div class="k">${k}</div></a>`).join("");
}

// ---------------------------------------------------------------- how it works
async function how() {
  if (!BENCH) BENCH = (await api("/api/benchmarks").catch(() => ({}))).summary || {};
  const rows = [
    ["Tier 1 checks every part", "screw/test/scratch_head/002.png", "The fine-tuned 7B answers in JSON: verdict, defect type, 3×3 location. Its probability of 'defective' comes from the verdict token's log-probabilities. If it is confident the part is good (below a threshold set on validation parts), the part is accepted at once; 5% of those are randomly audited by tier 2.", "#models", "See model quality"],
    ["Tier 2 double-checks doubtful parts", "hazelnut/test/crack/003.png", "Qwen3.8-27B gets the part next to a known-good reference and a prompt that tells it normal variation is not a defect. It confirms or clears the flag and explains what it sees, and a training-free patch distance to 40 good parts draws the heatmap.", "#inspect", "Try it on a part"],
    ["The cost rule decides who decides", "bottle/test/contamination/009.png", "Both answers put the part in an evidence bucket. Bayes' rule with the line's defect rate gives P(defect); the part goes to a person only if the cheaper automatic option costs more than a review. The quality manager sets the costs.", "#policy", "Open the policy"],
    ["The line gets an instruction", "transistor/test/misplaced/002.png", "The SOP turns the verdict into a disposition, containment and a process check; tier 2 fills in the nonconformance report; schema-validated JSON goes to the PLC / MES, and repeated defects stop the line.", "#sop", "Open the SOP"],
  ];
  $("#how-rows").innerHTML = rows.map(([t, img, txt, h, b], i) => `<div class="alt ${i % 2 ? "flip" : ""}">
      <a class="alt-img" href="${h}"><img src="${thumb(img, 900)}" alt=""><span class="tag">Step ${i + 1}</span></a>
      <div><h3>${t}</h3><p>${txt}</p><a class="pill" href="${h}">${b}</a></div></div>`).join("");
}

// ---------------------------------------------------------------- SOP explorer
let SOP = null, sopProduct = "bottle";
async function sop(product) {
  if (!SOP) SOP = await api("/api/sop");
  if (product) sopProduct = product;
  $("#sop-meta").textContent = `${SOP.sop_id} · version ${SOP.version} · effective ${SOP.effective} · owner: ${SOP.owner}`;
  $("#sop-basis").textContent = SOP.basis + " " + SOP.note;
  $("#sop-products").innerHTML = Object.keys(SOP.products).map(c => `<button class="${c === sopProduct ? "on" : ""}" data-sop="${c}">${c.replace("_", " ")}</button>`).join("");
  const P = SOP.products[sopProduct];
  $("#sop-table").innerHTML = `<tr><th>Defect type (${esc(P.station)})</th><th>Severity</th><th>Disposition</th><th>Likely causes</th><th>Process check</th><th>Rework</th></tr>` +
    Object.entries(P.defects).map(([d, e]) => `<tr><td><b>${d.replaceAll("_", " ")}</b></td><td><span class="pill ${e.severity}">${e.severity}</span></td>
      <td>${esc(SOP.dispositions[e.severity].disposition.replaceAll("_", " "))} → <code>${SOP.dispositions[e.severity].line_command}</code></td>
      <td>${e.likely_causes.map(esc).join("; ")}</td><td>${esc(e.process_check)}</td><td>${e.rework_allowed ? "allowed" : "no"}</td></tr>`).join("");
  $("#sop-disp").innerHTML = `<tr><th>Severity</th><th>Disposition</th><th>Line command</th><th>Containment</th></tr>` +
    Object.entries(SOP.dispositions).map(([k, d]) => `<tr><td><span class="pill ${k}">${k}</span></td><td>${d.disposition.replaceAll("_", " ")}</td><td><code>${d.line_command}</code></td>
      <td class="small">stop line if ${d.containment.stop_line_if_repeats} in ${d.containment.window_parts} parts; re-check last ${d.containment.recheck_last_n}; notify ${d.containment.notify.join(", ")}</td></tr>`).join("");
  const C = await api("/api/controller?n=1").catch(() => null);
  let msg = C?.messages?.[0];
  if (!msg) msg = await fetch("/results/example_machine_instruction.json").then(r => r.json()).catch(() => null);
  $("#sop-json").textContent = msg ? JSON.stringify(msg, null, 2) : "No instruction yet: inspect a defective part.";
}

// ---------------------------------------------------------------- demo scenarios (modal + inspect page)
let SCEN = [];
async function loadScenarios() {
  SCEN = await api("/api/demo_scenarios").catch(() => []);
  $("#scenarios").innerHTML = SCEN.map(s => `<div class="sc" id="sc-${s.id}"><img src="${thumb(s.path, 500)}" alt="">
      <div class="in"><span class="pill ${s.expect}">${DEC[s.expect]}</span><b>${esc(s.title)}</b><p>${esc(s.story)}</p>
      <div class="res" id="sc-res-${s.id}"></div><button class="primary" data-run="${s.id}">Run on the Nano</button></div></div>`).join("");
  $("#demo-mini").innerHTML = SCEN.map(s => `<button data-mini="${s.id}">${esc(s.title)} → ${DEC[s.expect].toLowerCase()}</button>`).join("");
}
async function runScenario(id, where) {
  const s = SCEN.find(x => x.id === id); if (!s) return;
  if (where === "modal") $(`#sc-res-${id}`).innerHTML = `<span class="muted">Running tier 1${s.expect !== "accept" ? " and tier 2" : ""} on the Nano…</span>`;
  else { go("inspect"); $("#cat").value = s.category; busy(); }
  let r;
  try { r = await post("/api/inspect_path", { category: s.category, path: s.path }); }
  catch (e) { const m = `Models unavailable (${esc(e.message)}). Are both vLLM tiers serving?`; if (where === "modal") $(`#sc-res-${id}`).innerHTML = m; else $("#result").innerHTML = `<div class="placeholder">${m}</div>`; return; }
  if (where === "modal") {
    $(`#sc-res-${id}`).innerHTML = `<div><b>Result:</b> <span class="pill ${r.decision}">${DEC[r.decision]}</span> at tier ${r.tier} · ${r.total_s.toFixed(1)} s</div>
      <div class="muted small">${esc(r.reason).slice(0, 180)}</div><a class="small" href="#inspect" data-open="${id}">Open the full result →</a>`;
    lastScenario = r;
  } else renderResult(r);
}
let lastScenario = null;
function openDemo() { $("#demo").hidden = false; if (!SCEN.length) loadScenarios(); }

// ---------------------------------------------------------------- search
const PAGE_WORDS = Object.entries(PAGES).filter(([k, v]) => v).map(([k, v]) => ({ label: v[0], sub: v[1], href: "#" + k }));
function searchResults(q) {
  q = q.trim().toLowerCase(); if (!q) return [];
  const out = PAGE_WORDS.filter(p => (p.label + " " + p.sub).toLowerCase().includes(q)).map(p => ({ ...p, kind: "page" }));
  (META?.categories || []).forEach(c => {
    if (c.replace("_", " ").includes(q)) out.push({ label: `Inspect a ${c.replace("_", " ")}`, sub: `${(META.defect_types[c] || []).length} defect types in the SOP`, act: () => { go("inspect"); $("#cat").value = c; } });
    (META.defect_types[c] || []).forEach(d => { if (d.replaceAll("_", " ").includes(q)) out.push({ label: `${d.replaceAll("_", " ")} (${c.replace("_", " ")})`, sub: "SOP entry: severity, causes, process check", act: () => { go("sop"); setTimeout(() => sop(c), 50); } }); });
  });
  return out.slice(0, 12);
}
let SR = [];
$("#search").addEventListener("input", e => {
  SR = searchResults(e.target.value);
  $("#search-res").hidden = !SR.length;
  $("#search-res").innerHTML = SR.map((r, i) => `<a data-sr="${i}" ${r.href ? `href="${r.href}"` : ""}><b>${esc(r.label)}</b><small>${esc(r.sub)}</small></a>`).join("");
});
$("#search").addEventListener("keydown", e => { if (e.key === "Enter" && SR.length) { pickSearch(0); } if (e.key === "Escape") $("#search-res").hidden = true; });
function pickSearch(i) { const r = SR[i]; $("#search-res").hidden = true; $("#search").value = ""; if (r.act) r.act(); else if (r.href) go(r.href.slice(1)); }

// ---------------------------------------------------------------- global clicks: menus, demo, re-inspect, lightbox, SOP chips
document.querySelectorAll(".mi").forEach(m => {
  let t; m.addEventListener("mouseenter", () => { clearTimeout(t); document.querySelectorAll(".mi").forEach(x => x !== m && x.classList.remove("open")); m.classList.add("open"); });
  m.addEventListener("mouseleave", () => { t = setTimeout(() => m.classList.remove("open"), 180); });
});
document.addEventListener("click", e => {
  const el = e.target.closest("[data-src],[data-demo],#demo-open,[data-run],[data-mini],[data-open],[data-reinspect],[data-sr],[data-sop],.gallery img,.refs img,img.figure,.mega a");
  if (!el) { if (!e.target.closest(".search")) $("#search-res").hidden = true; return; }
  if (el.matches(".mega a")) { el.closest(".mi").classList.remove("open"); return; }
  if (el.dataset.src) { e.preventDefault(); if (view !== "inspect") go("inspect");
    if (el.dataset.src === "upload") file.click(); else $("#cam-on").click(); return; }
  if (el.matches("[data-demo],#demo-open")) { e.preventDefault(); openDemo(); }
  else if (el.dataset.run) runScenario(el.dataset.run, "modal");
  else if (el.dataset.mini) { if (!SCEN.length) loadScenarios().then(() => runScenario(el.dataset.mini, "page")); else runScenario(el.dataset.mini, "page"); }
  else if (el.dataset.open) { e.preventDefault(); $("#demo").hidden = true; go("inspect"); if (lastScenario) renderResult(lastScenario); }
  else if (el.dataset.reinspect) { go("inspect"); $("#cat").value = el.dataset.cat; busy();
    post("/api/inspect_path", { category: el.dataset.cat, path: el.dataset.reinspect }).then(renderResult).catch(err => $("#result").innerHTML = `<div class="placeholder">${esc(err.message)}</div>`); }
  else if (el.dataset.sr) { e.preventDefault(); pickSearch(+el.dataset.sr); }
  else if (el.dataset.sop) sop(el.dataset.sop);
  else if (el.matches(".gallery img,.refs img,img.figure")) { const d = document.createElement("div"); d.className = "lightbox"; d.innerHTML = `<img src="${el.src}">`; d.onclick = () => d.remove(); document.body.appendChild(d); }
});
$("#demo-close").onclick = () => $("#demo").hidden = true;
$("#burger").onclick = () => $(".nav").classList.toggle("open");
window.addEventListener("hashchange", () => $(".nav").classList.remove("open"));
document.querySelectorAll(".mi > a").forEach(a => a.addEventListener("click", e => {   // on touch screens, first tap opens the mega menu
  if (window.matchMedia("(max-width: 1000px)").matches && !a.parentElement.classList.contains("open")) { e.preventDefault(); a.parentElement.classList.add("open"); }
}));
$("#demo").addEventListener("click", e => { if (e.target.id === "demo") $("#demo").hidden = true; });
document.addEventListener("keydown", e => { if (e.key === "Escape") { $("#demo").hidden = true; document.querySelector(".lightbox")?.remove(); } });
$("#drop").addEventListener("click", e => { if (!e.target.closest("a")) file.click(); });

// ---------------------------------------------------------------- fine-tuning & results page
async function results() {
  const R = await api("/api/results").catch(() => null); if (!R) return;
  const idx = a => Object.fromEntries((a || []).map(r => [r.config, r]));
  const B = idx(R.before), A = idx(R.after);
  const t2b = B["Tier 2: Qwen3.8-27B with good reference"] || {}, t2a = A["Tier 2: Qwen3.8-27B with good reference"] || {};
  const t1a = A["Tier 1: Qwen2.5-VL-7B + LoRA (fine-tuned on Nano)"] || {}, z7 = A["Baseline: Qwen2.5-VL-7B zero-shot"] || {};
  const cap = (R.capacity || {})["NanoInspect cascade"] || {}, thr = (R.throughput || {})["NanoInspect cascade"] || {}, hum = (R.throughput || {})["Human inspects every part"] || {};
  $("#res-kpis").innerHTML = [
    ["27B overall score (ROC-AUC)", t2a.roc_auc?.toFixed(3), `before fine-tuning: ${t2b.roc_auc?.toFixed(3)}`],
    ["27B defects caught", pct(t2a.recall, 0), `before: ${pct(t2b.recall, 0)}`],
    ["27B good parts flagged", pct(t2a.false_positive_rate, 0), `before: ${pct(t2b.false_positive_rate, 0)}`],
    ["7B defects caught", pct(t1a.recall, 0), `untrained: ${pct(z7.recall, 0)}`],
    ["Cost per 1,000 parts", cap.cost_per_1000_usd ? "$" + cap.cost_per_1000_usd.toFixed(0) : "–", `capacity mode · default $${thr.cost_per_1000_usd?.toFixed(0)} · human-only $${hum.cost_per_1000_usd?.toFixed(0)}`],
  ].map(([k, v, s2]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v ?? "–"}</div><div class="s">${s2}</div></div>`).join("");
  const T = R.training || {};
  const trow = (k, lab) => T[k] ? `<tr><td>${lab}</td><td>${(T[k].trainable_params / 1e6).toFixed(1)}M of ${(T[k].total_params / 1e9).toFixed(1)}B (${T[k].trainable_pct}%)</td><td>${Math.round(T[k].train_seconds / 60)} min</td><td>${T[k].optimizer_steps}</td><td>${T[k].peak_gpu_mem_gb} GB</td></tr>` : "";
  $("#res-train").innerHTML = `<table><tr><th>Model</th><th>Trained parameters</th><th>Time on the Nano</th><th>Steps</th><th>Peak memory</th></tr>
      ${trow("tier1", "Tier 1 · Qwen2.5-VL-7B")}${trow("tier2", "Tier 2 · Qwen3.8-27B")}</table>
    <p class="muted small">LoRA adapters (rank 16) on the language model; the image encoder and original weights stay frozen. Training data: 1,258 real photos (half the real defects + as many good parts), none from the test set.
    ${R.decision ? `<br><b>Keep-or-revert rule, fixed before training:</b> keep the 27B fine-tune only if the cascade beats $${R.decision.baseline_cost.toFixed(2)} per 1,000 parts. Result: $${R.decision.fine_tuned_cost.toFixed(2)} → <b>${R.decision.keep ? "kept" : "reverted"}</b>.` : ""}</p>`;
  const d = (a, b, lowGood) => a == null || b == null ? "" : (() => { const x = a - b; const good = lowGood ? x < 0 : x > 0; return Math.abs(x) < 0.0005 ? "" : ` <span class="${good ? "delta-up" : "delta-down"}">${x > 0 ? "+" : ""}${(x * 100).toFixed(1)}</span>`; })();
  $("#res-models").innerHTML = `<tr><th>Model</th><th>ROC-AUC</th><th>Defects caught</th><th>Good parts flagged</th><th>Right defect type</th><th>Right location</th></tr>` +
    (R.after || []).map(r => { const b = B[r.config] || {}; const ours = r.config.startsWith("Tier");
      return `<tr${ours ? ' style="font-weight:600"' : ""}><td>${esc(r.config)}</td><td>${r.roc_auc?.toFixed(3) ?? "–"}${d(r.roc_auc, b.roc_auc)}</td>
        <td>${pct(r.recall)}${d(r.recall, b.recall)}</td><td>${pct(r.false_positive_rate)}${d(r.false_positive_rate, b.false_positive_rate, true)}</td>
        <td>${pct(r.defect_type_acc)}${d(r.defect_type_acc, b.defect_type_acc)}</td><td>${pct(r.location_acc)}${d(r.location_acc, b.location_acc)}</td></tr>`; }).join("");
  const F = R.flips || {};
  $("#res-flips").innerHTML = `Coloured numbers are the change in percentage points since before the 27B fine-tune (only tier 2 changed; the rest is run-to-run noise). ` +
    (F.tier1 ? `Fine-tuning the 7B turned <b>${F.tier1.fixed}</b> wrong answers right and ${F.tier1.broken} right answers wrong (vs untrained). ` : "") +
    (F.tier2 ? `Fine-tuning the 27B: <b>${F.tier2.fixed}</b> fixed, ${F.tier2.broken} broken, out of ${F.tier2.total}.` : "");
  const rows = Object.entries({ ...(R.capacity ? { "NanoInspect, capacity mode": R.capacity["NanoInspect cascade"] } : {}), ...(R.throughput || {}) })
    .map(([k, v]) => [k === "NanoInspect cascade" ? "NanoInspect, default (throughput) mode" : k, v]).sort((a, b) => a[1].cost_per_1000_usd - b[1].cost_per_1000_usd);
  const max = Math.max(...rows.map(r => r[1].cost_per_1000_usd));
  $("#res-cost").innerHTML = `<div class="bars">` + rows.map(([k, v]) => `<div class="b ${k.startsWith("NanoInspect") ? "me" : ""}"><span>${esc(k)}</span>
      <div class="track"><div class="fill" style="width:${Math.max(0.5, 100 * v.cost_per_1000_usd / max)}%"></div></div><b>$${v.cost_per_1000_usd.toFixed(0)}</b></div>`).join("") + `</div>
    <div class="table-wrap"><table><tr><th>Strategy</th><th>Escaped defects</th><th>Good parts scrapped</th><th>Human reviews</th><th>Tier-2 calls</th></tr>` +
    rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.escapes_per_1000.toFixed(1)}</td><td>${v.false_rejects_per_1000.toFixed(1)}</td><td>${v.human_reviews_per_1000.toFixed(0)}</td><td>${v.vlm_calls_per_1000.toFixed(0)}</td></tr>`).join("") + `</table></div>`;
  const sb = (R.serving_before || []).reduce((m, r) => !m || r.requests_per_s > m.requests_per_s ? r : m, null), sa = (R.serving_after || []).reduce((m, r) => !m || r.requests_per_s > m.requests_per_s ? r : m, null);
  $("#res-tradeoff").innerHTML = `<ul>
    <li><b>Better decisions:</b> the fine-tuned 27B catches more defects with fewer false alarms and names the defect and location far more often.</li>
    <li><b>Slower and hungrier:</b> the adapter needs the full-precision (BF16) weights, so tier 2 serves ${sa ? sa.requests_per_s : "–"} parts/s instead of ${sb ? sb.requests_per_s : "–"} (NVFP4) and uses ${R.energy_after ? R.energy_after.joules_per_item.toFixed(0) : "–"} J instead of ${R.energy_before ? R.energy_before.joules_per_item.toFixed(0) : "–"} J per check. That is why the cascade still matters: the small model handles most parts.</li>
    <li><b>Plainer sentences:</b> trained on template answers, the fine-tuned 27B writes template-like explanations, so the untrained 27B on the same server writes the operator's sentence, the defect report and the chat replies. The decision always comes from the fine-tuned model.</li>
    <li><b>Capacity mode</b> sends up to about half of all parts to tier 2 (threshold ${R.capacity_t_lo ? R.capacity_t_lo.toFixed(3) : "–"}, set from validation parts and measured capacity). It fits lines up to about 50 parts/min; beyond that, run the default mode or add a second tier-2 server.</li></ul>`;
}

// ---------------------------------------------------------------- boot
(async () => {
  document.querySelectorAll("img[data-thumb]").forEach(i => i.src = thumb(i.dataset.thumb, 600));
  META = await api("/api/meta");
  $("#site").textContent = META.site;
  $("#cloud-top").href = (META.cloud_url || "").replace("127.0.0.1", location.hostname);
  $("#cat").innerHTML = META.categories.map(c => `<option ${c === "bottle" ? "selected" : ""}>${c}</option>`).join("");
  loadScenarios();
  route();
  setInterval(tick, 2500);
  setInterval(async () => { META = await api("/api/meta").catch(() => META); }, 15000);
})();
