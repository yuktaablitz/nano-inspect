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
let META = null, view = "overview", devHist = [], lastInspection = null, chatHist = [];

// ---------------------------------------------------------------- navigation
document.querySelectorAll("nav a").forEach(a => a.onclick = () => { history.replaceState(null, "", "#" + a.dataset.view); show(a.dataset.view); });
function show(v) {
  view = v;
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("on", a.dataset.view === v));
  document.querySelectorAll(".view").forEach(s => s.hidden = s.id !== `v-${v}`);
  if (v === "policy") loadPolicy();
  if (v === "evidence") loadEvidence();
  if (v === "models") loadModels();
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
  } catch (e) { $("#st-sync").innerHTML = `<span class="dot bad"></span>Edge API unreachable`; }
}
function header(o) {
  const d = o.device || {}, s = o.sync;
  devHist.push({ t: Date.now(), temp: d.temp_c, w: d.power_w, u: d.gpu_util_pct }); devHist = devHist.slice(-120);
  $("#dev").innerHTML = `HP ZGX Nano · GB10<br>${d.gpu_util_pct ?? "–"}% GPU · ${d.temp_c ?? "–"} °C · ${d.power_w ?? "–"} W`;
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
    ["Tier 2 · Qwen3.8-27B NVFP4", "compares with a known-good reference · uncertain + audit", t2, "var(--t2)"],
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
  return `<div class="part ${r.decision}" title="${esc(r.reason)}">${img ? `<img src="${img}" loading="lazy">` : `<div style="aspect-ratio:1"></div>`}
    <span class="t t${r.tier}">T${r.tier}</span><div class="lbl"><b>${esc(r.category)}</b><span>${DEC[r.decision]}</span></div></div>`;
}
function spark(vals, label, max) {
  const v = vals.filter(x => x != null && !isNaN(x)); const W = 320, H = 46;
  if (v.length < 2) return `<div class="small muted">${label}: –</div>`;
  const m = Math.max(max, ...v), pts = v.map((y, i) => `${(i / (v.length - 1)) * W},${H - (y / m) * (H - 4) - 2}`).join(" ");
  return `<div class="small"><b>${label}</b> <span class="muted">${v[v.length - 1].toFixed(0)}</span></div>
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2"/></svg>`;
}
$("#hero-start").onclick = () => { history.replaceState(null, "", "#line"); show("line"); startLine(); };

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
    (r.explanation ? `<div class="small">“${esc(r.explanation)}”</div>` : "");
}
function renderResult(r) {
  lastInspection = r.inspection_id; chatHist = []; $("#chat-log").innerHTML = "";
  if (!r.marked_b64) { $("#result").innerHTML = `<div class="banner manual_review"><div class="d">HUMAN REVIEW</div></div><div class="why">${esc(r.reason)}</div>`; return; }
  const path = [1, 2, 3].map(t => `<span class="chip ${r.tier >= t ? "on t" + t : ""}">${TIER[t]}</span>`).join("");
  const gt = r.ground_truth ? `<div>Ground truth (dataset)</div><div><b>${esc(r.ground_truth.label)}</b>${r.ground_truth.label === "defective" ? ` · ${esc(r.ground_truth.defect_type)} @ ${esc(r.ground_truth.location)}` : ""}</div>` : "";
  $("#result").innerHTML = `
    <div class="banner ${r.decision}"><div><div class="d">${DEC[r.decision]}</div><div class="small">${esc(r.recommended_action)}</div></div><div class="path">${path}</div></div>
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
  $("#controller").innerHTML = (C.stopped ? `<div class="banner reject"><div><div class="d">LINE STOP</div><div class="small">${esc(C.stopped.defect_type)} on ${esc(C.stopped.product)} repeated ${C.stopped.hits}× within ${C.stopped.window} parts (SOP containment rule) at ${C.stopped.time}</div></div><button id="ctl-reset">Acknowledge</button></div>` : "") +
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
  const b = [...(m.tier1.benchmark || []).map(r => ({ tier: "Tier 1 · 7B + LoRA", ...r })), ...(m.tier2.benchmark || []).map(r => ({ tier: "Tier 2 · 27B NVFP4", ...r }))];
  $("#servebench").innerHTML = b.length ? `<tr><th>Model</th><th>Concurrent clients</th><th>Requests/s</th><th>Output tokens/s</th><th>p50 latency</th><th>p95 latency</th></tr>` +
    b.map(r => `<tr><td>${r.tier}</td><td>${r.concurrency}</td><td>${r.requests_per_s}</td><td>${r.output_tokens_per_s}</td><td>${sec(r.p50_latency_s)}</td><td>${sec(r.p95_latency_s)}</td></tr>`).join("") :
    `<tr><td>Run the notebook's serving benchmark to fill this table.</td></tr>`;
}

// ---------------------------------------------------------------- policy
async function loadPolicy() {
  const p = await api("/api/policy");
  $("#c-esc").value = p.costs.escape_usd; $("#c-fr").value = p.costs.false_reject_usd; $("#c-hr").value = p.costs.human_review_usd;
  $("#c-dr").value = +(p.defect_rate * 100).toFixed(2); $("#c-ar").value = +(p.audit_rate * 100).toFixed(1); $("#c-pv").value = p.privacy;
  renderPolicy(p, await api("/api/policy/simulate"));
}
$("#apply").onclick = async () => {
  const p = await post("/api/policy", { escape_usd: +$("#c-esc").value, false_reject_usd: +$("#c-fr").value, human_review_usd: +$("#c-hr").value,
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
  if (!sim) { $("#pol-sim").innerHTML = `<div class="muted">No held-out simulation available yet.</div>`; return; }
  const max = Math.max(...Object.values(sim).map(v => v.cost_per_1000_usd));
  $("#pol-sim").innerHTML = `<div class="bars">` + Object.entries(sim).map(([k, v]) => `<div class="b ${k.startsWith("NanoInspect") ? "me" : ""}">
      <span>${esc(k)}</span><div class="track"><div class="fill" style="width:${Math.max(0.5, 100 * v.cost_per_1000_usd / max)}%"></div></div><b>$${v.cost_per_1000_usd.toFixed(2)}</b></div>`).join("") + `</div>
    <div class="table-wrap"><table><tr><th>Strategy (per 1,000 parts)</th><th>Escaped defects</th><th>Good parts scrapped</th><th>Tier-2 calls on the Nano</th><th>Human reviews</th></tr>` +
    Object.entries(sim).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.escapes_per_1000.toFixed(2)}</td><td>${v.false_rejects_per_1000.toFixed(2)}</td><td>${v.vlm_calls_per_1000.toFixed(1)}</td><td>${v.human_reviews_per_1000.toFixed(1)}</td></tr>`).join("") + `</table></div>`;
}

// ---------------------------------------------------------------- evidence
const FIG = {
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

// ---------------------------------------------------------------- boot
(async () => {
  META = await api("/api/meta");
  $("#site").textContent = META.site;
  $("#cat").innerHTML = META.categories.map(c => `<option ${c === "bottle" ? "selected" : ""}>${c}</option>`).join("");
  const h = location.hash.slice(1);
  if (h && document.getElementById(`v-${h}`)) show(h); else tick();
  setInterval(tick, 2000);
  setInterval(async () => { META = await api("/api/meta").catch(() => META); }, 15000);
})();
window.addEventListener("hashchange", () => { const h = location.hash.slice(1); if (document.getElementById(`v-${h}`)) show(h); });
