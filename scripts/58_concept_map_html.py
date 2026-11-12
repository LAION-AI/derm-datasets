"""Build report/concept_map/index.html: interactive map of diagnosis concepts, ours vs Derm1M.

Inputs: work/concepts/concept_map.json (56_concept_map.py), work/concepts/cluster_names.json (57_cluster_names.py).
Single self-contained HTML file (data inline; d3 7.9.0 from cdnjs).
"""
import json

from common import WORK

OUT = WORK.parent / "report/concept_map/index.html"

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Concept Map</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --ours: #2a78d6; --derm: #eb6834; --both: #1baf7a; --below: #c3c2b7;
  --hull: 11,11,11;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --ours: #3987e5; --derm: #d95926; --both: #199e70; --below: #4a4a46;
    --hull: 240,239,236;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --ours: #3987e5; --derm: #d95926; --both: #199e70; --below: #4a4a46;
  --hull: 240,239,236;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Arial, sans-serif; }
header { padding: 20px 24px 8px; max-width: 1600px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--ink2); margin: 0; max-width: 960px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; padding: 12px 24px; max-width: 1600px; margin: 0 auto; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.tile .k { color: var(--ink2); font-size: 12px; display: flex; align-items: center; gap: 6px; }
.tile .v { font-size: 22px; font-weight: 650; font-variant-numeric: tabular-nums; }
.tile .s { color: var(--muted); font-size: 12px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; padding: 4px 24px 10px; max-width: 1600px; margin: 0 auto; }
.controls label { display: inline-flex; align-items: center; gap: 6px; color: var(--ink2); cursor: pointer; }
select, input[type=search], button { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; }
button { cursor: pointer; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 12px; padding: 0 24px 16px; max-width: 1600px; margin: 0 auto; }
#mapwrap { position: relative; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; height: min(78vh, 900px); min-height: 420px; overflow: hidden; }
canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
canvas:active { cursor: grabbing; }
#tip { position: absolute; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 13px; box-shadow: 0 4px 16px rgba(0,0,0,.12); max-width: 300px; display: none; z-index: 2; }
#tip b { display: block; font-size: 14px; margin-bottom: 2px; }
#tip .row { display: flex; justify-content: space-between; gap: 16px; font-variant-numeric: tabular-nums; }
#tip .m { color: var(--ink2); }
.hint { position: absolute; left: 10px; bottom: 8px; color: var(--muted); font-size: 12px; pointer-events: none; }
#reset { position: absolute; right: 10px; top: 10px; }
aside { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; display: flex; flex-direction: column; height: min(78vh, 900px); min-height: 420px; }
aside h2 { font-size: 14px; margin: 0; padding: 10px 12px 6px; }
aside .note { color: var(--muted); font-size: 12px; padding: 0 12px 6px; margin: 0; }
#legend { overflow-y: auto; flex: 1; padding: 0 6px 8px; }
.cl { display: grid; grid-template-columns: 28px 1fr; gap: 2px 8px; padding: 6px; border-radius: 6px; cursor: pointer; }
.cl:hover, .cl.on { background: rgba(var(--hull), .06); }
.cl .no { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; text-align: right; padding-top: 1px; }
.cl .nm { font-weight: 600; font-size: 13px; }
.cl .meta { grid-column: 2; color: var(--ink2); font-size: 12px; font-variant-numeric: tabular-nums; }
.bar { grid-column: 2; display: flex; height: 6px; gap: 2px; margin-top: 2px; }
.bar span { display: block; height: 100%; border-radius: 2px; }
footer { color: var(--ink2); font-size: 12.5px; padding: 0 24px 28px; max-width: 1600px; margin: 0 auto; }
footer p { margin: 4px 0; max-width: 1100px; }
svg.sh { vertical-align: middle; flex: none; }
@media (max-width: 900px) {
  header, .tiles, .controls, main, footer { padding-left: 16px; padding-right: 16px; }
  main { grid-template-columns: 1fr; }
  aside { height: 420px; }
  #mapwrap { height: min(70vh, 120vw); }
}
</style>
</head>
<body>
<header>
  <h1>Diagnosis concepts: our dataset vs Derm1M</h1>
  <p class="sub">Each point is one diagnosis concept (SapBERT-normalised label). Position: PaCMAP layout of the concept-name embedding, so similar concepts lie close together. Shaded regions: HDBSCAN clusters, named by an LLM. Shape and colour show which dataset contains the concept with at least the selected number of images.</p>
</header>
<section class="tiles" id="tiles" aria-live="polite"></section>
<div class="controls">
  <label>Concept present if ≥
    <select id="thr"><option>2</option><option>3</option><option selected>5</option><option>10</option><option>20</option></select> images</label>
  <label><input type="checkbox" id="f-ours" checked> <span id="lg-ours"></span> only ours</label>
  <label><input type="checkbox" id="f-both" checked> <span id="lg-both"></span> both</label>
  <label><input type="checkbox" id="f-derm" checked> <span id="lg-derm"></span> only Derm1M</label>
  <label><input type="checkbox" id="f-below"> <span id="lg-below"></span> below threshold</label>
  <label><input type="checkbox" id="f-hull" checked> cluster shading</label>
  <input type="search" id="q" placeholder="Find a concept…" aria-label="Find a concept" size="22">
  <button id="theme" type="button" aria-label="Toggle dark mode">◐ Theme</button>
</div>
<main>
  <div id="mapwrap">
    <canvas id="map" aria-label="Concept map"></canvas>
    <div id="tip"></div>
    <button id="reset" type="button">Reset view</button>
    <div class="hint">Scroll to zoom · drag to pan · hover for details · click a cluster in the legend to zoom</div>
  </div>
  <aside>
    <h2>Clusters</h2>
    <p class="note" id="lnote"></p>
    <div id="legend"></div>
  </aside>
</main>
<footer>
  <p><b>Concepts.</b> Diagnosis strings from our atlas labels, from Derm1M's disease labels and from phrases in English journal captions that match this shared vocabulary were normalised with SapBERT. Spelling variants and near-synonyms (cosine ≥ 0.95) were merged into one concept. Counts are unique images per dataset.</p>
  <p><b>Caveat.</b> The two datasets were labelled differently. Derm1M has a free-text disease label for every image, so it contributes many rare label variants; that is why its count at ≥ 1 image is high. Our journal figures receive concepts only by caption phrase matching. Differences in concept counts therefore partly reflect labelling style, not only clinical coverage. The map shows the __NMAP__ concepts with at least __MIN__ images in one of the datasets.</p>
  <p><b>Clusters.</b> PaCMAP (15 neighbours, FP ratio 8, seed 0); HDBSCAN (min. cluster size 25, min. samples 8) was run on the 2-D layout; __NOISE__% of concepts belong to no cluster. Cluster names were generated automatically by __MODEL__ from the 25 most frequent concepts of each cluster and have not been reviewed.</p>
</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const DATA = __DATA__;
const NAMES = __NAMES__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = d3.format(",");
const P = DATA.points, CL = DATA.clusters;
const cname = c => (NAMES[c] && NAMES[c].name) || ("Cluster " + c);
const state = { t: 5, show: { ours: true, both: true, derm: true, below: false }, hull: true, hoverC: null, query: "" };

function cat(p) {
  const o = p.o >= state.t, d = p.d >= state.t;
  return o && d ? "both" : o ? "ours" : d ? "derm" : "below";
}
const LABEL = { ours: "only ours", both: "in both", derm: "only Derm1M", below: "below threshold" };

/* --- shapes (shared by canvas, legend icons and tooltip) --- */
function shapePath(ctx, kind, x, y, r) {
  ctx.beginPath();
  if (kind === "ours" || kind === "below") ctx.arc(x, y, r, 0, 2 * Math.PI);
  else if (kind === "derm") { const h = r * 1.25; ctx.moveTo(x, y - h); ctx.lineTo(x + h * 0.95, y + h * 0.7); ctx.lineTo(x - h * 0.95, y + h * 0.7); ctx.closePath(); }
  else { const h = r * 1.3; ctx.moveTo(x, y - h); ctx.lineTo(x + h, y); ctx.lineTo(x, y + h); ctx.lineTo(x - h, y); ctx.closePath(); }
}
function icon(kind, size = 14) {
  const c = document.createElement("canvas"), s = window.devicePixelRatio || 1;
  c.width = c.height = size * s; c.style.width = c.style.height = size + "px"; c.className = "sh";
  const x = c.getContext("2d"); x.scale(s, s);
  shapePath(x, kind, size / 2, size / 2 + (kind === "derm" ? 1 : 0), size * 0.32);
  x.fillStyle = css("--" + kind); x.fill();
  return c;
}
function drawIcons() {
  for (const k of ["ours", "both", "derm", "below"]) {
    const el = document.getElementById("lg-" + k); el.replaceChildren(icon(k));
  }
}

/* --- stat tiles --- */
function tiles() {
  let n = { ours: 0, both: 0, derm: 0 };
  for (const p of P) { const c = cat(p); if (c in n) n[c]++; }
  const T = DATA.totals[state.t] || null, T1 = DATA.totals["1"];
  const tot = T ? T : { ours: n.ours + n.both, derm1m: n.derm + n.both, both: n.both };
  const el = document.getElementById("tiles");
  const card = (k, label, v, s) => `<div class="tile"><div class="k">${k ? `<span data-ic="${k}"></span>` : ""}${label}</div><div class="v">${fmt(v)}</div><div class="s">${s}</div></div>`;
  el.innerHTML =
    card("", "Concepts in our dataset", tot.ours, `≥ ${state.t} images · ${fmt(T1.ours)} with ≥ 1`) +
    card("", "Concepts in Derm1M", tot.derm1m, `≥ ${state.t} images · ${fmt(T1.derm1m)} with ≥ 1`) +
    card("both", "In both", tot.both, `${fmt(T1.both)} with ≥ 1 image in each`) +
    card("ours", "Only in ours", tot.ours - tot.both, `${fmt(T1.ours - T1.both)} at ≥ 1`) +
    card("derm", "Only in Derm1M", tot.derm1m - tot.both, `${fmt(T1.derm1m - T1.both)} at ≥ 1`);
  el.querySelectorAll("[data-ic]").forEach(s => s.replaceWith(icon(s.dataset.ic, 12)));
}

/* --- legend --- */
const byC = d3.group(P.filter(p => p.c >= 0), p => p.c);
function legend() {
  const L = document.getElementById("legend");
  const rows = CL.map(c => {
    const pts = byC.get(c.c) || [];
    const n = { ours: 0, both: 0, derm: 0 };
    for (const p of pts) { const k = cat(p); if (k in n) n[k]++; }
    return { c: c.c, size: c.size, n };
  }).sort((a, b) => b.size - a.size);
  document.getElementById("lnote").textContent = `${CL.length} clusters, sorted by size. Bar: only ours / both / only Derm1M at ≥ ${state.t} images.`;
  L.innerHTML = rows.map(r => {
    const tot = r.n.ours + r.n.both + r.n.derm || 1;
    const seg = k => r.n[k] ? `<span style="width:${100 * r.n[k] / tot}%;background:var(--${k})"></span>` : "";
    return `<div class="cl" data-c="${r.c}" tabindex="0"><div class="no">${r.c + 1}</div><div class="nm">${cname(r.c)}</div>
      <div class="meta">${r.size} concepts · ours ${r.n.ours + r.n.both} · Derm1M ${r.n.derm + r.n.both} · both ${r.n.both}</div>
      <div class="bar">${seg("ours")}${seg("both")}${seg("derm")}</div></div>`;
  }).join("");
  L.querySelectorAll(".cl").forEach(el => {
    const c = +el.dataset.c;
    el.onmouseenter = () => { state.hoverC = c; draw(); };
    el.onmouseleave = () => { state.hoverC = null; draw(); };
    el.onclick = el.onkeydown = e => { if (e.type === "click" || e.key === "Enter") zoomToCluster(c); };
  });
}

/* --- canvas map --- */
const wrap = document.getElementById("mapwrap"), canvas = document.getElementById("map"), ctx = canvas.getContext("2d");
let W = 0, H = 0, dpr = 1, transform = d3.zoomIdentity;
// initial view: 1st-99th percentile (a few far outlier groups would otherwise compress the map); pan to see them
const q = (a, f) => d3.quantile(a, f), xs = P.map(p => p.x).sort(d3.ascending), ys = P.map(p => p.y).sort(d3.ascending);
const xe = [q(xs, 0.01), q(xs, 0.99)], ye = [q(ys, 0.01), q(ys, 0.99)];
const sx = d3.scaleLinear(), sy = d3.scaleLinear();
const hulls = CL.map(c => {
  const pts = (byC.get(c.c) || []).map(p => [p.x, p.y]);
  const h = pts.length >= 3 ? d3.polygonHull(pts) : null;
  return { c: c.c, hull: h, cen: h ? d3.polygonCentroid(h) : [d3.mean(pts, p => p[0]), d3.mean(pts, p => p[1])], size: c.size };
});
const rad = p => 1.8 + 1.1 * Math.log10(1 + Math.max(p.o, p.d));
let visible = [], delaunay = null, hits = new Set();

function resize() {
  dpr = window.devicePixelRatio || 1;
  W = wrap.clientWidth; H = wrap.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const pad = 24, dx = xe[1] - xe[0], dy = ye[1] - ye[0];
  const k = Math.min((W - 2 * pad) / dx, (H - 2 * pad) / dy);
  sx.domain(xe).range([(W - k * dx) / 2, (W + k * dx) / 2]);
  sy.domain(ye).range([(H + k * dy) / 2, (H - k * dy) / 2]);
  draw();
}
function filter() {
  visible = P.filter(p => state.show[cat(p)]);
  const order = { below: 0, derm: 1, ours: 2, both: 3 };
  visible.sort((a, b) => order[cat(a)] - order[cat(b)]);
  delaunay = d3.Delaunay.from(visible, p => sx(p.x), p => sy(p.y));
}
const X = p => transform.applyX(sx(p.x)), Y = p => transform.applyY(sy(p.y));

function hullPath(h, padPx) {
  const cx = transform.applyX(sx(h.cen[0])), cy = transform.applyY(sy(h.cen[1]));
  const pts = h.hull.map(([x, y]) => {
    const px = transform.applyX(sx(x)), py = transform.applyY(sy(y));
    const dx = px - cx, dy = py - cy, l = Math.hypot(dx, dy) || 1;
    return [px + dx / l * padPx, py + dy / l * padPx];
  });
  d3.line().curve(d3.curveCatmullRomClosed.alpha(0.5)).context(ctx)(pts);
}
function draw() {
  if (!W) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, W, H);
  const hc = css("--hull");
  if (state.hull) {
    for (const h of hulls) {
      if (!h.hull) continue;
      ctx.beginPath(); hullPath(h, 10);
      const on = state.hoverC === h.c;
      ctx.fillStyle = `rgba(${hc},${on ? 0.13 : 0.05})`; ctx.fill();
      ctx.strokeStyle = `rgba(${hc},${on ? 0.45 : 0.12})`; ctx.lineWidth = on ? 1.5 : 1; ctx.stroke();
    }
  }
  const col = { ours: css("--ours"), both: css("--both"), derm: css("--derm"), below: css("--below") };
  const surf = css("--surface"), k = Math.sqrt(transform.k);
  const dim = state.hoverC !== null || hits.size > 0;
  for (const p of visible) {
    const c = cat(p), x = X(p), y = Y(p);
    if (x < -10 || y < -10 || x > W + 10 || y > H + 10) continue;
    const r = rad(p) * Math.min(k, 2.2);
    const focus = hits.size ? hits.has(p) : state.hoverC === null || p.c === state.hoverC;
    ctx.globalAlpha = dim && !focus ? 0.18 : 0.9;
    shapePath(ctx, c, x, y, r);
    ctx.fillStyle = col[c]; ctx.fill();
    if (r > 3) { ctx.lineWidth = 1; ctx.strokeStyle = surf; ctx.stroke(); }
  }
  ctx.globalAlpha = 1;
  for (const p of hits) {   // search results: ring
    shapePath(ctx, "ours", X(p), Y(p), rad(p) * Math.min(k, 2.2) + 4);
    ctx.lineWidth = 2; ctx.strokeStyle = css("--ink"); ctx.stroke();
  }
  labels();
}
function labels() {
  if (!state.hull) return;
  const boxes = [], ink = css("--ink"), surf = css("--surface");
  ctx.font = "600 12px system-ui, -apple-system, Segoe UI, Arial, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const minSize = transform.k > 3 ? 0 : transform.k > 1.6 ? 45 : 80;
  const order = [...hulls].sort((a, b) => (b.c === state.hoverC) - (a.c === state.hoverC) || b.size - a.size);
  for (const h of order) {
    if (!h.hull || (h.size < minSize && h.c !== state.hoverC)) continue;
    const t = cname(h.c), x = transform.applyX(sx(h.cen[0])), y = transform.applyY(sy(h.cen[1]));
    const w = ctx.measureText(t).width + 8, b = [x - w / 2, y - 9, x + w / 2, y + 9];
    if (x < 0 || x > W || y < 0 || y > H) continue;
    if (boxes.some(o => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1])) continue;
    boxes.push(b);
    ctx.lineWidth = 4; ctx.strokeStyle = surf; ctx.lineJoin = "round"; ctx.strokeText(t, x, y);
    ctx.fillStyle = ink; ctx.fillText(t, x, y);
  }
}

/* --- interaction --- */
const zoom = d3.zoom().scaleExtent([0.4, 40]).on("zoom", e => { transform = e.transform; hideTip(); draw(); });
const sel = d3.select(canvas).call(zoom);
document.getElementById("reset").onclick = () => sel.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
function zoomToCluster(c) {
  const pts = byC.get(c) || []; if (!pts.length) return;
  const x0 = d3.min(pts, p => sx(p.x)), x1 = d3.max(pts, p => sx(p.x)), y0 = d3.min(pts, p => sy(p.y)), y1 = d3.max(pts, p => sy(p.y));
  const k = Math.min(12, 0.8 / Math.max((x1 - x0) / W, (y1 - y0) / H, 1e-3));
  state.hoverC = c;
  sel.transition().duration(650).call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2));
  document.querySelectorAll(".cl").forEach(el => el.classList.toggle("on", +el.dataset.c === c));
}
const tip = document.getElementById("tip");
function hideTip() { tip.style.display = "none"; }
function showTip(html, mx, my) {
  tip.innerHTML = html; tip.style.display = "block";
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = Math.min(mx + 14, W - tw - 6) + "px";
  tip.style.top = (my + 14 + th > H ? my - th - 10 : my + 14) + "px";
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
canvas.addEventListener("pointermove", e => {
  const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const ix = transform.invertX(mx), iy = transform.invertY(my);
  let p = null;
  if (delaunay && visible.length) {
    const i = delaunay.find(ix, iy), q = visible[i];
    if (q && Math.hypot(X(q) - mx, Y(q) - my) <= Math.max(8, rad(q) * Math.sqrt(transform.k) + 4)) p = q;
  }
  if (p) {
    const c = cat(p);
    const holder = document.createElement("div"); holder.appendChild(icon(c, 12));
    showTip(`<b>${esc(p.n)}</b><div class="m" style="display:flex;gap:6px;align-items:center">${holder.innerHTML}${LABEL[c]} (≥ ${state.t} images)</div>
      <div class="row"><span>Images in ours</span><span>${fmt(p.o)}</span></div>
      <div class="row"><span>Images in Derm1M</span><span>${fmt(p.d)}</span></div>
      ${p.c >= 0 ? `<div class="m">Cluster ${p.c + 1}: ${esc(cname(p.c))}</div>` : `<div class="m">No cluster</div>`}
      ${p.s && p.s.length ? `<div class="m">Also: ${esc(p.s.join("; "))}</div>` : ""}
      ${p.on ? `<div class="m">Derm1M ontology: ${esc(p.on)}</div>` : ""}`, mx, my);
    tip.querySelector("canvas")?.replaceWith(icon(c, 12));
    return;
  }
  if (state.hull) {   // over a cluster region?
    for (const h of hulls) {
      if (h.hull && d3.polygonContains(h.hull, [sx.invert(ix), sy.invert(iy)])) {
        const pts = byC.get(h.c), n = pts.filter(q => q.o >= state.t).length, m = pts.filter(q => q.d >= state.t).length;
        showTip(`<b>${esc(cname(h.c))}</b><div class="m">Cluster ${h.c + 1} · ${h.size} concepts</div>
          <div class="row"><span>in ours (≥ ${state.t})</span><span>${n}</span></div><div class="row"><span>in Derm1M (≥ ${state.t})</span><span>${m}</span></div>`, mx, my);
        return;
      }
    }
  }
  hideTip();
});
canvas.addEventListener("pointerleave", hideTip);

/* --- controls --- */
document.getElementById("thr").onchange = e => { state.t = +e.target.value; update(); };
for (const k of ["ours", "both", "derm", "below"]) document.getElementById("f-" + k).onchange = e => { state.show[k] = e.target.checked; filter(); draw(); };
document.getElementById("f-hull").onchange = e => { state.hull = e.target.checked; draw(); };
document.getElementById("q").addEventListener("input", e => {
  const q = e.target.value.trim().toLowerCase();
  hits = new Set(q.length >= 3 ? visible.filter(p => p.n.toLowerCase().includes(q) || (p.s || []).some(s => s.toLowerCase().includes(q))) : []);
  draw();
});
document.getElementById("q").addEventListener("keydown", e => {
  if (e.key !== "Enter" || !hits.size) return;
  const pts = [...hits];
  const x0 = d3.min(pts, p => sx(p.x)), x1 = d3.max(pts, p => sx(p.x)), y0 = d3.min(pts, p => sy(p.y)), y1 = d3.max(pts, p => sy(p.y));
  const k = Math.min(10, 0.7 / Math.max((x1 - x0) / W, (y1 - y0) / H, 0.02));
  sel.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2));
});
document.getElementById("theme").onclick = () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("concept-map-theme", root.dataset.theme); } catch (e) {}
  drawIcons(); tiles(); legend(); draw();
};
try { const t = localStorage.getItem("concept-map-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawIcons(); tiles(); draw(); });

function update() { tiles(); legend(); filter(); draw(); }
new ResizeObserver(() => { resize(); filter(); draw(); }).observe(wrap);
drawIcons(); update(); resize();
</script>
</body>
</html>
"""


def model_label(model):
    """'incoai/Qwen3.8-27B-Splash' -> 'Qwen3.8-27B-Splash (local Splash server)'; 'gemma2:latest' -> 'Gemma2 (local, via Ollama)'."""
    if ":" in model and "/" not in model:
        return model.split(":")[0].capitalize() + " (local, via Ollama)"
    name = model.split("/")[-1]
    return name + (" (local Splash server)" if "splash" in name.lower() else "")


def main():
    M = json.load(open(WORK / "concepts/concept_map.json"))
    N = json.load(open(WORK / "concepts/cluster_names.json"))
    # identical LLM names for different clusters: append the cluster's most frequent concept
    from collections import Counter
    dup = {k for k, v in Counter(x["name"] for x in N.values()).items() if v > 1}
    top = {str(c["c"]): c["top"][0] for c in M["clusters"]}
    N = {k: {**v, "name": f"{v['name']} ({top[k]})" if v["name"] in dup else v["name"]} for k, v in N.items()}
    noise = 100 * sum(p["c"] < 0 for p in M["points"]) / len(M["points"])
    model = next(iter(N.values()))["model"] if N else "an LLM"
    html = (TEMPLATE.replace("__DATA__", json.dumps(M, ensure_ascii=False, separators=(",", ":")))
            .replace("__NAMES__", json.dumps(N, ensure_ascii=False, separators=(",", ":")))
            .replace("__NMAP__", f"{len(M['points']):,}").replace("__MIN__", str(M["min_images"]))
            .replace("__NOISE__", f"{noise:.0f}").replace("__MODEL__", model_label(model)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(OUT, f"{len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
