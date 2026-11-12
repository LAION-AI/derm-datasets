"""Interactive map of the permissive reference pool: three views (image / caption / image + caption embeddings, PaCMAP +
HDBSCAN from 83_ref_clusters.py), cluster names from 84_ref_cluster_names.py (Splash vision LLM), hover shows the
thumbnail. Writes work/refs/clusters/index.html (next to thumbs/; local file, data inline).
"""
import json

import pandas as pd

from common import WORK

D = WORK / "refs/clusters"

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reference Clusters</title>
<style>
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --border:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s0:#c3c2b7; --hull:11,11,11; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --border:rgba(255,255,255,.10); --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s0:#4a4a46; --hull:240,239,236; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --border:rgba(255,255,255,.10); --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s0:#4a4a46; --hull:240,239,236; }
* { box-sizing:border-box; } body { margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",Arial,sans-serif; }
header, .controls, main, footer { max-width:1600px; margin:0 auto; padding:12px 24px; }
h1 { font-size:20px; margin:4px 0; } .sub { color:var(--ink2); margin:0; max-width:1000px; }
.controls { display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; }
.tabs button { font:inherit; border:1px solid var(--border); background:var(--surface); color:var(--ink); padding:5px 12px; cursor:pointer; }
.tabs button:first-child { border-radius:6px 0 0 6px; } .tabs button:last-child { border-radius:0 6px 6px 0; }
.tabs button[aria-pressed=true] { background:var(--ink); color:var(--surface); }
select, input[type=search], .btn { font:inherit; color:var(--ink); background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:4px 8px; }
.key { display:inline-flex; gap:12px; color:var(--ink2); font-size:13px; } .key i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; vertical-align:-1px; }
main { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:12px; }
#wrap { position:relative; background:var(--surface); border:1px solid var(--border); border-radius:8px; height:min(78vh,900px); min-height:420px; overflow:hidden; }
canvas { display:block; width:100%; height:100%; cursor:grab; } canvas:active { cursor:grabbing; }
#tip { position:absolute; pointer-events:none; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:8px; width:300px;
  box-shadow:0 4px 16px rgba(0,0,0,.15); display:none; font-size:12.5px; z-index:2; }
#tip img { width:100%; max-height:200px; object-fit:contain; background:#000; border-radius:4px; display:block; margin-bottom:6px; }
#tip b { display:block; font-size:13.5px; } #tip .m { color:var(--ink2); }
#reset { position:absolute; right:10px; top:10px; }
aside { background:var(--surface); border:1px solid var(--border); border-radius:8px; display:flex; flex-direction:column; height:min(78vh,900px); min-height:420px; }
aside h2 { font-size:14px; margin:0; padding:10px 12px 4px; } aside p { color:var(--muted); font-size:12px; margin:0 12px 6px; }
#legend { overflow-y:auto; flex:1; padding:0 6px 8px; }
.cl { padding:6px; border-radius:6px; cursor:pointer; } .cl:hover, .cl.on { background:rgba(var(--hull),.06); }
.cl .nm { font-weight:600; font-size:13px; } .cl .meta { color:var(--ink2); font-size:12px; }
footer { color:var(--ink2); font-size:12.5px; } footer p { max-width:1100px; }
@media (max-width:900px) { header,.controls,main,footer { padding-left:16px; padding-right:16px; } main { grid-template-columns:1fr; } aside { height:420px; } }
</style></head><body>
<header><h1>Permissively licensed dermatology references: clusters</h1>
<p class="sub">__N__ clinical photographs (CC BY / CC0, no text annotations) from open-access articles in Derm1M and our corpus. Three maps of the
same images: by image similarity, by caption similarity, and by both. Shaded regions are HDBSCAN clusters, named by Qwen3.8-27B-Splash from captions and
sample images. Hover a point for the thumbnail. Intended to find coherent image groups for per-disease LoRA training.</p></header>
<div class="controls">
  <div class="tabs" role="group" aria-label="Embedding view"><button data-v="image">Image</button><button data-v="text">Caption</button><button data-v="joint" aria-pressed="true">Image + caption</button></div>
  <label>Colour by <select id="colour"><option value="fst">Fitzpatrick estimate</option><option value="origin">Source</option></select></label>
  <span class="key" id="key"></span>
  <input type="search" id="q" placeholder="Search diagnosis or caption…" size="26" aria-label="Search">
  <button class="btn" id="theme" type="button">◐ Theme</button>
</div>
<main><div id="wrap"><canvas id="map" aria-label="Cluster map"></canvas><div id="tip"></div><button class="btn" id="reset" type="button">Reset view</button></div>
<aside><h2>Clusters</h2><p id="lnote"></p><div id="legend"></div></aside></main>
<footer><p>Embeddings: CLIP ViT-B/16 fine-tuned on our corpus + Derm1M (image and caption towers share one space; joint = normalised sum).
Layout: PaCMAP (15 neighbours, FP ratio 8). Clusters: HDBSCAN (min. size 40; leaf selection for the image view). Fitzpatrick: CLIP estimator
(under-detects dark skin). Every image keeps its CC BY attribution in work/refs/permissive_refs.parquet.</p></footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const P = __DATA__, NAMES = __NAMES__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let view = "joint", colourBy = "fst", hits = new Set(), hoverC = null;
const cname = c => (NAMES[view + ":" + c] && NAMES[view + ":" + c].name) || ("Cluster " + (c + 1));
const fstGroup = f => f <= 2 ? 1 : f <= 4 ? 2 : 3;
const colour = p => colourBy === "fst" ? css("--s" + fstGroup(p.f)) : css(p.o ? "--s2" : "--s1");
function key() {
  const k = colourBy === "fst" ? [["--s1","Fitzpatrick I–II"],["--s2","III–IV"],["--s3","V–VI"]] : [["--s1","Derm1M (PubMed)"],["--s2","Our corpus"]];
  document.getElementById("key").innerHTML = k.map(([c, l]) => `<span><i style="background:var(${c})"></i>${l}</span>`).join("");
}
const wrap = document.getElementById("wrap"), canvas = document.getElementById("map"), ctx = canvas.getContext("2d");
let W = 0, H = 0, dpr = 1, T = d3.zoomIdentity, sx = d3.scaleLinear(), sy = d3.scaleLinear(), delaunay, hulls = [];
const X = p => T.applyX(sx(p[view][0])), Y = p => T.applyY(sy(p[view][1]));
function layout() {
  const q = (a, f) => d3.quantile(a, f), xs = P.map(p => p[view][0]).sort(d3.ascending), ys = P.map(p => p[view][1]).sort(d3.ascending);
  const xe = [q(xs, .01), q(xs, .99)], ye = [q(ys, .01), q(ys, .99)], pad = 24;
  const k = Math.min((W - 2 * pad) / (xe[1] - xe[0]), (H - 2 * pad) / (ye[1] - ye[0]));
  sx.domain(xe).range([(W - k * (xe[1] - xe[0])) / 2, (W + k * (xe[1] - xe[0])) / 2]);
  sy.domain(ye).range([(H + k * (ye[1] - ye[0])) / 2, (H - k * (ye[1] - ye[0])) / 2]);
  delaunay = d3.Delaunay.from(P, p => sx(p[view][0]), p => sy(p[view][1]));
  const by = d3.group(P.filter(p => p[view][2] >= 0), p => p[view][2]);
  hulls = [...by].map(([c, pts]) => { const h = pts.length > 2 ? d3.polygonHull(pts.map(p => [p[view][0], p[view][1]])) : null;
    return { c, h, cen: h ? d3.polygonCentroid(h) : [0, 0], size: pts.length, pts }; });
}
function draw() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, W, H);
  const hc = css("--hull");
  for (const h of hulls) { if (!h.h) continue;
    const cx = T.applyX(sx(h.cen[0])), cy = T.applyY(sy(h.cen[1]));
    const pts = h.h.map(([x, y]) => { const px = T.applyX(sx(x)), py = T.applyY(sy(y)), dx = px - cx, dy = py - cy, l = Math.hypot(dx, dy) || 1; return [px + dx / l * 8, py + dy / l * 8]; });
    ctx.beginPath(); d3.line().curve(d3.curveCatmullRomClosed.alpha(.5)).context(ctx)(pts);
    const on = hoverC === h.c; ctx.fillStyle = `rgba(${hc},${on ? .13 : .05})`; ctx.fill(); ctx.strokeStyle = `rgba(${hc},${on ? .45 : .12})`; ctx.stroke(); }
  const r = 1.6 * Math.min(Math.sqrt(T.k), 2.5), dim = hoverC !== null || hits.size;
  for (const p of P) { const x = X(p), y = Y(p); if (x < -5 || y < -5 || x > W + 5 || y > H + 5) continue;
    const focus = hits.size ? hits.has(p) : hoverC === null || p[view][2] === hoverC;
    ctx.globalAlpha = dim && !focus ? .12 : .85; ctx.fillStyle = colour(p); ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); }
  ctx.globalAlpha = 1;
  ctx.font = "600 12px system-ui,-apple-system,Arial,sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const boxes = [], minSize = T.k > 3 ? 0 : T.k > 1.6 ? 120 : 250;
  for (const h of [...hulls].sort((a, b) => (b.c === hoverC) - (a.c === hoverC) || b.size - a.size)) {
    if (!h.h || (h.size < minSize && h.c !== hoverC)) continue;
    const t = cname(h.c), x = T.applyX(sx(h.cen[0])), y = T.applyY(sy(h.cen[1])), w = ctx.measureText(t).width + 8, b = [x - w / 2, y - 9, x + w / 2, y + 9];
    if (boxes.some(o => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1])) continue; boxes.push(b);
    ctx.lineWidth = 4; ctx.strokeStyle = css("--surface"); ctx.lineJoin = "round"; ctx.strokeText(t, x, y); ctx.fillStyle = css("--ink"); ctx.fillText(t, x, y); }
}
function legend() {
  const L = document.getElementById("legend");
  const rows = [...hulls].sort((a, b) => b.size - a.size);
  document.getElementById("lnote").textContent = `${rows.length} clusters in the ${view === "joint" ? "image + caption" : view} view; ${P.filter(p => p[view][2] < 0).length.toLocaleString()} images unclustered.`;
  L.innerHTML = rows.map(h => { const n = NAMES[view + ":" + h.c] || {}; const dx = (n.dominant_diagnoses || []).slice(0, 3).join(", ");
    return `<div class="cl" data-c="${h.c}" tabindex="0"><div class="nm">${esc(cname(h.c))}</div><div class="meta">${h.size} images${dx ? " · " + esc(dx) : ""}</div></div>`; }).join("");
  L.querySelectorAll(".cl").forEach(el => { const c = +el.dataset.c;
    el.onmouseenter = () => { hoverC = c; draw(); }; el.onmouseleave = () => { hoverC = null; draw(); };
    el.onclick = el.onkeydown = e => { if (e.type === "click" || e.key === "Enter") zoomTo(hulls.find(h => h.c === c).pts); }; });
}
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const zoom = d3.zoom().scaleExtent([.4, 40]).on("zoom", e => { T = e.transform; tip.style.display = "none"; draw(); });
const sel = d3.select(canvas).call(zoom);
function zoomTo(pts) { const xs = pts.map(p => sx(p[view][0])), ys = pts.map(p => sy(p[view][1]));
  const x0 = d3.min(xs), x1 = d3.max(xs), y0 = d3.min(ys), y1 = d3.max(ys), k = Math.min(12, .8 / Math.max((x1 - x0) / W, (y1 - y0) / H, .002));
  sel.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2)); }
document.getElementById("reset").onclick = () => sel.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
const tip = document.getElementById("tip");
canvas.addEventListener("pointermove", e => {
  const rc = canvas.getBoundingClientRect(), mx = e.clientX - rc.left, my = e.clientY - rc.top;
  const i = delaunay.find(T.invertX(mx), T.invertY(my)), p = P[i];
  if (!p || Math.hypot(X(p) - mx, Y(p) - my) > 10) { tip.style.display = "none"; return; }
  const c = p[view][2];
  tip.innerHTML = `<img src="thumbs/${p.t}.jpg" alt=""><b>${esc(p.d || "no diagnosis label")}</b>
    <div class="m">${c >= 0 ? esc(cname(c)) : "unclustered"} · Fitzpatrick ~${p.f} · ${p.o ? "our corpus" : "Derm1M PubMed"} · ${esc(p.l)}</div>
    <div>${esc(p.cap)}</div>`;
  tip.style.display = "block";
  tip.style.left = Math.min(mx + 14, W - 310) + "px"; tip.style.top = (my + 14 + tip.offsetHeight > H ? Math.max(4, my - tip.offsetHeight - 10) : my + 14) + "px";
});
canvas.addEventListener("pointerleave", () => tip.style.display = "none");
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(x => x.setAttribute("aria-pressed", x === b)); view = b.dataset.v; hoverC = null;
  layout(); legend(); sel.call(zoom.transform, d3.zoomIdentity); draw(); });
document.getElementById("colour").onchange = e => { colourBy = e.target.value; key(); draw(); };
document.getElementById("q").addEventListener("input", e => { const q = e.target.value.trim().toLowerCase();
  hits = new Set(q.length >= 3 ? P.filter(p => (p.d || "").toLowerCase().includes(q) || (p.cap || "").toLowerCase().includes(q)) : []); draw(); });
document.getElementById("q").addEventListener("keydown", e => { if (e.key === "Enter" && hits.size) zoomTo([...hits]); });
document.getElementById("theme").onclick = () => { const r = document.documentElement;
  const dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; r.dataset.theme = dark ? "light" : "dark"; draw(); };
function resize() { dpr = window.devicePixelRatio || 1; W = wrap.clientWidth; H = wrap.clientHeight; canvas.width = W * dpr; canvas.height = H * dpr; layout(); draw(); }
new ResizeObserver(resize).observe(wrap);
key(); resize(); legend();
</script></body></html>
"""


def main():
    C = pd.read_parquet(D / "clusters.parquet")
    names = json.load(open(D / "cluster_names.json")) if (D / "cluster_names.json").exists() else {}
    names = {k: v for k, v in names.items() if "error" not in v}
    pts = []
    for r in C.itertuples():
        pts.append({"image": [r.x_image, r.y_image, int(r.c_image)], "text": [r.x_text, r.y_text, int(r.c_text)],
                    "joint": [r.x_joint, r.y_joint, int(r.c_joint)], "t": r.sha256[:16],
                    "d": None if not isinstance(r.diagnosis, str) else r.diagnosis[:80],
                    "cap": (r.caption or "")[:200] if isinstance(r.caption, str) else "",
                    "f": int(r.fst_pred) if r.fst_pred == r.fst_pred else 3, "o": int(str(r.origin).startswith("ours")),
                    "l": str(r.license_raw)})
    html = (TEMPLATE.replace("__DATA__", json.dumps(pts, ensure_ascii=False, separators=(",", ":")))
            .replace("__NAMES__", json.dumps(names, ensure_ascii=False, separators=(",", ":"))).replace("__N__", f"{len(C):,}"))
    (D / "index.html").write_text(html)
    print(D / "index.html", f"{len(html) / 1e6:.1f} MB", len(names), "cluster names")


if __name__ == "__main__":
    main()
