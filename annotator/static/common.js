/* Shared helpers for the annotator pages: storage, API, prompt parsing, auto-grow, swipe, zoom viewer. */
"use strict";
const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNarrow = () => window.matchMedia("(max-width: 900px)").matches;

async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}

function isTyping(el) {
  el = el || document.activeElement;
  if (!el) return false;
  const tag = (el.tagName || "").toLowerCase();
  return tag === "textarea" || tag === "input" || tag === "select" || el.isContentEditable;
}

/* ---------- prompt parsing ---------- */
function hl(text) {
  let t = esc(text);
  t = t.replace(/\(Fitzpatrick skin type [IVX]+\)/g, m => `<mark>${m}</mark>`);
  t = t.replace(/\b(\d{1,3}-year-old (?:man|woman|boy|girl|male|female|child|infant|person))\b/g, m => `<mark>${m}</mark>`);
  t = t.replace(/\b(of [A-Z][\w\- ]+? descent)\b/g, m => `<mark>${m}</mark>`);
  return t;
}
function parsePrompt(p) {
  const parts = []; let rest = (p || "").trim(); let m;
  const take = (re) => { const x = rest.match(re); if (x) { rest = rest.slice(x[0].length).trim(); return x; } return null; };
  if ((m = take(/^Clinical (?:dermatology )?photograph[\s\S]*?\.(?=\s|$)/i))) parts.push(["Style", m[0], "boiler"]);
  if ((m = take(/^Non-sexual medical photograph:[\s\S]*?\.(?=\s|$)/i))) parts.push(["Safety", m[0], "boiler"]);
  if ((m = take(/^Add this skin condition to the photograph\.\s*/i))) parts.push(["Edit", "Add this skin condition to the photograph.", "boiler"]);
  const keep = rest.search(/\bKeep the (?:person|subject|body|framing|image)/i);
  if (/^(Add|Make|Change|Turn|Place|Put|Show)\b/i.test(rest) && keep > 0) {
    parts.push(["Edit (lesion)", rest.slice(0, keep).trim(), "strong"]); parts.push(["Preserve", rest.slice(keep).trim(), "boiler"]); return parts;
  }
  let keepTail = null; const k2 = rest.search(/\bKeep the (?:person|subject|body|framing|image)/i);
  if (k2 > 0) { keepTail = rest.slice(k2).trim(); rest = rest.slice(0, k2).trim(); }
  if ((m = take(/^Healthy skin, no lesions:\s*/i))) parts.push(["Diagnosis", "Healthy skin, no lesions", "strong"]);
  const dxAt = rest.search(/\bDiagnosis:\s*/);
  if (dxAt >= 0) {
    const pre = rest.slice(0, dxAt).trim(); if (pre) parts.push(["Patient / view", pre, "hl"]);
    rest = rest.slice(dxAt).replace(/^Diagnosis:\s*/, "");
    const dm = rest.match(/^([\s\S]*?)(?:,\s*stage:\s*([^.]*))?\.(?=\s|$)/);
    if (dm) {
      rest = rest.slice(dm[0].length).trim(); const dv = dm[1].split(/,\s*/);
      parts.push(["Diagnosis", dv[0], "strong"]); if (dv.length > 1) parts.push(["Variant", dv.slice(1).join(", "), "strong"]);
      if (dm[2]) parts.push(["Stage", dm[2], "strong"]);
    }
  }
  const on = rest.search(/\bOn (?:this|these) [^:]{1,80}:\s*/);
  if (on >= 0) {
    const morph = rest.slice(0, on).trim(); if (morph) parts.push([dxAt >= 0 ? "Morphology" : "Patient / view", morph, dxAt >= 0 ? "" : "hl"]);
    const sm = rest.slice(on).match(/^On (?:this|these) ([^:]{1,80}):\s*([\s\S]*)$/);
    parts.push(["On this skin", sm ? sm[2] : rest.slice(on), "", sm ? sm[1] : ""]);
  } else if (rest) parts.push([dxAt >= 0 ? "Morphology" : "Patient / view", rest, dxAt >= 0 ? "" : "hl"]);
  if (keepTail) parts.push(["Preserve", keepTail, "boiler"]);
  return parts;
}
function promptPartsHTML(p) {
  const parts = parsePrompt(p);
  if (!parts.length) return `<div class="pparts"><div class="lab">—</div><div class="val boiler">(no prompt)</div></div>`;
  return `<div class="pparts">${parts.map(([l, v, c, sub]) =>
    `<div class="lab ${c === "boiler" ? "boiler" : ""}">${esc(l)}${sub ? `<span class="sub">(${esc(sub)})</span>` : ""}</div>` +
    `<div class="val ${c === "boiler" ? "boiler" : c === "strong" ? "strong" : ""}">${c === "hl" ? hl(v) : esc(v)}</div>`).join("")}</div>`;
}

/* ---------- textarea auto-grow ---------- */
function autoGrow(ta) {
  ta.style.height = "auto";
  const max = Math.max(160, window.innerHeight * 0.6);
  ta.style.height = Math.min(ta.scrollHeight + 2, max) + "px";
}
function attachAutoGrow(root) {
  (root || document).querySelectorAll("textarea[data-grow]").forEach(ta => {
    if (ta.dataset.growBound) return; ta.dataset.growBound = "1";
    ta.addEventListener("input", () => autoGrow(ta)); autoGrow(ta);
  });
}

/* ---------- horizontal swipe (ignored while a form field is focused) ---------- */
function onSwipe(el, cb) {
  let s = null;
  el.addEventListener("touchstart", e => { s = e.touches.length === 1 && !isTyping() ? { x: e.touches[0].clientX, y: e.touches[0].clientY, t: Date.now() } : null; }, { passive: true });
  el.addEventListener("touchend", e => {
    if (!s || isTyping()) { s = null; return; }
    const t = e.changedTouches[0], dx = t.clientX - s.x, dy = t.clientY - s.y, dt = Date.now() - s.t; s = null;
    if (Math.abs(dx) > 60 && Math.abs(dx) > 1.8 * Math.abs(dy) && dt < 800) cb(dx < 0 ? 1 : -1);
  }, { passive: true });
}

/* ---------- zoom viewer: full-resolution original, pinch / drag / wheel / double-tap ---------- */
const Zoom = (() => {
  let root, img, info, st = { s: 1, x: 0, y: 0, w: 0, h: 0 }, ptrs = new Map(), gesture = null, lastTap = 0, moved = false;
  const CSS = `#zoom{position:fixed;inset:0;background:rgba(8,8,8,.96);z-index:100;display:none;overflow:hidden;touch-action:none;user-select:none;-webkit-user-select:none}
#zoom.open{display:block} #zoom img{position:absolute;left:0;top:0;transform-origin:0 0;max-width:none;cursor:zoom-in;-webkit-user-drag:none}
#zoom .zbar{position:absolute;top:max(10px,env(safe-area-inset-top));right:10px;display:flex;gap:6px;z-index:2}
#zoom .zb{min-width:48px;min-height:44px;padding:0 12px;border-radius:10px;border:1px solid #555;background:rgba(30,30,30,.88);color:#eee;font:15px/1 -apple-system,system-ui,sans-serif;cursor:pointer}
#zoom .zclose{font-size:18px}
#zoom .zinfo{position:absolute;left:50%;transform:translateX(-50%);max-width:calc(100% - 20px);bottom:max(10px,env(safe-area-inset-bottom));color:#e6e6e6;background:rgba(0,0,0,.62);padding:3px 10px;border-radius:8px;font:12px/1.4 -apple-system,system-ui,sans-serif;text-align:center;pointer-events:none}
html.zoom-open,html.zoom-open body{overflow:hidden}`;
  function build() {
    const style = document.createElement("style"); style.textContent = CSS; document.head.appendChild(style);
    root = document.createElement("div"); root.id = "zoom";
    root.innerHTML = `<img id="zimg" alt="" draggable="false"><div class="zbar"><button class="zb" data-z="fit">Fit</button><button class="zb" data-z="1">100%</button><button class="zb" data-z="2">200%</button><button class="zb zclose" data-z="close" aria-label="Close">✕</button></div><div class="zinfo"></div>`;
    document.body.appendChild(root);
    img = root.querySelector("img"); info = root.querySelector(".zinfo");
    root.querySelectorAll(".zb").forEach(b => b.addEventListener("click", e => { e.stopPropagation(); const z = b.dataset.z; z === "close" ? close() : z === "fit" ? fit() : zoomTo(+z); }));
    root.addEventListener("pointerdown", down); root.addEventListener("pointermove", move);
    root.addEventListener("pointerup", up); root.addEventListener("pointercancel", up);
    root.addEventListener("wheel", e => { e.preventDefault(); zoomAt(st.s * Math.exp(-e.deltaY * 0.0015), e.clientX, e.clientY); }, { passive: false });
    window.addEventListener("resize", () => { if (isOpen()) clamp(); });
  }
  const isOpen = () => root && root.classList.contains("open");
  const fitScale = () => Math.min(root.clientWidth / st.w, (root.clientHeight) / st.h);
  function apply() { img.style.transform = `translate(${st.x}px, ${st.y}px) scale(${st.s})`;
    info.textContent = st.w ? `${st.w}×${st.h}px original · ${Math.round(st.s * 100)}% · pinch / scroll to zoom, drag to pan, double-tap: fit ↔ 100%` : "Loading full resolution…"; }
  function clamp() {
    const W = root.clientWidth, H = root.clientHeight, w = st.w * st.s, h = st.h * st.s;
    st.x = w <= W ? (W - w) / 2 : Math.min(0, Math.max(W - w, st.x));
    st.y = h <= H ? (H - h) / 2 : Math.min(0, Math.max(H - h, st.y));
    apply();
  }
  function zoomAt(s, cx, cy) {
    s = Math.max(fitScale() * 0.5, Math.min(8, s));
    st.x = cx - (cx - st.x) * (s / st.s); st.y = cy - (cy - st.y) * (s / st.s); st.s = s; clamp();
  }
  function zoomTo(s, cx, cy) { zoomAt(s, cx ?? root.clientWidth / 2, cy ?? root.clientHeight / 2); }
  function fit() { st.s = fitScale(); st.x = 0; st.y = 0; clamp(); }
  function down(e) {
    if (e.target.closest(".zbar")) return;
    root.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    moved = false; startGesture();
  }
  function startGesture() {
    const p = [...ptrs.values()];
    if (p.length === 1) gesture = { kind: "pan", x0: p[0].x, y0: p[0].y, sx: st.x, sy: st.y };
    else if (p.length >= 2) {
      const d = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y), mx = (p[0].x + p[1].x) / 2, my = (p[0].y + p[1].y) / 2;
      gesture = { kind: "pinch", d0: d, s0: st.s, mx, my, sx: st.x, sy: st.y };
    }
  }
  function move(e) {
    if (!ptrs.has(e.pointerId) || !gesture) return;
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY }); const p = [...ptrs.values()];
    if (gesture.kind === "pan" && p.length === 1) {
      const dx = p[0].x - gesture.x0, dy = p[0].y - gesture.y0; if (Math.abs(dx) + Math.abs(dy) > 6) moved = true;
      st.x = gesture.sx + dx; st.y = gesture.sy + dy; clamp();
    } else if (gesture.kind === "pinch" && p.length >= 2) {
      moved = true;
      const d = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y), mx = (p[0].x + p[1].x) / 2, my = (p[0].y + p[1].y) / 2;
      const s = Math.max(fitScale() * 0.5, Math.min(8, gesture.s0 * d / gesture.d0)), k = s / gesture.s0;
      st.s = s; st.x = mx - (gesture.mx - gesture.sx) * k; st.y = my - (gesture.my - gesture.sy) * k; clamp();
    }
  }
  function up(e) {
    if (!ptrs.has(e.pointerId)) return;
    ptrs.delete(e.pointerId);
    if (ptrs.size) { startGesture(); return; }
    gesture = null;
    if (moved || e.type === "pointercancel") return;
    const onImg = e.target === img;
    if (e.pointerType === "mouse") {
      if (!onImg) { close(); return; }
      const f = fitScale(); const next = st.s < 0.99 ? 1 : st.s < 1.99 ? 2 : f;  // click cycles fit -> 100% -> 200% -> fit
      zoomTo(next, e.clientX, e.clientY);
    } else {
      const now = Date.now();
      if (now - lastTap < 320) { lastTap = 0; const f = fitScale(); zoomTo(Math.abs(st.s - f) < 0.02 ? Math.max(1, f * 2) : f, e.clientX, e.clientY); }
      else lastTap = now;
    }
  }
  function open(src) {
    if (!root) build();
    st = { s: 1, x: 0, y: 0, w: 0, h: 0 }; ptrs.clear(); gesture = null;
    root.classList.add("open"); document.documentElement.classList.add("zoom-open");
    img.style.visibility = "hidden"; apply();
    img.onload = () => { st.w = img.naturalWidth; st.h = img.naturalHeight; img.style.width = st.w + "px"; img.style.height = st.h + "px";
      const coarse = window.matchMedia("(pointer: coarse)").matches;   // touch: start fitted; mouse: start at 100%
      st.s = coarse ? fitScale() : Math.max(fitScale(), 1); st.x = 0; st.y = 0; clamp(); img.style.visibility = ""; };
    img.src = src;
  }
  function close() { if (!root) return; root.classList.remove("open"); document.documentElement.classList.remove("zoom-open"); img.removeAttribute("src"); }
  document.addEventListener("keydown", e => { if (isOpen() && e.key === "Escape") { e.preventDefault(); e.stopImmediatePropagation(); close(); } }, true);
  return { open, close, isOpen };
})();
