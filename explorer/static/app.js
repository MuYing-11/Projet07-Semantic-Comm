/* Explorateur Projet 07 — interface. Toute la physique vient du serveur (semcom.*) ; ici on ne fait que
   dessiner la chaîne, avancer d'étape en étape et tracer les courbes. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const SVGNS = "http://www.w3.org/2000/svg";
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

const S = {
  cat: null, trace: null, i: 0, timer: null, playing: false,
  cmpSel: new Set(["sep", "jscc_mse", "jscc_sem"]), chartOff: new Set(), chart: null, redraw: [],
};

/* ------------------------------------------------------------------ glossaire (les concepts de cours) */
const GLOSS = {
  kn: ["Rapport de bande k/n",
    "k = nombre de symboles complexes émis, n = 3·H·W valeurs réelles de l'image source. k/n = 1/12 veut dire : " +
    "un symbole radio pour douze nombres d'image. C'est le budget commun à tous les systèmes comparés — sans " +
    "lui, la comparaison n'aurait aucun sens.", "semcom/channel.py · ratio_from_channels()"],
  power: ["Contrainte de puissance",
    "La puissance moyenne est ramenée à 1 par symbole complexe. Un émetteur ne peut pas simplement crier plus " +
    "fort pour couvrir le bruit ; sans cette normalisation, un réseau appris gagnerait en augmentant " +
    "l'amplitude, ce qui ne serait pas réalisable physiquement.", "semcom/channel.py · power_normalize()"],
  awgn: ["Canal AWGN",
    "Bruit blanc gaussien additif : y = x + n, avec n tiré d'une gaussienne de variance σ² = 10^(−SNR/10). " +
    "C'est le canal thermique idéal, sans obstacle ni mobilité. Le SNR en dB vaut 10·log10(P_signal/P_bruit) : " +
    "0 dB = autant de bruit que de signal, 20 dB = cent fois moins de bruit.", "semcom/channel.py · awgn()"],
  rayleigh: ["Évanouissement de Rayleigh par blocs",
    "Modèle d'un canal mobile : le signal arrive par plusieurs trajets réfléchis qui s'additionnent avec des " +
    "phases différentes, donc l'amplitude reçue devient aléatoire. y = h·x + n avec h ~ CN(0,1), constant sur " +
    "toute l'image (« par blocs »). Le récepteur connaît h (CSI parfaite) et divise par h : c'est l'égalisation " +
    "zero-forcing, qui ramène le problème à un AWGN de SNR·|h|². Quand |h| est petit, c'est un évanouissement " +
    "profond et tout le bloc en souffre.", "semcom/channel.py · rayleigh_block()"],
  shannon: ["Théorème de séparation de Shannon",
    "On peut, sans perte d'optimalité, séparer le problème en codage de source (compresser) puis codage de " +
    "canal (protéger). C'est le socle de JPEG + Wi-Fi, H.264 + 4G… Mais le résultat est asymptotique : il " +
    "suppose des blocs infiniment longs et un canal dont on connaît la statistique. Les images sont courtes et " +
    "le SNR réel diffère du SNR prévu : c'est cette faille que le JSCC exploite.", "semcom/digital.py"],
  conv: ["Code convolutif K=7 et poinçonnage",
    "Chaque bit émis dépend des 7 derniers bits d'entrée (polynômes 133/171 en octal, ceux du Wi-Fi 802.11a). " +
    "Le rendement 1/2 est ramené à 2/3 ou 3/4 par poinçonnage : on supprime des bits de redondance selon un " +
    "motif fixe. Chaque paquet (au plus 1024 bits utiles) est terminé par 6 bits de queue pour ramener le " +
    "treillis à l'état zéro.", "semcom/digital.py · conv_encode(), puncture()"],
  qam: ["Modulation QAM et codage de Gray",
    "Les bits sont groupés et placés sur les points d'une constellation du plan complexe : QPSK = 2 bits/" +
    "symbole, 16-QAM = 4, 64-QAM = 6. Plus de bits par symbole = plus de débit, mais des points plus serrés, " +
    "donc plus sensibles au bruit. Le codage de Gray numérote les points pour qu'une confusion entre voisins " +
    "ne coûte qu'un seul bit faux.", "semcom/digital.py · qam_mod()"],
  llr: ["Décisions souples (LLR)",
    "Le démodulateur ne tranche pas 0/1 : pour chaque bit il calcule un log-rapport de vraisemblance " +
    "log P(0)/P(1) (approximation max-log). Le signe donne le bit le plus probable, l'amplitude donne la " +
    "confiance. Un LLR proche de zéro veut dire « je ne sais pas ». Ces décisions souples valent environ 2 dB " +
    "par rapport à des décisions dures.", "semcom/digital.py · qam_llr()"],
  viterbi: ["Décodage de Viterbi",
    "Le décodage optimal du code convolutif : on cherche le chemin le plus vraisemblable dans le treillis à " +
    "2^6 = 64 états, en accumulant les LLR. Tant que le canal reste au-dessus du seuil du code, la correction " +
    "est quasi parfaite ; en dessous, le décodeur choisit le mauvais chemin et les erreurs passent en masse.",
    "semcom/digital.py · viterbi()"],
  cliff: ["Effet falaise (cliff effect)",
    "Une chaîne numérique est conçue pour un SNR cible. Au-dessus, elle sature et n'exploite pas la marge " +
    "disponible. En dessous, le décodeur décroche, le fichier devient illisible et l'image est perdue d'un " +
    "coup — pas dégradée, perdue (ici : une image grise uniforme). Le JSCC n'a pas ce comportement : sa " +
    "dégradation est progressive.", "scripts/evaluate.py · figures/cifar_snr.png"],
  jscc: ["Deep JSCC (codage conjoint source-canal appris)",
    "Un encodeur CNN produit directement les symboles analogiques, le canal est inséré comme une couche du " +
    "réseau, et un décodeur CNN reconstruit l'image. Comme le bruit traverse la rétropropagation, le réseau " +
    "apprend un code adapté au canal. L'entraînement tire le SNR dans [0, 20] dB et la bande au hasard : un " +
    "seul modèle couvre toutes les conditions.", "semcom/models.py · DeepJSCC"],
  judge: ["Juges indépendants",
    "La perte sémantique utilise le classifieur A (ResNet). Le juge, lui, est le classifieur B (VGG, autre " +
    "architecture) et un ResNet-18 ImageNet — jamais utilisés dans aucune perte. Sinon le système apprendrait " +
    "à flatter son propre correcteur et les chiffres ne voudraient rien dire.", "semcom/metrics.py"],
  task: ["Transmission orientée tâche",
    "Si le récepteur ne veut que la décision, pourquoi reconstruire l'image ? L'encodeur envoie directement " +
    "vers une tête de classification, sans décodeur image : k/n descend à 1/384 pour une précision comparable " +
    "au JSCC complet. C'est le point extrême de la communication sémantique.", "semcom/models.py · TaskJSCC"],
};

const SERIES_COLOR = (label) => {
  if (label.includes("sémantique")) return "#d87a4a";
  if (label.includes("syntaxique")) return "#58b0ea";
  if (label.includes("tâche")) return "#5fd0a6";
  if (label.includes("entraîné 0")) return "#b9a6f2";
  if (label.includes("entraîné 10")) return "#8f7ad8";
  if (label.includes("entraîné 20")) return "#6150a8";
  if (label.includes("capacité")) return "#9aa7bb";
  if (label.includes("meilleur MCS")) return "#e5706b";
  return "#55606f";
};

/* ------------------------------------------------------------------------------------------- réseau */
async function api(path, body) {
  const r = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

/* --------------------------------------------------------- état dans l'URL (liens partageables) */
function readHash() {
  const out = {};
  for (const kv of location.hash.replace(/^#/, "").split("&")) {
    const [k, v] = kv.split("=");
    if (k) out[decodeURIComponent(k)] = decodeURIComponent(v || "");
  }
  return out;
}

function writeHash() {
  const s = settings();
  const tab = ($(".tab.is-on") || {}).dataset?.tab || "explore";
  const q = [`system=${s.system}`, `image=${s.image}`, `snr=${s.snr}`, `c=${s.c}`, `channel=${s.channel}`,
    `mcs=${encodeURIComponent(s.mcs)}`, `seed=${s.seed}`, `stage=${S.i}`, `tab=${tab}`].join("&");
  history.replaceState(null, "", "#" + q);
}

function applyHash(h) {
  const set = (sel, v) => { if (v != null && v !== "") $(sel).value = v; };
  set("#imgSel", h.image); set("#snr", h.snr); set("#cSel", h.c);
  set("#chanSel", h.channel); set("#mcsSel", h.mcs); set("#seed", h.seed);
  $("#snrOut").textContent = $("#snr").value + " dB";
  if (h.system) {
    const b = $(`.sysitem[data-sys="${h.system}"]:not([disabled])`);
    if (b) { $$(".sysitem").forEach(e => e.classList.remove("is-on")); b.classList.add("is-on"); }
  }
  if (h.tab) { const t = $(`.tab[data-tab="${h.tab}"]`); if (t) t.click(); }
}

/* ------------------------------------------------------------------------------------------ amorçage */
async function boot() {
  S.booting = true;
  S.cat = await api("/api/catalog");
  $("#devBadge").textContent = `torch ${S.cat.torch} · ${S.cat.device}`;
  buildSystems();
  buildControls();
  buildLearn();
  buildCompareChips();
  wire();
  const h = readHash();
  applyHash(h);
  await run();
  if (h.stage) show(+h.stage);
  loadResults();
  if (h.tab === "compare") compare();
  S.booting = false;
}

function buildSystems() {
  const box = $("#sysList");
  box.innerHTML = "";
  let group = null;
  for (const s of S.cat.systems) {
    if (s.group !== group) {
      group = s.group;
      const h = document.createElement("div");
      h.className = "sysgroup";
      h.textContent = group;
      box.append(h);
    }
    const b = document.createElement("button");
    b.className = "sysitem";
    b.dataset.sys = s.id;
    b.dataset.sem = s.id === "jscc_sem" ? "1" : "0";
    b.innerHTML = `${s.label}<small>${s.available ? s.short : "modèle manquant : " + s.missing.join(", ")}</small>`;
    b.disabled = !s.available;
    b.onclick = () => { $$(".sysitem").forEach(e => e.classList.remove("is-on")); b.classList.add("is-on"); run(); };
    box.append(b);
  }
  const first = $(".sysitem[data-sys='jscc_sem']:not([disabled])") || $(".sysitem:not([disabled])");
  if (first) first.classList.add("is-on");
}

function sysId() { return ($(".sysitem.is-on") || {}).dataset?.sys || "jscc_sem"; }
function sysDef(id) { return S.cat.systems.find(s => s.id === id) || {}; }

function buildControls() {
  const img = $("#imgSel");
  let g = null;
  for (const im of S.cat.images) {
    if (im.group !== g) { g = im.group; img.append(Object.assign(document.createElement("optgroup"), { label: g })); }
    const o = new Option(im.label, im.key);
    img.lastChild.append(o);
  }
  $("#chanSel").append(...S.cat.channels.map(c => new Option(c.label, c.id)));
  $("#cSel").append(...S.cat.widths.map(w => new Option(`${w.ratio}  (${w.c} canaux latents)`, w.c)));
  $("#cSel").value = "8";
  $("#mcsSel").append(...S.cat.mcs.map(m => new Option(m, m)));
  $("#resMetric").append(...S.cat.metrics.cifar.map(m => new Option(m.label, m.id)));
}

function buildCompareChips() {
  const box = $("#cmpPick");
  box.innerHTML = "";
  for (const s of S.cat.systems.filter(x => x.available)) {
    const b = document.createElement("button");
    b.className = "chk" + (S.cmpSel.has(s.id) ? " is-on" : "");
    b.innerHTML = `<span>${S.cmpSel.has(s.id) ? "✓" : "+"}</span> ${s.short}`;
    b.onclick = () => {
      S.cmpSel.has(s.id) ? S.cmpSel.delete(s.id) : S.cmpSel.add(s.id);
      buildCompareChips();
    };
    box.append(b);
  }
}

function buildLearn() {
  $("#learnList").innerHTML = Object.entries(GLOSS).map(([, [t, d, f]]) =>
    `<article class="lcard"><h3>${t}</h3><p>${d}</p><span class="f">${f}</span></article>`).join("");
}

/* -------------------------------------------------------------------------------------- lancer un run */
function settings() {
  return {
    system: sysId(), image: $("#imgSel").value, snr: +$("#snr").value, c: +$("#cSel").value,
    channel: $("#chanSel").value, mcs: $("#mcsSel").value, seed: +$("#seed").value,
  };
}

async function run() {
  pause();
  const def = sysDef(sysId());
  $("#mcsCtl").hidden = !def.mcs;
  $("#runBtn").disabled = true;
  $("#runHint").innerHTML = '<span class="spin"></span> transmission en cours…';
  try {
    S.trace = await api("/api/run", settings());
    S.i = 0;
    $("#sysBlurb").innerHTML = `<b>${S.trace.label}</b> — ${S.trace.blurb}`;
    $("#scrub").max = String(S.trace.stages.length - 1);
    renderStageList();
    show(0);
    $("#runHint").textContent = `${S.trace.image_title} · k/n = ${fmtRatio(S.trace.summary.ratio)} · ` +
      `${S.trace.stages.length} étapes`;
  } catch (e) {
    $("#runHint").textContent = "⚠ " + e.message;
    $("#inspector").innerHTML = `<p class="placeholder">${e.message}</p>`;
  } finally {
    $("#runBtn").disabled = false;
  }
}

const fmtRatio = (r) => r ? `1/${Math.round(1 / r)}` : "—";

/* ------------------------------------------------------------------------------------------ la carte */
function renderMap() {
  const svg = $("#map");
  const t = S.trace;
  if (!t) return;
  svg.innerHTML = "";
  const W = 1240, n = t.nodes.length;
  const span = n > 1 ? t.nodes[1].x - t.nodes[0].x : W;
  const bw = Math.max(86, Math.min(136, span - 16)), bh = 56;
  const active = t.stages[S.i].node;
  const order = t.nodes.map(x => x.id);
  const ai = order.indexOf(active);
  const mk = (tag, at, parent = svg) => {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(at)) e.setAttribute(k, v);
    parent.append(e);
    return e;
  };

  // wires
  t.links.forEach(([a, b], idx) => {
    const A = t.nodes.find(x => x.id === a), B = t.nodes.find(x => x.id === b);
    const x1 = A.x + bw / 2, x2 = B.x - bw / 2;
    mk("line", { x1, y1: A.y, x2, y2: B.y, class: "wire" + (idx < ai ? " done" : "") });
    for (let d = 0; d < 3; d++) {
      mk("circle", { cx: x1 + (x2 - x1) * (0.3 + 0.2 * d), cy: A.y, r: 1.7,
        fill: idx < ai ? "var(--tx)" : "var(--line2)" });
    }
  });

  // nodes
  t.nodes.forEach((nd, idx) => {
    const col = nd.side === "ch" ? "var(--ch)" : nd.side === "rx" ? "var(--rx)" : "var(--tx)";
    const on = nd.id === active;
    const g = mk("g", {});
    mk("rect", { x: nd.x - bw / 2, y: nd.y - bh / 2, width: bw, height: bh, rx: 5,
      class: "nd-box" + (on ? " on" : ""), stroke: on ? col : "var(--line2)" }, g);
    if (on) {
      mk("rect", { x: nd.x - bw / 2 - 4, y: nd.y - bh / 2 - 4, width: bw + 8, height: bh + 8, rx: 7,
        fill: "none", stroke: col, "stroke-opacity": .33, "stroke-width": 1.4 }, g);
    }
    const lines = wrap(nd.label, Math.floor(bw / 7.2));
    lines.forEach((ln, k) => mk("text", { x: nd.x, y: nd.y - 6 + k * 13 - (lines.length - 1) * 6,
      class: "nd-lab", "text-anchor": "middle", "font-size": bw < 110 ? 11.5 : 13 }, g).textContent = ln);
    mk("text", { x: nd.x, y: nd.y + bh / 2 - 8, class: "nd-sub", "text-anchor": "middle",
      "font-size": bw < 110 ? 9.5 : 10.5 }, g).textContent = nd.sub;
    mk("text", { x: nd.x - bw / 2 + 7, y: nd.y - bh / 2 + 13, class: "nd-num", fill: col }, g)
      .textContent = String(idx + 1);
    // side band
    mk("rect", { x: nd.x - bw / 2, y: nd.y + bh / 2 - 3, width: bw, height: 3, rx: 1.5, fill: col,
      "fill-opacity": on ? .95 : .35 }, g);
  });

  // headers TX / canal / RX
  const bands = [["ÉMETTEUR", "tx"], ["CANAL", "ch"], ["RÉCEPTEUR", "rx"]];
  for (const [lab, side] of bands) {
    const ns = t.nodes.filter(x => x.side === side);
    if (!ns.length) continue;
    const x0 = Math.min(...ns.map(x => x.x)) - bw / 2, x1 = Math.max(...ns.map(x => x.x)) + bw / 2;
    const col = side === "ch" ? "var(--ch)" : side === "rx" ? "var(--rx)" : "var(--tx)";
    mk("line", { x1: x0, y1: 24, x2: x1, y2: 24, stroke: col, "stroke-opacity": .35, "stroke-width": 1 });
    mk("text", { x: (x0 + x1) / 2, y: 18, "text-anchor": "middle", fill: col, "fill-opacity": .8,
      style: "font:600 10px var(--sans);letter-spacing:.16em" }).textContent = lab;
  }

  // moving packet into the active node
  if (ai > 0) {
    const A = t.nodes[ai - 1], B = t.nodes[ai];
    const x1 = A.x + bw / 2, x2 = B.x - bw / 2;
    const p = mk("circle", { cx: x1, cy: A.y, r: 5, class: "pkt" });
    mk("animate", { attributeName: "cx", from: x1, to: x2, dur: reduce ? "0.01s" : "0.75s",
      begin: "0s", fill: "freeze", repeatCount: "1" }, p);
    mk("animate", { attributeName: "r", values: "5;7;5", dur: "0.75s", fill: "freeze" }, p);
  }

  // stage caption under the active node
  const nd = t.nodes.find(x => x.id === active);
  if (nd) {
    const txt = mk("text", { x: nd.x, y: nd.y + bh / 2 + 24, "text-anchor": "middle",
      style: "font:600 12px var(--sans)", fill: "var(--fg)" });
    txt.textContent = `${S.i + 1}. ${t.stages[S.i].title}`;
    const sz = mk("text", { x: nd.x, y: nd.y + bh / 2 + 40, "text-anchor": "middle",
      style: "font:10.5px var(--mono)", fill: "var(--muted)" });
    sz.textContent = t.stages[S.i].size || "";
  }
}

function wrap(s, max) {
  if (s.length <= max) return [s];
  const w = s.split(" "), out = [""];
  for (const x of w) {
    if ((out[out.length - 1] + " " + x).trim().length > max && out[out.length - 1]) out.push(x);
    else out[out.length - 1] = (out[out.length - 1] + " " + x).trim();
  }
  return out.slice(0, 2);
}

/* ------------------------------------------------------------------------------------- liste d'étapes */
function renderStageList() {
  const ol = $("#stageList");
  ol.innerHTML = "";
  S.trace.stages.forEach((st, k) => {
    const li = document.createElement("li");
    li.className = "stagerow";
    li.dataset.k = k;
    li.innerHTML = `<span class="sr-n">${String(k + 1).padStart(2, "0")}</span>
      <span><span class="sr-t">${st.title}</span><span class="sr-m">${st.size || ""}</span>
      <span class="sr-k">${st.kind}</span></span>`;
    li.onclick = () => { pause(); show(k); };
    ol.append(li);
  });
}

/* ---------------------------------------------------------------------------------------- inspecteur */
function show(i) {
  if (!S.trace) return;
  S.i = Math.max(0, Math.min(S.trace.stages.length - 1, i));
  const st = S.trace.stages[S.i];
  $("#scrub").value = String(S.i);
  $("#count").textContent = `${S.i + 1} / ${S.trace.stages.length}`;
  $$(".stagerow").forEach((r, k) => {
    r.classList.toggle("is-on", k === S.i);
    r.classList.toggle("past", k < S.i);
  });
  const row = $(`.stagerow[data-k="${S.i}"]`);
  if (row && !S.booting) row.scrollIntoView({ block: "nearest" });  // au chargement, ne pas sauter la page
  renderMap();
  renderInspector(st);
  $("#footState").textContent = `${S.trace.system} · ${S.trace.channel} · SNR ${S.trace.snr} dB · ` +
    `k/n ${fmtRatio(S.trace.summary.ratio)}`;
  writeHash();
}

function renderInspector(st) {
  const box = $("#inspector");
  box.innerHTML = "";
  S.redraw = [];  // the canvases of the previous stage are gone; drop their redraw callbacks
  const head = document.createElement("div");
  head.className = "ins-head";
  head.innerHTML = `<h3>${st.title}</h3><span class="ins-kind">${st.kind}</span>
    <span class="ins-size">${st.size || ""}</span>`;
  box.append(head);
  if (st.why) {
    const p = document.createElement("p");
    p.className = "ins-why";
    p.textContent = st.why;
    box.append(p);
  }
  if (st.concept && GLOSS[st.concept]) {
    const c = document.createElement("button");
    c.className = "chip";
    c.textContent = `concept : ${GLOSS[st.concept][0]}`;
    c.onclick = (e) => popover(e.currentTarget, GLOSS[st.concept]);
    box.append(c);
  }
  const views = document.createElement("div");
  views.className = "views";
  box.append(views);
  for (const v of st.views || []) views.append(renderView(v));
}

function renderView(v) {
  const d = document.createElement("div");
  d.className = "view" + (["latent", "constellation", "bits", "llr"].includes(v.type) ? " wide" : "");
  if (v.title) d.innerHTML = `<h4>${v.title}</h4>`;
  const body = document.createElement("div");
  d.append(body);
  ({
    image: () => { body.innerHTML = `<img src="${v.png}" alt="">`; },
    metrics: () => {
      body.innerHTML = `<div class="mgrid">${v.items.map(it => `<div class="mrow ${it.good ? "good" : ""}
        ${it.bad ? "bad" : ""}"><span class="k">${it.k}${it.hint ? `<span class="h">${it.hint}</span>` : ""}</span>
        <b>${it.v}</b></div>`).join("")}</div>`;
    },
    bars: () => {
      body.innerHTML = v.items.map(it => `<div class="bar ${it.truth ? "truth" : ""}">
        <span class="t">${it.label}</span><span class="g"><i style="width:${(it.value * 100).toFixed(1)}%"></i></span>
        <span class="v">${(it.value * 100).toFixed(1)}%</span></div>`).join("");
    },
    bits: () => {
      const errs = new Set(v.errors || []);
      let html = "";
      for (let i = 0; i < v.bits.length; i += 8) {
        const byte = v.bits.slice(i, i + 8).split("").map((b, j) =>
          errs.has(i + j) ? `<b>${b}</b>` : b).join("");
        html += byte + " ";
      }
      body.innerHTML = `<div class="bits">${html}</div>`;
      const info = [`${v.total.toLocaleString("fr-FR")} bits au total`,
        v.shown < v.total ? `${v.shown} premiers affichés` : null,
        v.n_errors != null ? `${v.n_errors} bit(s) faux · BER ${(v.ber * 100).toFixed(3)} %` : null]
        .filter(Boolean).join(" · ");
      body.insertAdjacentHTML("beforeend", `<p class="note">${info}</p>`);
    },
    llr: () => body.append(llrCanvas(v)),
    constellation: () => body.append(constCanvas(v)),
    latent: () => body.append(latentCanvas(v)),
  }[v.type] || (() => { body.textContent = JSON.stringify(v).slice(0, 400); }))();
  if (v.note) body.insertAdjacentHTML("beforeend", `<p class="note">${v.note}</p>`);
  return d;
}

/* ------------------------------------------------------------- vue : constellation (plan I/Q) */
function constCanvas(v) {
  const wrapEl = document.createElement("div");
  const cv = document.createElement("canvas");
  wrapEl.append(cv);
  const draw = () => {
    if (!v.points.length) return;
    const w = Math.max(260, wrapEl.clientWidth || 300), h = Math.min(330, Math.max(230, w * 0.72));
    const dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr; cv.style.width = w + "px"; cv.style.height = h + "px";
    const g = cv.getContext("2d");
    g.scale(dpr, dpr);
    g.fillStyle = "#0b0f14"; g.fillRect(0, 0, w, h);
    const all = (v.ref || []).concat(v.points);
    const m = Math.max(0.6, ...all.flat().map(Math.abs)) * 1.12;
    const X = (x) => w / 2 + x / m * (w / 2 - 26), Y = (y) => h / 2 - y / m * (h / 2 - 18);
    // grid
    g.strokeStyle = "#1e2733"; g.lineWidth = 1;
    for (let t = -Math.ceil(m); t <= Math.ceil(m); t++) {
      g.beginPath(); g.moveTo(X(t), 0); g.lineTo(X(t), h); g.stroke();
      g.beginPath(); g.moveTo(0, Y(t)); g.lineTo(w, Y(t)); g.stroke();
    }
    g.strokeStyle = "#39465a"; g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(0, Y(0)); g.lineTo(w, Y(0)); g.moveTo(X(0), 0); g.lineTo(X(0), h); g.stroke();
    g.fillStyle = "#6b7889"; g.font = "10px ui-monospace,monospace";
    g.fillText("I", w - 12, Y(0) - 6); g.fillText("Q", X(0) + 6, 12);
    g.fillText(`±${m.toFixed(2)}`, 6, h - 6);
    // sent symbols in grey
    if (v.ref) {
      g.fillStyle = "rgba(147,160,180,.5)";
      for (const [x, y] of v.ref) { g.beginPath(); g.arc(X(x), Y(y), 1.6, 0, 7); g.fill(); }
    }
    g.fillStyle = v.ref ? "rgba(224,163,60,.8)" : "rgba(88,176,234,.75)";
    for (const [x, y] of v.points) { g.beginPath(); g.arc(X(x), Y(y), 1.8, 0, 7); g.fill(); }
  };
  S.redraw.push(draw);
  requestAnimationFrame(draw);
  const info = document.createElement("p");
  info.className = "note";
  info.textContent = `${v.total.toLocaleString("fr-FR")} symboles complexes` +
    (v.points.length < v.total ? ` (${v.points.length} tirés au hasard pour le dessin)` : "") +
    (v.power != null ? ` · puissance moyenne mesurée ${v.power}` : "");
  wrapEl.append(info);
  return wrapEl;
}

/* ------------------------------------------------------------- vue : LLR (décisions souples) */
function llrCanvas(v) {
  const wrapEl = document.createElement("div");
  const cv = document.createElement("canvas");
  wrapEl.append(cv);
  const draw = () => {
    if (!v.values.length) return;
    const w = Math.max(260, wrapEl.clientWidth || 300), h = 150, dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr; cv.style.width = w + "px"; cv.style.height = h + "px";
    const g = cv.getContext("2d");
    g.scale(dpr, dpr);
    g.fillStyle = "#0b0f14"; g.fillRect(0, 0, w, h);
    const vals = v.values, m = Math.max(1, ...vals.map(Math.abs));
    const bw = Math.max(1.2, w / vals.length - 1);
    g.strokeStyle = "#39465a"; g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
    vals.forEach((x, i) => {
      const px = i * (w / vals.length), py = (x / m) * (h / 2 - 8);
      g.fillStyle = x >= 0 ? "rgba(88,176,234,.85)" : "rgba(229,112,107,.85)";
      g.fillRect(px, h / 2 - Math.max(0, py), bw, Math.abs(py) || 1);
    });
    g.fillStyle = "#6b7889"; g.font = "10px ui-monospace,monospace";
    g.fillText("plutôt 0 ↑", 4, 12); g.fillText("plutôt 1 ↓", 4, h - 5);
    g.fillText(`±${m.toFixed(1)}`, w - 44, 12);
  };
  S.redraw.push(draw);
  requestAnimationFrame(draw);
  return wrapEl;
}

/* ------------------------------------------------------------- vue : latent 3D (plans empilés) */
function latentCanvas(v) {
  const wrapEl = document.createElement("div");
  const cv = document.createElement("canvas");
  wrapEl.append(cv);
  const ctl = document.createElement("div");
  ctl.className = "ctrl-row";
  ctl.innerHTML = `<span>vue</span><input type="range" min="0" max="100" value="46" aria-label="inclinaison">
    <span id="lz">3D empilée</span>`;
  wrapEl.append(ctl);
  const slider = ctl.querySelector("input");
  const draw = () => {
    const tilt = +slider.value / 100;                   // 0 = grille plate, 1 = empilement vu de biais
    const w = Math.max(260, wrapEl.clientWidth || 320), h = 320, dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr; cv.style.width = w + "px"; cv.style.height = h + "px";
    const g = cv.getContext("2d");
    g.scale(dpr, dpr);
    g.fillStyle = "#0b0f14"; g.fillRect(0, 0, w, h);
    const C = v.maps.length, mw = Math.max(...v.maps.map(m => m.w)), mh = Math.max(...v.maps.map(m => m.h));
    const cellOf = (mp, active, i, sx, sy, shear, ox, oy) => {
      for (let y = 0; y < mp.h; y++) {
        for (let x = 0; x < mp.w; x++) {
          const u = mp.v[y * mp.w + x] / 255;
          g.fillStyle = active
            ? `rgb(${Math.round(16 + 72 * u)},${Math.round(48 + 152 * u)},${Math.round(78 + 172 * u)})`
            : `rgba(${Math.round(68 + 52 * u)},${Math.round(74 + 48 * u)},${Math.round(86 + 42 * u)},.45)`;
          g.fillRect(ox + x * sx + y * shear, oy + y * sy, sx + 0.6, sy + 0.6);
        }
      }
      g.strokeStyle = active ? "rgba(88,176,234,.75)" : "rgba(107,120,137,.3)";
      g.lineWidth = active ? 1.1 : 0.7;
      g.beginPath();
      g.moveTo(ox, oy); g.lineTo(ox + mp.w * sx, oy);
      g.lineTo(ox + mp.w * sx + mp.h * shear, oy + mp.h * sy);
      g.lineTo(ox + mp.h * shear, oy + mp.h * sy); g.closePath(); g.stroke();
      g.fillStyle = active ? "#9fb6cc" : "#4d5868";
      g.font = "9.5px ui-monospace,monospace";
      g.fillText(`c${i}`, ox - 17, oy + mp.h * sy * 0.6);
    };

    if (tilt < 0.12) {                                   // vue à plat : une grille de 16 cartes
      const cols = Math.ceil(Math.sqrt(C)), rows = Math.ceil(C / cols);
      const cell = Math.min((w - 40) / (cols * mw + cols), (h - 52) / (rows * mh + rows));
      v.maps.forEach((mp, i) => {
        const r = Math.floor(i / cols), c = i % cols;
        const ox = 24 + c * (mw * cell + cell * 1.6), oy = 26 + r * (mh * cell + cell * 2.2);
        cellOf(mp, i < v.c_active, i, cell, cell, 0, ox, oy);
      });
    } else {                                             // vue isométrique : les plans empilés
      const shear = 0.55 * tilt;                         // décalage horizontal par ligne
      const sqz = 1 - 0.5 * tilt;                        // écrasement vertical
      const lift = 10 + 9 * tilt;                        // écart entre deux plans
      let cell = Math.min(14, (w - 70) / (mw + mh * shear),
        (h - 60 - (C - 1) * lift) / (mh * sqz));
      cell = Math.max(2, cell);
      const planeW = mw * cell + mh * cell * shear, planeH = mh * cell * sqz;
      const ox = (w - planeW) / 2 + 8;
      const oyBottom = h - 30 - planeH;                  // le plan c0 (le plus proche) en bas
      for (let i = C - 1; i >= 0; i--) {                 // du plus lointain au plus proche
        cellOf(v.maps[i], i < v.c_active, i, cell, cell * sqz, cell * shear, ox, oyBottom - i * lift);
      }
    }
    g.fillStyle = "#93a0b4"; g.font = "11px ui-monospace,monospace";
    g.fillText(`${v.c_active} / ${v.c_total} canaux émis`, 8, 14);
    g.fillStyle = "#4d5868";
    g.fillText("les canaux grisés ne traversent pas le canal", 8, h - 8);
  };
  slider.oninput = () => {
    $("#lz") && ($("#lz").textContent = +slider.value < 12 ? "grille plate" : "3D empilée");
    draw();
  };
  S.redraw.push(draw);
  requestAnimationFrame(draw);
  return wrapEl;
}

/* ------------------------------------------------------------------------------------------ popover */
function popover(anchor, [title, text, file]) {
  const p = $("#pop");
  p.innerHTML = `<h4>${title}</h4><p>${text}</p>${file ? `<p class="f" style="color:#6b7889;
    font:11px var(--mono);margin-top:8px">${file}</p>` : ""}`;
  p.hidden = false;
  const r = anchor.getBoundingClientRect();
  p.style.left = Math.min(innerWidth - 350, r.left) + "px";
  p.style.top = (r.bottom + 8) + "px";
  const close = (e) => {
    if (e.target === anchor) return;
    p.hidden = true;
    removeEventListener("click", close);
  };
  setTimeout(() => addEventListener("click", close), 0);
}

/* ------------------------------------------------------------------------------------------- lecture */
function play() {
  if (!S.trace) return;
  S.playing = true;
  $("#play").textContent = "❙❙";
  if (S.i >= S.trace.stages.length - 1) show(0);
  clearInterval(S.timer);
  S.timer = setInterval(() => {
    if (S.i >= S.trace.stages.length - 1) return pause();
    show(S.i + 1);
  }, +$("#speed").value);
}
function pause() {
  S.playing = false;
  clearInterval(S.timer);
  $("#play").textContent = "▶";
}

/* --------------------------------------------------------------------------------------- comparaison */
async function compare() {
  const grid = $("#cmpGrid");
  if (!S.trace) { grid.innerHTML = "<p class=\"placeholder\">Lancez d'abord une transmission.</p>"; return; }
  if (!S.cmpSel.size) { grid.innerHTML = '<p class="placeholder">Choisissez au moins un système.</p>'; return; }
  const s = settings();
  grid.innerHTML = `<p class="placeholder"><span class="spin"></span> ${S.cmpSel.size} transmissions en cours…</p>`;
  $("#cmpBtn").disabled = true;
  try {
    const r = await api("/api/compare", { ...s, systems: Array.from(S.cmpSel) });
    grid.innerHTML = `<div class="cmp-card"><h4>Image originale</h4>
      <img src="${S.trace.original}" alt=""><div class="mgrid">
      <div class="mrow"><span class="k">image</span><b>${S.trace.image_title}</b></div>
      <div class="mrow"><span class="k">SNR</span><b>${r.snr} dB</b></div>
      <div class="mrow"><span class="k">canal</span><b>${r.channel}</b></div>
      <div class="mrow"><span class="k">k/n</span><b>${fmtRatio(S.trace.summary.ratio)}</b></div></div></div>` +
      r.items.map(it => {
        if (it.error) return `<div class="cmp-card"><h4>${it.label}</h4><p class="err">${it.error}</p></div>`;
        const m = it.summary;
        const rows = [
          ["PSNR", m.psnr != null ? m.psnr.toFixed(2) + " dB" : "—"],
          ["SSIM", m.ssim != null ? m.ssim.toFixed(4) : "—"],
          m.judge_class ? ["classe reçue", m.judge_class] : null,
          m.sem_cos != null ? ["sem_cos", m.sem_cos.toFixed(4)] : null,
          ["k/n", fmtRatio(m.ratio)],
        ].filter(Boolean);
        return `<div class="cmp-card"><h4>${it.label}</h4>
          ${it.received ? `<img src="${it.received}" alt="">` :
            '<p class="err" style="color:var(--muted)">aucune image : système orienté tâche</p>'}
          <div class="mgrid">${rows.map(([k, v]) => `<div class="mrow ${k === "classe reçue" ?
            (m.judge_ok ? "good" : "bad") : ""}"><span class="k">${k}</span><b>${v}</b></div>`).join("")}</div>
          </div>`;
      }).join("");
  } catch (e) {
    grid.innerHTML = `<p class="placeholder">${e.message}</p>`;
  } finally {
    $("#cmpBtn").disabled = false;
  }
}

/* ------------------------------------------------------------------------------------------- courbes */
async function loadResults() {
  const dataset = $("#resData").value, metric = $("#resMetric").value || "psnr", chan = $("#resChan").value;
  try {
    S.chart = await api(`/api/results?dataset=${dataset}&metric=${metric}&channel=${chan}`);
    if (!S.userToggled) {  // l'étude de désadaptation encombre la comparaison principale : masquée au départ
      S.chartOff = new Set(S.chart.series.filter(s => s.label.includes("entraîné")).map(s => s.label));
    }
    drawChart();
  } catch (e) {
    $("#chartNote").textContent = e.message;
  }
}

function metricLabel() {
  const ms = S.cat.metrics[$("#resData").value] || [];
  return (ms.find(m => m.id === ($("#resMetric").value || "psnr")) || {}).label || "";
}

function drawChart() {
  const svg = $("#chart"), d = S.chart;
  if (!d) return;
  svg.innerHTML = "";
  const thin = $("#resThin").checked;
  const series = d.series.filter(s => (thin || !s.thin) && !S.chartOff.has(s.label));
  const L = 62, R = 16, T = 18, B = 46, W = 900, H = 480;
  const mk = (tag, at, parent = svg) => {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(at)) e.setAttribute(k, v);
    parent.append(e);
    return e;
  };
  if (!series.length) {
    mk("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "axl" }).textContent =
      "aucune courbe — lancez scripts/evaluate.py, ou activez une entrée de la légende";
    $("#chartLegend").innerHTML = "";
    return;
  }
  const ys = series.flatMap(s => s.points.map(p => p[1]));
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = (y1 - y0) * 0.08 || 1;
  y0 -= pad; y1 += pad;
  const X = (v) => L + (v / 20) * (W - L - R), Y = (v) => H - B - ((v - y0) / (y1 - y0)) * (H - T - B);

  for (let s = 0; s <= 20; s += 2) {
    mk("line", { x1: X(s), y1: T, x2: X(s), y2: H - B, class: "gl" });
    mk("text", { x: X(s), y: H - B + 16, "text-anchor": "middle", class: "axt" }).textContent = s;
  }
  for (let t = 0; t <= 5; t++) {
    const v = y0 + (y1 - y0) * t / 5;
    mk("line", { x1: L, y1: Y(v), x2: W - R, y2: Y(v), class: "gl" });
    mk("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end", class: "axt" })
      .textContent = Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3);
  }
  mk("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, class: "ax" });
  mk("line", { x1: L, y1: T, x2: L, y2: H - B, class: "ax" });
  mk("text", { x: (L + W - R) / 2, y: H - 10, "text-anchor": "middle", class: "axl" })
    .textContent = "SNR du canal (dB)";
  mk("text", { x: 14, y: (T + H - B) / 2, "text-anchor": "middle", class: "axl",
    transform: `rotate(-90 14 ${(T + H - B) / 2})` }).textContent = metricLabel();

  const dashOf = (l) => l.includes("entraîné 0") ? "2 4" : l.includes("entraîné 20") ? "8 3" : "";
  for (const s of series) {
    const col = SERIES_COLOR(s.label);
    const pts = s.points.map(p => `${X(p[0])},${Y(p[1])}`).join(" ");
    mk("polyline", { points: pts, fill: "none", stroke: col, "stroke-width": s.thin ? 1 : 2.4,
      "stroke-opacity": s.thin ? .5 : 1, "stroke-dasharray": s.dash ? "7 4" : dashOf(s.label),
      "stroke-linejoin": "round" });
    if (!s.thin) {
      for (const p of s.points) mk("circle", { cx: X(p[0]), cy: Y(p[1]), r: 2.6, fill: col });
    }
  }
  $("#chartNote").textContent = d.notes.join(" ") + " — source : results/eval/*.csv";
  $("#chartLegend").innerHTML = d.series.filter(s => thin || !s.thin).map(s =>
    `<span class="lg ${S.chartOff.has(s.label) ? "off" : ""}" data-l="${s.label}">
      <i style="background:${SERIES_COLOR(s.label)}"></i>${s.label}</span>`).join("");
  $$("#chartLegend .lg").forEach(el => el.onclick = () => {
    const l = el.dataset.l;
    S.userToggled = true;
    S.chartOff.has(l) ? S.chartOff.delete(l) : S.chartOff.add(l);
    drawChart();
  });
}

/* --------------------------------------------------------------------------------------------- liens */
function wire() {
  $("#tabs").onclick = (e) => {
    const b = e.target.closest(".tab");
    if (!b) return;
    $$(".tab").forEach(t => t.classList.toggle("is-on", t === b));
    $$(".panel").forEach(p => p.classList.toggle("is-on", p.id === "tab-" + b.dataset.tab));
    if (b.dataset.tab === "results") drawChart();
    if (b.dataset.tab === "compare" && !$("#cmpGrid").children.length) compare();
  };
  $("#snr").oninput = () => { $("#snrOut").textContent = $("#snr").value + " dB"; };
  $("#snr").onchange = run;
  for (const id of ["#imgSel", "#chanSel", "#cSel", "#mcsSel"]) $(id).onchange = run;
  $("#seed").onchange = run;
  $("#runBtn").onclick = run;
  $("#play").onclick = () => (S.playing ? pause() : play());
  $("#next").onclick = () => { pause(); show(S.i + 1); };
  $("#prev").onclick = () => { pause(); show(S.i - 1); };
  $("#first").onclick = () => { pause(); show(0); };
  $("#last").onclick = () => { pause(); show(S.trace.stages.length - 1); };
  $("#scrub").oninput = () => { pause(); show(+$("#scrub").value); };
  $("#speed").onchange = () => { if (S.playing) play(); };
  $("#cmpBtn").onclick = compare;
  for (const id of ["#resData", "#resMetric", "#resChan"]) {
    $(id).onchange = () => {
      if (id === "#resData") {
        const ms = S.cat.metrics[$("#resData").value];
        $("#resMetric").innerHTML = "";
        $("#resMetric").append(...ms.map(m => new Option(m.label, m.id)));
      }
      S.chartOff.clear();
      loadResults();
    };
  }
  $("#resThin").onchange = drawChart;

  let rt = null;
  addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => S.redraw.forEach(f => f()), 120);
  }, { passive: true });

  addEventListener("keydown", (e) => {
    if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (!S.trace && e.key !== "c" && e.key !== "C") return;
    const k = e.key;
    if (k === " ") { e.preventDefault(); S.playing ? pause() : play(); }
    else if (k === "ArrowRight") { pause(); show(S.i + 1); }
    else if (k === "ArrowLeft") { pause(); show(S.i - 1); }
    else if (k === "Home") { pause(); show(0); }
    else if (k === "End") { pause(); show(S.trace.stages.length - 1); }
    else if (k === "ArrowUp" || k === "ArrowDown") {
      e.preventDefault();
      const s = Math.max(0, Math.min(20, +$("#snr").value + (k === "ArrowUp" ? 1 : -1)));
      $("#snr").value = s; $("#snrOut").textContent = s + " dB"; run();
    } else if (k === "c" || k === "C") $('.tab[data-tab="compare"]').click();
  });
}

boot().catch(e => { $("#inspector").innerHTML = `<p class="placeholder">Erreur de démarrage : ${e.message}</p>`; });
