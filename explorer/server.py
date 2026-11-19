"""Local HTTP server and JSON API for the explorer. Standard library only — no extra dependency.

    python scripts/explore.py            # then open the printed address

Routes
    GET  /                     the page
    GET  /api/catalog          systems, images, options, device
    POST /api/run              one traced transmission  {system, image, snr, c, channel, mcs, seed}
    POST /api/compare          the same image and settings through several systems (received image + metrics)
    GET  /api/results          curves measured by the real campaign (results/eval/*.csv)
"""
import csv
import json
import mimetypes
import os
import threading
import traceback
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from semcom import ROOT

from . import catalog as C
from . import trace as T

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
EVAL = os.path.join(ROOT, "results", "eval")
LOCK = threading.Lock()  # torch models are shared; serialise the heavy work
MAIN_RATIO = 8 / 96  # k/n = 1/12, the operating point of the article


# --------------------------------------------------------------------------------------------- results
# The eval CSVs are read with the standard library on purpose: pandas corrupts the heap when read_csv runs
# inside a request thread on this install (Windows fatal exception 0xc0000374), and these files are tiny.
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read(name):
    """results/eval/<name> -> list of dicts; numeric fields become floats, empty fields become None."""
    p = os.path.join(EVAL, name)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({k: (_num(v) if _num(v) is not None else (v or None)) for k, v in r.items()})
        return rows


def _pick(rows, ratio=None, **eq):
    """Filter rows on exact fields, plus an approximate match on the bandwidth ratio."""
    out = []
    for r in rows:
        if any(r.get(k) != v for k, v in eq.items()):
            continue
        if ratio is not None and (r.get("ratio") is None or abs(r["ratio"] - ratio) > 1e-6):
            continue
        out.append(r)
    return out


def _series(rows, label, metric, group=None, **flags):
    """Mean of `metric` per SNR, as a curve."""
    acc = defaultdict(list)
    for r in rows:
        v, s = r.get(metric), r.get("snr")
        if isinstance(v, float) and isinstance(s, float) and v == v:
            acc[s].append(v)
    if not acc:
        return None
    pts = [[s, sum(v) / len(v)] for s, v in sorted(acc.items())]
    return dict(label=label, group=group, points=pts, **flags)


def _best_mcs(rows, metric, label):
    """The genie transmitter: per SNR, the MCS with the best value of `metric` (as in make_figures.py)."""
    per = defaultdict(dict)  # snr -> mcs -> mean
    acc = defaultdict(list)
    for r in rows:
        v, s, m = r.get(metric), r.get("snr"), r.get("mcs")
        if isinstance(v, float) and isinstance(s, float) and v == v:
            acc[(s, m)].append(v)
    for (s, m), vs in acc.items():
        per[s][m] = sum(vs) / len(vs)
    if not per:
        return None, []
    best = dict(label=label, group="séparé", dash=True,
                points=[[s, max(d.values())] for s, d in sorted(per.items())])
    thin = []
    for m in sorted({m for d in per.values() for m in d}):
        pts = [[s, d[m]] for s, d in sorted(per.items()) if m in d]
        if pts:
            thin.append(dict(label=f"Séparé — {m}", group="séparé, MCS fixe", points=pts, thin=True))
    return best, thin


def results(dataset="cifar", metric="psnr", channel="awgn"):
    """Curves from the measured campaign. Nothing is recomputed here — these are the article's numbers."""
    out, notes = [], []
    if dataset == "cifar":
        analog, dig, cap = _read("cifar_analog.csv"), _read("cifar_digital.csv"), _read("cifar_capacity.csv")
        for scheme, lab in [("jscc_mse", "JSCC syntaxique"), ("jscc_sem", "JSCC sémantique"),
                            ("jscc_mse_snr0", "JSCC entraîné 0 dB"), ("jscc_mse_snr10", "JSCC entraîné 10 dB"),
                            ("jscc_mse_snr20", "JSCC entraîné 20 dB")]:
            s = _series(_pick(analog, scheme=scheme, channel=channel, c=8.0), lab, metric, "appris")
            if s:
                out.append(s)
        s = _series(_pick(analog, scheme="task", channel=channel, c=8.0), "Orienté tâche (k/n = 1/48)",
                    metric, "extension")
        if s:
            out.append(s)
        if channel == "awgn":
            s = _series(_pick(cap, ratio=MAIN_RATIO, scheme="sep_ae_capacity"), "Séparé — borne de capacité",
                        metric, "séparé")
            if s:
                out.append(s)
        best, thin = _best_mcs(_pick(dig, ratio=MAIN_RATIO, channel=channel), metric,
                               "Séparé — meilleur MCS (génie)")
        out += ([best] if best else []) + thin
        notes.append("CIFAR-10, 5 000 images de test, k/n = 1/12. Précision mesurée par le juge B (VGG).")
    else:
        jscc = _read("kodak_jscc.csv" if dataset == "kodak" else "video_jscc.csv")
        sep = _read("kodak_sep.csv" if dataset == "kodak" else "video_sep.csv")
        for scheme, lab in [("jscc_mse", "JSCC syntaxique"), ("jscc_sem", "JSCC sémantique")]:
            s = _series(_pick(jscc, ratio=MAIN_RATIO, scheme=scheme, channel=channel), lab, metric, "appris")
            if s:
                out.append(s)
        s = _series(_pick(sep, ratio=MAIN_RATIO, scheme="sep_jpeg_capacity", channel=channel),
                    "Séparé — borne de capacité", metric, "séparé")
        if s:
            out.append(s)
        best, thin = _best_mcs(_pick(sep, ratio=MAIN_RATIO, scheme="sep_jpeg", channel=channel), metric,
                               "Séparé — meilleur MCS (génie)")
        out += ([best] if best else []) + thin
        notes.append("Kodak : 24 images 768×512, jamais vues à l'entraînement (modèles entraînés sur CIFAR-10)."
                     if dataset == "kodak" else
                     "Vidéo : 30 premières images de akiyo (statique), foreman (mouvement modéré) et bus (pan).")
    if not out:
        notes.append("Aucune donnée pour ce réglage : lancez scripts/evaluate.py all.")
    return dict(dataset=dataset, metric=metric, channel=channel, series=out, notes=notes)


METRICS = {
    "cifar": [dict(id="psnr", label="PSNR (dB)"), dict(id="ssim", label="SSIM"),
              dict(id="acc", label="Précision du juge B")],
    "kodak": [dict(id="psnr", label="PSNR (dB)"), dict(id="ssim", label="SSIM"),
              dict(id="sem_cos", label="Similarité sémantique"), dict(id="top1_agree", label="Accord top-1")],
    "video": [dict(id="psnr", label="PSNR (dB)"), dict(id="ssim", label="SSIM"),
              dict(id="sem_cos", label="Similarité sémantique"), dict(id="top1_agree", label="Accord top-1")],
}


# ---------------------------------------------------------------------------------------------- compare
def compare(systems, image, snr, c_active, channel, mcs, seed):
    out = []
    for s in systems:
        try:
            tr = T.run(s, image, snr, c_active, channel, mcs, seed)
            out.append(dict(system=s, label=C.ID2SYS[s]["short"], received=tr["received"],
                            summary=tr["summary"], ratio=tr["summary"].get("ratio")))
        except Exception as exc:
            out.append(dict(system=s, label=C.ID2SYS.get(s, {}).get("short", s), error=f"{exc}"))
    return dict(image=image, snr=snr, c_active=c_active, channel=channel, mcs=mcs, items=out)


# ----------------------------------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    server_version = "SemComExplorer/1.0"

    def log_message(self, fmt, *args):  # one tidy line per request
        if not self.path.startswith("/static"):
            print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

    # ---- plumbing
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, allow_nan=False, default=float))

    def _fail(self, exc):
        traceback.print_exc()
        self._json(dict(error=f"{type(exc).__name__}: {exc}"), 400)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _static(self, path):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.normpath(os.path.join(STATIC, rel))
        if not full.startswith(STATIC) or not os.path.isfile(full):
            return self._send(404, "not found", "text/plain; charset=utf-8")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    # ---- routes
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/catalog":
                return self._json(dict(C.catalog(), metrics=METRICS))
            if u.path == "/api/results":
                return self._json(results(q.get("dataset", "cifar"), q.get("metric", "psnr"),
                                          q.get("channel", "awgn")))
            return self._static(u.path)
        except Exception as exc:
            self._fail(exc)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self._body()
            if u.path == "/api/run":
                with LOCK:
                    return self._json(T.run(b.get("system", "jscc_sem"), b.get("image", "cifar:0"),
                                            b.get("snr", 10), b.get("c", 8), b.get("channel", "awgn"),
                                            b.get("mcs", "QPSK 1/2"), b.get("seed", 0)))
            if u.path == "/api/compare":
                with LOCK:
                    return self._json(compare(b.get("systems", ["sep", "jscc_mse", "jscc_sem"]),
                                              b.get("image", "cifar:0"), b.get("snr", 10), b.get("c", 8),
                                              b.get("channel", "awgn"), b.get("mcs", "QPSK 1/2"),
                                              b.get("seed", 0)))
            return self._send(404, "not found", "text/plain; charset=utf-8")
        except Exception as exc:
            self._fail(exc)


def serve(host="127.0.0.1", port=8077, open_browser=True):
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Explorateur Projet 07 : {url}")
    print(f"  modèles sur {C.device()} · Ctrl+C pour arrêter")
    if open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\narrêt")
    finally:
        httpd.server_close()
