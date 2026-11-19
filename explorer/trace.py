"""Run one transmission and record every stage.

Everything here calls the project's own code: `semcom.channel` for the physics, `semcom.digital` for the
separated chain, `semcom.models` for the learned ones, `semcom.metrics` for the numbers. The explorer adds no
simulation of its own — it only watches, so what the page shows is what `scripts/evaluate.py` measures.
"""
import math

import numpy as np
import torch

from semcom import channel as ch
from semcom import digital as dg
from semcom.metrics import psnr, ssim
from semcom.models import CIFAR_MEAN, CIFAR_STD

from . import encode as E
from .catalog import CIFAR_CLASSES, ID2SYS, load_image, model

PAIRS = "Deux valeurs réelles voisines forment un symbole complexe (I + jQ)."


# ------------------------------------------------------------------------------------------------ helpers
def to_complex(t):
    """Real tensor -> complex symbols, pairing consecutive values as (I, Q)."""
    v = np.asarray(t.detach().cpu().reshape(-1), dtype=np.float32)
    if v.size % 2:
        v = v[:-1]
    return v[0::2] + 1j * v[1::2]


def ratio_str(k, n):
    return f"1/{round(n / k)}" if k else "0"


def node(nid, label, sub, x, y=110, side="tx"):
    return dict(id=nid, label=label, sub=sub, x=x, y=y, side=side)


def spread(labels, width=1200, y=110):
    """Lay nodes out evenly along the wire. labels: [(id, label, sub, side)]."""
    n = len(labels)
    margin = width / (2 * n)
    return [node(i, l, s, round(margin + k * (width - 2 * margin) / max(1, n - 1)), y, sd)
            for k, (i, l, s, sd) in enumerate(labels)]


def judge_views(x, xh, label, kind):
    """Final metrics: pixel fidelity always, plus the semantic judge that fits the image."""
    views, items = [], []
    p = float(psnr(xh, x).mean())
    s = float(ssim(xh, x).mean())
    items += [dict(k="PSNR", v=f"{p:.2f} dB", hint="fidélité pixel ; 30 dB est bon, 20 dB visiblement dégradé"),
              dict(k="SSIM", v=f"{s:.4f}", hint="structure locale, 0 à 1 ; plus proche de l'œil que le PSNR")]
    out = dict(psnr=round(p, 3), ssim=round(s, 4))
    if kind == "cifar" and label is not None:
        judge = model("clsB")
        dev = next(judge.parameters()).device
        with torch.no_grad():
            p0 = torch.softmax(judge(x.to(dev)), 1)[0].cpu()
            p1 = torch.softmax(judge(xh.to(dev)), 1)[0].cpu()
        views.append(dict(type="bars", title="Juge B (VGG) sur l'image originale",
                          note=f"vérité : {CIFAR_CLASSES[label]}", items=E.bars(p0, CIFAR_CLASSES, label)))
        views.append(dict(type="bars", title="Juge B (VGG) sur l'image reçue",
                          note="le juge n'a jamais servi à l'entraînement d'aucun système",
                          items=E.bars(p1, CIFAR_CLASSES, label)))
        ok = int(p1.argmax()) == label
        items.append(dict(k="Classe reçue", v=f"{CIFAR_CLASSES[int(p1.argmax())]} ({float(p1.max()):.0%})",
                          hint="verdict du juge indépendant", good=ok, bad=not ok))
        out.update(judge_ok=ok, judge_class=CIFAR_CLASSES[int(p1.argmax())])
    else:
        try:
            from semcom.metrics import semantic_similarity

            cos, agree, top5 = semantic_similarity(x, xh)
            items += [dict(k="sem_cos", v=f"{float(cos.mean()):.4f}",
                           hint="similarité cosinus des caractéristiques ResNet-18 ImageNet"),
                      dict(k="top-1 identique", v="oui" if float(agree.mean()) > 0.5 else "non",
                           hint="le ResNet-18 donne-t-il la même étiquette avant et après transmission ?",
                           good=float(agree.mean()) > 0.5, bad=float(agree.mean()) <= 0.5)]
            out.update(sem_cos=round(float(cos.mean()), 4), top1_agree=float(agree.mean()))
        except Exception as exc:  # ResNet-18 weights missing: pixel metrics still stand
            items.append(dict(k="Juge sémantique", v="indisponible", hint=f"{type(exc).__name__}: {exc}"))
    views.insert(0, dict(type="metrics", title="Mesures", items=items))
    return views, out


# ---------------------------------------------------------------------------------------- learned chains
def trace_jscc(sys_id, x, label, kind, snr, c_active, channel, seed):
    """Deep JSCC, syntactic or semantic: encoder -> power normalisation -> channel -> decoder."""
    net = model(sys_id)
    dev = next(net.parameters()).device
    n_src = int(x[0].numel())
    semantic = sys_id == "jscc_sem"

    with torch.no_grad():
        z = net.enc(x.to(dev)).cpu()
        xn = ch.power_normalize(z, c_active)
        gen = torch.Generator().manual_seed(seed)
        if channel == "rayleigh":
            y_full, snr_inst = ch.rayleigh_block(xn, snr, generator=gen)
            snr_inst = float(snr_inst[0])
        else:
            torch.manual_seed(seed)
            y_full, snr_inst = ch.awgn(xn, snr), float(snr)
        y = ch.truncate(y_full, c_active)
        xh = net.dec(y.to(dev)).cpu().clamp(0, 1)

    sym_tx = to_complex(xn[:, :c_active])
    sym_rx = to_complex(y[:, :c_active])
    k = int(sym_tx.size)
    nodes = spread([("src", "Source", "image", "tx"), ("enc", "Encodeur CNN", "stride 4, 16 canaux", "tx"),
                    ("norm", "Normalisation", "P = 1 / symbole", "tx"), ("chan", "Canal", channel.upper(), "ch"),
                    ("dec", "Décodeur CNN", "reconstruction", "rx"), ("judge", "Juge", "mesures", "rx")])
    st = []

    st.append(dict(title="Image source", node="src", kind="source", size=f"{n_src} valeurs réelles",
                   concept="kn",
                   why="L'émetteur part d'une image dans [0,1]. Le dénominateur du rapport de bande est "
                       f"n = 3·H·W = {n_src} : c'est le nombre de nombres à transmettre.",
                   views=[dict(type="image", title="x — image à transmettre", png=E.png(x),
                               note=f"{x.shape[-2]}×{x.shape[-1]} pixels, 3 canaux couleur")]))

    st.append(dict(title="Encodeur CNN", node="enc", kind="codage conjoint",
                   size=f"{z.shape[1]} canaux × {z.shape[2]}×{z.shape[3]}", concept="jscc",
                   why="Un réseau convolutif comprime l'image en un tenseur latent au stride 4. Il n'y a ni "
                       "bits ni fichier : les valeurs latentes partiront telles quelles sur le canal. "
                       + ("Ce réseau a été entraîné avec la perte sémantique, donc ses canaux latents gardent "
                          "ce qui fait le sens de l'image." if semantic else
                          "Ce réseau a été entraîné avec la seule MSE pixel."),
                   views=[dict(type="latent", title="z — 16 cartes latentes (vue empilée)",
                               note="les canaux au-delà de la bande choisie ne seront pas émis",
                               **E.latent_stack(z, c_active))]))

    st.append(dict(title="Troncature de bande", node="norm", kind="bande",
                   size=f"{c_active}/16 canaux → k = {k} symboles", concept="kn",
                   why=f"Seuls les {c_active} premiers canaux latents sont émis ; les autres sont mis à zéro au "
                       f"récepteur. Pendant l'entraînement, la bande était tirée au hasard, donc les canaux sont "
                       f"rangés par importance : on peut choisir le débit au dernier moment. "
                       f"k/n = {c_active}/96 = {ratio_str(k, n_src)}.",
                   views=[dict(type="metrics", title="Budget radio", items=[
                       dict(k="k (symboles complexes)", v=str(k)),
                       dict(k="n (valeurs source)", v=str(n_src)),
                       dict(k="k/n", v=f"{ratio_str(k, n_src)} ({k / n_src:.4f})",
                            hint="plus c'est petit, plus la compression est agressive"),
                       dict(k="canaux latents actifs", v=f"{c_active} sur 16")])]))

    st.append(dict(title="Normalisation de puissance", node="norm", kind="contrainte", size="P = 1 / symbole",
                   concept="power",
                   why="Un émetteur ne peut pas crier plus fort pour couvrir le bruit. La puissance moyenne est "
                       "ramenée à 1 par symbole complexe. Sans cette contrainte, le réseau gagnerait en "
                       "augmentant simplement l'amplitude, et la comparaison avec la chaîne numérique n'aurait "
                       "plus de sens.",
                   views=[dict(type="constellation", title="Symboles émis (plan I/Q)",
                               note="nuage continu : aucune constellation discrète, contrairement à la QAM",
                               **E.cloud(sym_tx, seed=seed))]))

    sigma2 = 10 ** (-snr_inst / 10)
    chan_items = [dict(k="SNR consigne", v=f"{snr:.0f} dB"),
                  dict(k="σ² (bruit complexe)", v=f"{sigma2:.4f}")]
    if channel == "rayleigh":
        chan_items.insert(1, dict(k="SNR instantané", v=f"{snr_inst:.2f} dB",
                                  hint="après le tirage de l'évanouissement |h|²",
                                  bad=snr_inst < snr - 3, good=snr_inst > snr + 3))
        chan_items.append(dict(k="|h|²", v=f"{10 ** ((snr_inst - snr) / 10):.3f}",
                               hint="gain de puissance du canal, loi exponentielle(1)"))
    st.append(dict(title=f"Canal {channel.upper()}", node="chan", kind="canal",
                   size=f"{k} symboles bruités", concept="rayleigh" if channel == "rayleigh" else "awgn",
                   why=("Évanouissement par blocs : un seul coefficient h ~ CN(0,1) pour toute l'image. Le "
                        "récepteur connaît h (CSI parfaite) et divise par h — égalisation zero-forcing — ce qui "
                        "ramène le problème à un AWGN de SNR·|h|². Quand le tirage est mauvais, tout le reste en "
                        "souffre." if channel == "rayleigh" else
                        "Le canal ajoute à chaque symbole un bruit gaussien indépendant : y = x + n. Comparez le "
                        "nuage reçu au nuage émis — c'est exactement l'information que le décodeur perd."),
                   views=[dict(type="constellation", title="Symboles reçus (plan I/Q)",
                               note="en gris, les symboles émis ; en couleur, ce qui arrive au récepteur",
                               ref=E.cloud(sym_tx, seed=seed)["points"], **E.cloud(sym_rx, seed=seed)),
                          dict(type="metrics", title="Canal", items=chan_items)]))

    st.append(dict(title="Décodeur CNN", node="dec", kind="reconstruction", size="image reconstruite",
                   concept="cliff",
                   why="Le décodeur reconstruit l'image directement à partir des symboles bruités. Comme il a "
                       "appris avec le bruit dans la boucle de rétropropagation, la dégradation est "
                       "progressive : pas de seuil, pas d'effondrement brutal. Faites varier le SNR et "
                       "regardez : l'image se dégrade, elle ne disparaît jamais d'un coup.",
                   views=[dict(type="image", title="x̂ — image reçue", png=E.png(xh)),
                          dict(type="image", title="x — originale, pour comparer", png=E.png(x))]))

    views, summary = judge_views(x, xh, label, kind)
    st.append(dict(title="Juges et mesures", node="judge", kind="évaluation", size="PSNR · SSIM · sens",
                   concept="judge",
                   why="La fidélité pixel et le sens sont deux choses différentes. Comparez le JSCC sémantique "
                       "au JSCC syntaxique au même SNR : le sémantique perd ~0,5 dB de PSNR et gagne 10 à 14 "
                       "points de précision. Les pixels perdus n'étaient pas ceux qui portaient le sens.",
                   views=views))
    summary.update(k=k, n=n_src, ratio=k / n_src, snr_inst=snr_inst)
    return nodes, st, summary, E.png(xh)


def trace_task(x, label, snr, c_active, channel, seed):
    """Task-oriented transmission: no image is reconstructed, only the decision crosses the channel."""
    net = model("task")
    dev = next(net.parameters()).device
    n_src = int(x[0].numel())
    with torch.no_grad():
        xx = (x.to(dev) - CIFAR_MEAN.to(dev)) / CIFAR_STD.to(dev)
        z = net.enc(xx).cpu()
        xn = ch.power_normalize(z, c_active)
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        if channel == "rayleigh":
            y_full, snr_inst = ch.rayleigh_block(xn, snr, generator=gen)
            snr_inst = float(snr_inst[0])
        else:
            y_full, snr_inst = ch.awgn(xn, snr), float(snr)
        y = ch.truncate(y_full, c_active)
        logits = net.head(y.to(dev)).cpu()
        probs = torch.softmax(logits, 1)[0]
        judge = model("clsB")
        jdev = next(judge.parameters()).device
        p_ref = torch.softmax(judge(x.to(jdev)), 1)[0].cpu()

    sym = to_complex(xn[:, :c_active])
    k = int(sym.size)
    nodes = spread([("src", "Source", "image", "tx"), ("enc", "Encodeur tâche", "stride 8", "tx"),
                    ("norm", "Normalisation", "P = 1", "tx"), ("chan", "Canal", channel.upper(), "ch"),
                    ("head", "Tête de décision", "10 classes", "rx")])
    st = [
        dict(title="Image source", node="src", kind="source", size=f"{n_src} valeurs", concept="kn",
             why="Le récepteur ne veut pas l'image : il veut savoir ce qu'elle contient. Tout ce qui ne sert pas "
                 "à cette décision est du gaspillage de bande.",
             views=[dict(type="image", title="x — image à classer", png=E.png(x))]),
        dict(title="Encodeur orienté tâche", node="enc", kind="codage tâche",
             size=f"{z.shape[1]} canaux × {z.shape[2]}×{z.shape[3]}", concept="task",
             why="Encodeur au stride 8, sans décodeur image. Il est entraîné par entropie croisée : il "
                 "n'apprend à transmettre que ce qui discrimine les classes.",
             views=[dict(type="latent", title="z — carte latente de tâche", note="4×4 positions sur CIFAR-10",
                         **E.latent_stack(z, c_active))]),
        dict(title="Canal", node="chan", kind="canal", size=f"k = {k} symboles",
             concept="rayleigh" if channel == "rayleigh" else "awgn",
             why=f"k/n = {k}/{n_src} = {ratio_str(k, n_src)}. À comparer au 1/12 du JSCC complet : on peut "
                 f"descendre bien plus bas quand la tâche est connue d'avance.",
             views=[dict(type="constellation", title="Symboles reçus", ref=E.cloud(sym, seed=seed)["points"],
                         **E.cloud(to_complex(y[:, :c_active]), seed=seed)),
                    dict(type="metrics", title="Budget", items=[
                        dict(k="k", v=str(k)), dict(k="k/n", v=ratio_str(k, n_src)),
                        dict(k="SNR", v=f"{snr_inst:.2f} dB")])]),
        dict(title="Décision au récepteur", node="head", kind="décision", size="10 probabilités",
             concept="task",
             why="Il n'y a pas d'image à montrer : le récepteur produit directement une classe. C'est le point "
                 "extrême de la communication sémantique — on ne transmet que ce dont la tâche a besoin.",
             views=[dict(type="bars", title="Décision du récepteur orienté tâche",
                         note=f"vérité : {CIFAR_CLASSES[label]}" if label is not None else "",
                         items=E.bars(probs, CIFAR_CLASSES, label)),
                    dict(type="bars", title="Pour référence : juge B sur l'image originale",
                         items=E.bars(p_ref, CIFAR_CLASSES, label)),
                    dict(type="metrics", title="Résultat", items=[
                        dict(k="Classe décidée", v=CIFAR_CLASSES[int(probs.argmax())],
                             good=label is not None and int(probs.argmax()) == label,
                             bad=label is not None and int(probs.argmax()) != label),
                        dict(k="Confiance", v=f"{float(probs.max()):.1%}"),
                        dict(k="Image reconstruite", v="aucune", hint="c'est le principe du système")])]),
    ]
    ok = label is not None and int(probs.argmax()) == label
    return nodes, st, dict(k=k, n=n_src, ratio=k / n_src, snr_inst=snr_inst, judge_ok=ok,
                           judge_class=CIFAR_CLASSES[int(probs.argmax())]), None


# -------------------------------------------------------------------------------------- separated chain
def _traced_link(bits, mcs, snr_db, seed, fading_gain=None):
    """One bit stream through code -> QAM -> channel -> LLR -> Viterbi, keeping every intermediate.

    Mirrors `semcom.digital.transmit_bits` for a single stream (same packetisation, code, mapping and decoder).
    """
    mod, bps, rate = mcs
    rng = np.random.default_rng(seed)
    snr = snr_db + (10 * math.log10(max(fading_gain, 1e-12)) if fading_gain is not None else 0.0)
    n_pk, L = dg._packets(len(bits))
    pad = np.zeros(n_pk * L, np.uint8)
    pad[:len(bits)] = bits
    pk = pad.reshape(n_pk, L)
    coded, mask = dg.puncture(dg.conv_encode(pk), rate)
    n_c = coded.shape[1]
    coded_pad = np.concatenate([coded, np.zeros((n_pk, (-n_c) % bps), np.uint8)], 1)
    sym = dg.qam_mod(coded_pad.reshape(-1), bps).reshape(n_pk, -1)
    sigma2 = 10 ** (-snr / 10)
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(sym.shape) + 1j * rng.standard_normal(sym.shape))
    y = sym + noise
    llr = dg.qam_llr(y.reshape(-1), bps, np.full(y.size, sigma2)).reshape(n_pk, -1)[:, :n_c]
    dec = np.zeros_like(pk)
    for a in range(0, n_pk, 512):
        dec[a:a + 512] = dg.viterbi(dg.depuncture(llr[a:a + 512], mask), L)
    rec = dec.reshape(-1)[:len(bits)]
    hard = (llr < 0).astype(np.uint8)  # LLR = log P(0)/P(1): negative means "rather a 1"
    return dict(n_pk=n_pk, L=L, coded=coded, sym=sym, y=y, llr=llr, rec=rec, snr=snr, sigma2=sigma2,
                raw_ber=float(np.mean(hard != coded)), n_coded=int(n_c))


def trace_separated(x, label, kind, snr, c_active, channel, mcs_name, seed, capacity=False):
    """Compress, protect, modulate, transmit, decode — the chain of today, bit by bit."""
    n_src = int(x[0].numel())
    H, W = int(x.shape[-2]), int(x.shape[-1])
    natural = kind != "cifar"
    mcs = next(m for m in dg.MCS if dg.mcs_name(m) == mcs_name) if not capacity else None
    k = int(c_active * n_src / 96)
    gain = float(np.random.default_rng(seed + 7).exponential(1.0)) if channel == "rayleigh" else None
    eff_snr = snr + (10 * math.log10(max(gain, 1e-12)) if gain is not None else 0.0)

    st = []
    st.append(dict(title="Image source", node="src", kind="source", size=f"{n_src} valeurs réelles", concept="kn",
                   why=f"Même image, même budget que les systèmes appris : k = {k} symboles complexes pour "
                       f"n = {n_src} valeurs, soit k/n = {ratio_str(k, n_src)}.",
                   views=[dict(type="image", title="x — image à transmettre", png=E.png(x))]))

    if capacity:
        budget = int(k * math.log2(1 + 10 ** (eff_snr / 10)))
        st.append(dict(title="Budget donné par la capacité", node="enc", kind="borne", size=f"{budget} bits",
                       concept="shannon",
                       why="Hypothèse idéale : on suppose que le canal livre k·log2(1+SNR) bits sans la moindre "
                           "erreur. Aucun code réel n'atteint exactement ce débit sur des paquets courts, donc "
                           "c'est une borne supérieure, pas un système que l'on pourrait construire.",
                       views=[dict(type="metrics", title="Capacité de Shannon", items=[
                           dict(k="k", v=str(k)), dict(k="SNR", v=f"{eff_snr:.2f} dB"),
                           dict(k="log2(1+SNR)", v=f"{math.log2(1 + 10 ** (eff_snr / 10)):.3f} bits/symbole"),
                           dict(k="budget", v=f"{budget} bits")])]))

    if natural:
        budget = (int(k * math.log2(1 + 10 ** (eff_snr / 10))) if capacity else dg.max_bits(k, mcs))
        data, q = dg.jpeg_fit(x[0].numpy(), budget)
        if data is None:
            xh = torch.full_like(x, 0.5)
            st.append(dict(title="Codage source JPEG — échec", node="enc", kind="codage source",
                           size="aucun fichier ne tient", concept="cliff",
                           why="Même en qualité 1, le fichier JPEG ne tient pas dans le budget. Le récepteur "
                               "n'a rien à afficher : image grise. C'est le plancher de la chaîne séparée.",
                           views=[dict(type="image", title="x̂ — rien reçu", png=E.png(xh))]))
            nodes = spread([("src", "Source", "image", "tx"), ("enc", "JPEG", "échec", "tx"),
                            ("judge", "Juge", "mesures", "rx")])
            views, summary = judge_views(x, xh, label, kind)
            st.append(dict(title="Juges et mesures", node="judge", kind="évaluation", size="—", concept="judge",
                           views=views))
            summary.update(k=k, n=n_src, ratio=k / n_src, snr_inst=eff_snr)
            return nodes, st, summary, E.png(xh)
        stream = dg.bytes_to_bits(data)
        st.append(dict(title="Codage source JPEG", node="enc", kind="codage source",
                       size=f"{len(data)} octets · qualité {q}", concept="shannon",
                       why=f"Premier bloc du théorème de séparation : on retire la redondance de l'image. La "
                           f"qualité JPEG la plus haute qui tient dans {budget} bits est {q}. Remarquez le "
                           f"couplage caché : le budget dépend du SNR supposé, donc le codage de source est "
                           f"déjà un pari sur l'état du canal.",
                       views=[dict(type="image", title="Ce que JPEG a gardé (avant canal)",
                                   png=E.png(dg.jpeg_decode(data, (H, W))[0]),
                                   note=f"qualité {q} ; {len(stream)} bits à protéger"),
                              dict(type="bits", title="Début du fichier JPEG, en bits",
                                   note="ces bits n'ont aucune robustesse propre : un seul bit faux peut "
                                        "désynchroniser le décodeur", **E.bitdump(stream))]))
    else:
        dae = model("dae")
        budget = (int(k * math.log2(1 + 10 ** (eff_snr / 10))) if capacity else dg.max_bits(k, mcs))
        c_src = min(dae.c, budget // (64 * dae.bits))
        if c_src == 0:
            xh = dae.from_levels(torch.zeros(1, 0, 8, 8, dtype=torch.int64), 0).clamp(0, 1).detach()
            nodes = spread([("src", "Source", "image", "tx"), ("enc", "AE 4 bits", "budget nul", "tx"),
                            ("judge", "Juge", "mesures", "rx")])
            st.append(dict(title="Codage source — budget nul", node="enc", kind="codage source",
                           size="0 canal latent", concept="cliff",
                           why="Le budget ne permet même pas un canal latent de 64 valeurs sur 4 bits. Le "
                               "décodeur affiche sa valeur par défaut.",
                           views=[dict(type="image", title="x̂ — image par défaut", png=E.png(xh))]))
            views, summary = judge_views(x, xh, label, kind)
            st.append(dict(title="Juges et mesures", node="judge", kind="évaluation", size="—", concept="judge",
                           views=views))
            summary.update(k=k, n=n_src, ratio=k / n_src, snr_inst=eff_snr)
            return nodes, st, summary, E.png(xh)
        with torch.no_grad():
            lev = dae.to_levels(x, c_src)
        stream = ((lev.numpy().reshape(1, -1)[:, :, None] >> np.arange(dae.bits - 1, -1, -1)) & 1) \
            .reshape(-1).astype(np.uint8)
        st.append(dict(title="Codage source — auto-encodeur 4 bits", node="enc", kind="codage source",
                       size=f"{c_src} canaux × 64 valeurs × 4 bits = {len(stream)} bits", concept="shannon",
                       why=f"Sur CIFAR-10, le codeur de source est un auto-encodeur dont le latent est "
                           f"quantifié uniformément sur 4 bits. Le budget de {budget} bits autorise "
                           f"{c_src} canaux latents. À partir d'ici, tout est binaire : l'image n'existe plus, "
                           f"il n'y a que des bits.",
                       views=[dict(type="latent", title="Niveaux quantifiés émis",
                                   **E.latent_stack(lev.float(), c_src)),
                              dict(type="bits", title="Train binaire à transmettre", **E.bitdump(stream))]))

    if capacity:
        nodes = spread([("src", "Source", "image", "tx"),
                        ("enc", "Codage source", "JPEG / AE 4 bits", "tx"),
                        ("chan", "Canal idéal", "0 erreur (hypothèse)", "ch"),
                        ("dec", "Décodage source", "reconstruction", "rx"),
                        ("judge", "Juge", "mesures", "rx")])
        st.append(dict(title="Transmission supposée parfaite", node="chan", kind="borne", size="0 erreur",
                       concept="shannon",
                       why="On saute tout le codage de canal : par hypothèse, aucun bit n'est faux. C'est ce qui "
                           "rend cette courbe plate par morceaux — elle ne bouge que quand le budget de bits "
                           "change assez pour autoriser une qualité supérieure.",
                       views=[dict(type="metrics", title="Hypothèse", items=[
                           dict(k="BER", v="0", hint="par construction"),
                           dict(k="Effet falaise", v="absent", hint="il n'y a pas de décodeur à faire décrocher")])]))
        if natural:
            img, ok = dg.jpeg_decode(data, (H, W))
            xh = torch.from_numpy(img)[None]
        else:
            with torch.no_grad():
                xh = dae(x, c_src).clamp(0, 1)
    else:
        link = _traced_link(stream, mcs, snr, seed, fading_gain=gain)
        nodes = spread([("src", "Source", "image", "tx"), ("enc", "Codage source", "JPEG / AE", "tx"),
                        ("conv", "Code conv.", f"K=7, {mcs[2]}", "tx"), ("qam", "QAM", mcs[0], "tx"),
                        ("chan", "Canal", channel.upper(), "ch"), ("llr", "Démod. souple", "LLR max-log", "rx"),
                        ("vit", "Viterbi", "treillis 64 états", "rx"), ("dec", "Décodage source", "image", "rx"),
                        ("judge", "Juge", "mesures", "rx")])
        st.append(dict(title="Codage de canal — convolutif K=7", node="conv", kind="codage canal",
                       size=f"{len(stream)} → {link['n_coded'] * link['n_pk']} bits", concept="conv",
                       why=f"Deuxième bloc de la séparation : on rajoute de la redondance calculée. Chaque bit "
                           f"émis dépend des 7 derniers bits d'entrée (polynômes 133/171, ceux du Wi-Fi "
                           f"802.11a). Le rendement {mcs[2]} est obtenu par poinçonnage — on supprime des bits "
                           f"de redondance selon un motif fixe. {link['n_pk']} paquet(s) de {link['L']} bits, "
                           f"chacun terminé par 6 bits de queue.",
                       views=[dict(type="bits", title="Bits codés (début du premier paquet)",
                                   note="plus long que l'entrée : c'est le prix de la protection",
                                   **E.bitdump(link["coded"][0])),
                              dict(type="metrics", title="Code", items=[
                                  dict(k="Contrainte K", v="7", hint="64 états dans le treillis"),
                                  dict(k="Rendement", v=mcs[2]),
                                  dict(k="Paquets", v=f"{link['n_pk']} × {link['L']} bits"),
                                  dict(k="Bits codés", v=str(link["n_coded"] * link["n_pk"]))])]))
        st.append(dict(title=f"Modulation {mcs[0]}", node="qam", kind="modulation",
                       size=f"{link['sym'].size} symboles · {mcs[1]} bits/symbole", concept="qam",
                       why=f"Les bits sont groupés par {mcs[1]} et placés sur les points d'une constellation "
                           f"{mcs[0]} à énergie moyenne unitaire. Le codage de Gray numérote les points pour "
                           f"qu'une confusion entre voisins ne coûte qu'un seul bit faux. Plus de bits par "
                           f"symbole = plus de débit, mais des points plus serrés, donc plus fragiles.",
                       views=[dict(type="constellation", title=f"Constellation {mcs[0]} émise",
                                   note="points discrets et réguliers : exactement l'inverse du nuage continu "
                                        "du JSCC", **E.cloud(link["sym"], seed=seed))]))
        chan_items = [dict(k="SNR consigne", v=f"{snr:.0f} dB"), dict(k="σ²", v=f"{link['sigma2']:.4f}"),
                      dict(k="BER brut (avant Viterbi)", v=f"{link['raw_ber']:.3%}",
                           hint="erreurs vues par le démodulateur, avant correction",
                           bad=link["raw_ber"] > 0.05)]
        if gain is not None:
            chan_items.insert(1, dict(k="SNR instantané", v=f"{link['snr']:.2f} dB",
                                      bad=link["snr"] < snr - 3, good=link["snr"] > snr + 3))
            chan_items.append(dict(k="|h|²", v=f"{gain:.3f}"))
        st.append(dict(title=f"Canal {channel.upper()}", node="chan", kind="canal",
                       size=f"{link['y'].size} symboles bruités",
                       concept="rayleigh" if channel == "rayleigh" else "awgn",
                       why="Le même canal que pour les systèmes appris, avec le même SNR. Regardez les points "
                           "quitter leur case : dès qu'un point franchit la frontière de décision de son voisin, "
                           "des bits deviennent faux.",
                       views=[dict(type="constellation", title="Constellation reçue",
                                   note="en gris les points émis ; les frontières de décision sont à mi-chemin",
                                   ref=E.cloud(link["sym"], seed=seed)["points"], **E.cloud(link["y"], seed=seed)),
                              dict(type="metrics", title="Canal", items=chan_items)]))
        st.append(dict(title="Démodulation souple — LLR", node="llr", kind="démodulation",
                       size=f"{link['llr'].size} LLR", concept="llr",
                       why="Le démodulateur ne tranche pas. Pour chaque bit il calcule un log-rapport de "
                           "vraisemblance (approximation max-log) : le signe dit quel bit est le plus probable, "
                           "l'amplitude dit à quel point. Ces décisions souples valent environ 2 dB par rapport "
                           "à des décisions dures — c'est pour cela qu'on les garde.",
                       views=[dict(type="llr", title="LLR des premiers bits",
                                   note="LLR = log P(0)/P(1) · positif → plutôt 0 · proche de zéro → le "
                                        "démodulateur ne sait pas", **E.llr_dump(link["llr"].reshape(-1)))]))
        rec = link["rec"]
        errors = rec != stream
        st.append(dict(title="Décodage de Viterbi", node="vit", kind="décodage canal",
                       size=f"BER {float(errors.mean()):.3%}", concept="viterbi",
                       why="Viterbi cherche le chemin le plus vraisemblable dans le treillis à 64 états : c'est "
                           "le décodage optimal de ce code. Comparez le BER brut d'avant et celui d'ici — tant "
                           "que le canal reste au-dessus du seuil, la correction est quasi parfaite ; en "
                           "dessous, le décodeur décroche et les erreurs passent en masse.",
                       views=[dict(type="bits", title="Bits décodés (en rouge, les bits faux)",
                                   **E.bitdump(rec, errors=errors)),
                              dict(type="metrics", title="Correction", items=[
                                  dict(k="BER brut", v=f"{link['raw_ber']:.3%}"),
                                  dict(k="BER après Viterbi", v=f"{float(errors.mean()):.3%}",
                                       good=float(errors.mean()) == 0, bad=float(errors.mean()) > 0.001),
                                  dict(k="Bits faux", v=f"{int(errors.sum())} sur {errors.size}")])]))
        if natural:
            if np.array_equal(stream, rec):
                img, ok = dg.jpeg_decode(data, (H, W))
                why = ("Aucun bit faux : le JPEG se décode normalement. Tant que le canal tient, la chaîne "
                       "séparée est très bonne — c'est bien pour cela qu'elle est partout.")
            else:
                img, ok = dg.jpeg_decode(dg.bits_to_bytes(rec), (H, W))
                why = ("Des bits sont faux et le fichier JPEG est corrompu. Le décodeur produit des bandes, des "
                       "dérives de couleur, ou refuse le fichier — dans ce cas le récepteur n'a qu'une image "
                       "grise à afficher. C'est l'effet falaise : pas une dégradation, une perte.")
            xh = torch.from_numpy(img)[None]
            st.append(dict(title="Décodage source JPEG", node="dec", kind="reconstruction",
                           size="décodé" if ok else "fichier rejeté ou corrompu", concept="cliff", why=why,
                           views=[dict(type="image", title="x̂ — image reçue", png=E.png(xh),
                                       note="décodage propre" if ok else "fichier abîmé par le canal"),
                                  dict(type="image", title="x — originale", png=E.png(x))]))
        else:
            rl = rec.reshape(-1, dae.bits) @ (1 << np.arange(dae.bits - 1, -1, -1))
            q = torch.from_numpy(rl).view(1, c_src, 8, 8)
            with torch.no_grad():
                xh = dae.from_levels(q, c_src).clamp(0, 1)
            st.append(dict(title="Décodage source — auto-encodeur", node="dec", kind="reconstruction",
                           size="image reconstruite", concept="cliff",
                           why="Les bits décodés sont regroupés en niveaux de 4 bits, puis repassés dans le "
                               "décodeur de l'auto-encodeur. Un bit faux dans un niveau n'est pas une petite "
                               "erreur : il peut faire basculer la valeur du poids fort et produire une tache.",
                           views=[dict(type="image", title="x̂ — image reçue", png=E.png(xh)),
                                  dict(type="image", title="x — originale", png=E.png(x))]))

    views, summary = judge_views(x, xh, label, kind)
    st.append(dict(title="Juges et mesures", node="judge", kind="évaluation", size="PSNR · SSIM · sens",
                   concept="judge",
                   why="Comparez maintenant avec un JSCC au même SNR et au même k/n. À 0 dB sur CIFAR-10, cette "
                       "chaîne tombe au niveau du hasard alors que les JSCC restent exploitables ; à fort SNR, "
                       "elle redevient compétitive. Tout le projet est dans cet écart.",
                   views=views))
    summary.update(k=k, n=n_src, ratio=k / n_src, snr_inst=eff_snr)
    return nodes, st, summary, E.png(xh)


# ----------------------------------------------------------------------------------------------- entry
def run(system, image, snr=10.0, c_active=8, channel="awgn", mcs="QPSK 1/2", seed=0):
    """One full transmission, returned as a list of stages the UI can step through."""
    if system not in ID2SYS:
        raise ValueError(f"système inconnu: {system}")
    sysdef = ID2SYS[system]
    x, label, title = load_image(image)
    kind = image.split(":")[0]
    if sysdef.get("cifar_only") and kind != "cifar":
        raise ValueError("Le système orienté tâche est entraîné sur CIFAR-10 : choisissez une image CIFAR-10.")
    snr, c_active, seed = float(snr), int(c_active), int(seed)

    if system == "task":
        nodes, stages, summary, png = trace_task(x, label, snr, c_active, channel, seed)
    elif system == "sep":
        nodes, stages, summary, png = trace_separated(x, label, kind, snr, c_active, channel, mcs, seed)
    elif system == "sep_cap":
        nodes, stages, summary, png = trace_separated(x, label, kind, snr, c_active, channel, mcs, seed,
                                                      capacity=True)
    else:
        nodes, stages, summary, png = trace_jscc(system, x, label, kind, snr, c_active, channel, seed)

    links = [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)]
    for i, s in enumerate(stages):
        s["id"] = f"s{i}"
        s["n"] = i + 1
    return dict(system=system, label=sysdef["label"], blurb=sysdef["blurb"], image=image, image_title=title,
                snr=snr, c_active=c_active, channel=channel, mcs=mcs if sysdef.get("mcs") else None,
                nodes=nodes, links=links, stages=stages, summary=summary, received=png,
                original=E.png(x))
