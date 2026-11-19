"""What the explorer offers: systems, images, options — and the cached models behind them."""
import functools
import os

import torch

from semcom import ROOT
from semcom import digital as dg
from semcom.train import JOBS, load, pick_device

CIFAR_CLASSES = ["avion", "automobile", "oiseau", "chat", "cerf", "chien", "grenouille", "cheval", "bateau", "camion"]
C_WIDTHS = [2, 4, 8, 12, 16]  # active latent channels -> k/n = 1/48, 1/24, 1/12, 1/8, 1/6
MAIN_C = 8  # k/n = 1/12, the operating point of the article

# Each system is one transmission chain. `needs` lists the trained models it requires.
SYSTEMS = [
    dict(id="sep", group="Chaîne séparée (Shannon)", label="Séparé — source + code + QAM",
         short="Séparé (réel)", needs=["dae"], digital=True, mcs=True,
         blurb="La chaîne d'aujourd'hui, simulée bit à bit : compression, code convolutif K=7, QAM de Gray, "
               "canal, LLR, Viterbi, décompression. C'est ici qu'on voit l'effet falaise."),
    dict(id="sep_cap", group="Chaîne séparée (Shannon)", label="Séparé — borne de capacité",
         short="Séparé (borne)", needs=["dae"], digital=True, mcs=False,
         blurb="Le même compresseur, mais on suppose k·log2(1+SNR) bits transmis sans aucune erreur. "
               "Personne ne peut faire mieux avec ce codage de source : c'est une borne, pas un système."),
    dict(id="jscc_mse", group="JSCC appris", label="JSCC syntaxique — perte MSE",
         short="JSCC syntaxique", needs=["jscc_mse"], digital=False, mcs=False,
         blurb="Plus de bits : l'encodeur CNN produit directement des symboles analogiques, le canal est une "
               "couche du réseau, et tout est appris de bout en bout avec une perte pixel."),
    dict(id="jscc_sem", group="JSCC appris", label="JSCC sémantique — MSE + sens",
         short="JSCC sémantique", needs=["jscc_sem"], digital=False, mcs=False,
         blurb="Architecture identique au JSCC syntaxique. Seule la perte change : MSE + 0,01·caractéristiques "
               "+ 0,003·entropie croisée, à travers un classifieur gelé."),
    dict(id="jscc_mse_snr0", group="Désadaptation de SNR", label="JSCC entraîné à 0 dB seulement",
         short="Entraîné 0 dB", needs=["jscc_mse_snr0"], digital=False, mcs=False,
         blurb="Entraîné uniquement à 0 dB. Il a appris à se méfier du canal : testez-le à 20 dB, il n'exploite "
               "pas la marge disponible."),
    dict(id="jscc_mse_snr10", group="Désadaptation de SNR", label="JSCC entraîné à 10 dB seulement",
         short="Entraîné 10 dB", needs=["jscc_mse_snr10"], digital=False, mcs=False,
         blurb="Entraîné uniquement à 10 dB. Excellent autour de son point de conception, moins ailleurs."),
    dict(id="jscc_mse_snr20", group="Désadaptation de SNR", label="JSCC entraîné à 20 dB seulement",
         short="Entraîné 20 dB", needs=["jscc_mse_snr20"], digital=False, mcs=False,
         blurb="Entraîné uniquement à 20 dB. Testez-le à 0 dB : il s'écroule, parce qu'il n'a jamais vu de bruit "
               "pendant son apprentissage."),
    dict(id="task", group="Extensions", label="Orienté tâche — on n'envoie que la classe",
         short="Orienté tâche", needs=["task"], digital=False, mcs=False, cifar_only=True,
         blurb="Aucune image n'est reconstruite. L'encodeur stride-8 envoie vers une tête de classification : "
               "k/n descend à 1/384 pour une précision comparable au JSCC complet."),
]

ID2SYS = {s["id"]: s for s in SYSTEMS}


def device():
    return pick_device()


@functools.lru_cache(maxsize=None)
def model(name):
    """Trained model, cached. Learned chains run on the GPU when there is one; the digital chain stays on CPU."""
    dev = "cpu" if name == "dae" else str(device())
    return load(name, dev)


@functools.lru_cache(maxsize=1)
def cifar_test():
    from semcom.data import cifar10

    x, y = cifar10(False)
    return x, y


@functools.lru_cache(maxsize=1)
def kodak_images():
    from semcom.data import kodak

    x, names = kodak()
    return x, names


@functools.lru_cache(maxsize=1)
def video_first_frames():
    from semcom.data import video_clips

    clips = video_clips(n_frames=2)
    return {name: fr[0] for name, fr in clips.items()} if isinstance(clips, dict) else \
        {str(i): fr[0] for i, fr in enumerate(clips)}


def trained(name):
    return os.path.exists(os.path.join(ROOT, "results", "models", name, "info.json"))


def image_choices(n_cifar=12):
    """The images the UI offers, as {key, label, group}."""
    out = []
    try:
        _, y = cifar_test()
        for i in range(n_cifar):
            out.append(dict(key=f"cifar:{i}", group="CIFAR-10 (32×32, vu à l'entraînement)",
                            label=f"#{i} · {CIFAR_CLASSES[int(y[i])]}"))
    except Exception:
        pass
    try:
        _, names = kodak_images()
        for i, nm in enumerate(names):
            out.append(dict(key=f"kodak:{i}", group="Kodak (768×512, jamais vu)", label=str(nm)))
    except Exception:
        pass
    try:
        for nm in video_first_frames():
            out.append(dict(key=f"video:{nm}", group="Vidéo CIF (première image)", label=str(nm)))
    except Exception:
        pass
    return out


def load_image(key):
    """'cifar:3' | 'kodak:0' | 'video:akiyo' -> (x (1,3,H,W) float, label|None, title)."""
    kind, _, ref = key.partition(":")
    if kind == "cifar":
        x, y = cifar_test()
        i = int(ref)
        return x[i:i + 1].clone(), int(y[i]), f"CIFAR-10 #{i} ({CIFAR_CLASSES[int(y[i])]})"
    if kind == "kodak":
        x, names = kodak_images()
        i = int(ref)
        return x[i:i + 1].clone(), None, f"Kodak {names[i]}"
    if kind == "video":
        fr = video_first_frames()
        return fr[ref][None].clone(), None, f"Vidéo {ref} (image 0)"
    raise ValueError(f"image inconnue: {key}")


def catalog():
    systems = []
    for s in SYSTEMS:
        systems.append(dict({k: v for k, v in s.items() if k != "needs"},
                            available=all(trained(n) for n in s["needs"]),
                            missing=[n for n in s["needs"] if not trained(n)]))
    return dict(
        systems=systems,
        images=image_choices(),
        mcs=[dg.mcs_name(m) for m in dg.MCS],
        widths=[dict(c=c, ratio=f"1/{round(96 / c)}") for c in C_WIDTHS],
        channels=[dict(id="awgn", label="AWGN (bruit gaussien)"), dict(id="rayleigh", label="Rayleigh par blocs")],
        device=str(device()),
        torch=torch.__version__,
        jobs=sorted(JOBS),
        classes=CIFAR_CLASSES,
    )
