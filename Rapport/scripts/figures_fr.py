"""Régénère en français, dans Rapport/figures/, les figures utilisées par le rapport.

Les scripts du projet (scripts/make_figures.py, scripts/demo.py) ne sont pas modifiés : ils produisent les
figures en anglais pour l'article IEEE. Ce script les importe, redirige leur dossier de sortie et traduit les
textes au moment où matplotlib les pose, de sorte qu'il n'existe qu'une seule implémentation des figures.

    python Rapport/scripts/figures_fr.py        # depuis la racine du projet

Les chiffres ne sont pas recalculés : les figures sont tracées à partir de results/eval/*.csv.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.path.join(ROOT, "Rapport", "figures")

# ------------------------------------------------------------------ table de traduction
FR = {
    # axes
    "SNR (dB)": "SNR du canal (dB)",
    "Test SNR (dB)": "SNR de test (dB)",
    "Average SNR (dB)": "SNR moyen (dB)",
    "PSNR (dB)": "PSNR (dB)",
    "Bandwidth ratio k/n": "Rapport de bande k/n",
    "Average bandwidth ratio k/n": "Rapport de bande moyen k/n",
    "Classification accuracy": "Précision de classification",
    "Accuracy of judge B": "Précision du juge B",
    "Complex channel symbols per image": "Symboles complexes par image",
    "ImageNet feature cosine similarity": "Similarité cosinus des caractéristiques ImageNet",
    "Frame index": "Numéro de trame",
    "Epoch": "Époque",
    # titres
    "CIFAR-10, bandwidth ratio k/n = 1/12, AWGN (simulation)":
        "CIFAR-10, rapport de bande k/n = 1/12, canal AWGN (simulation)",
    "CIFAR-10: quality versus compression (bandwidth) ratio at three SNRs (simulation)":
        "CIFAR-10 : qualité en fonction du rapport de bande, à trois SNR (simulation)",
    "Train/test SNR mismatch, syntactic JSCC, k/n = 1/12":
        "Désadaptation train/test du SNR, JSCC syntaxique, k/n = 1/12",
    "Extension: task-oriented vs reconstruct-then-classify":
        "Extension : orienté tâche contre « reconstruire puis classer »",
    "Semantic JSCC, 10 dB: per-frame quality": "JSCC sémantique à 10 dB : qualité trame par trame",
    "CIFAR-10 (dotted: AWGN)": "CIFAR-10 (pointillés : AWGN)",
    "Kodak": "Kodak",
    # légendes et annotations
    "clean images": "images originales",
    "thin grey: individual MCS": "en gris fin : chaque MCS",
    "trained at 0 dB": "entraîné à 0 dB",
    "trained at 10 dB": "entraîné à 10 dB",
    "trained at 20 dB": "entraîné à 20 dB",
    "trained on U[0,20] dB": "entraîné sur U[0,20] dB",
    "original": "originale",
    "Syntactic JSCC": "JSCC syntaxique",
    "Semantic JSCC": "JSCC sémantique",
    "Separated JPEG + 16QAM 1/2": "Séparé JPEG + 16QAM 1/2",
    # schéma d'architecture
    "(a) Separated digital chain": "(a) Chaîne numérique séparée",
    "(b) Syntactic deep JSCC": "(b) JSCC profond syntaxique",
    "(c) Semantic deep JSCC": "(c) JSCC profond sémantique",
    "Source coder\nJPEG or 4-bit AE": "Codage source\nJPEG ou AE 4 bits",
    "Conv. code K=7\nrate 1/2–3/4": "Code conv. K=7\nrendement 1/2–3/4",
    "Gray QAM\nQPSK/16/64": "QAM de Gray\nQPSK/16/64",
    "Soft demod.\n+ Viterbi": "Démod. souple\n+ Viterbi",
    "Source\ndecoder": "Décodage\nsource",
    "CNN encoder\n(stride 4)": "Encodeur CNN\n(stride 4)",
    "Truncate to c\nchannels": "Troncature à c\ncanaux",
    "Power norm.\n‖x‖² = k": "Norm. puissance\n‖x‖² = k",
    "CNN decoder": "Décodeur CNN",
    "x̂ → frozen\nclassifier A": "x̂ → classifieur A\ngelé",
}

PREFIX_FR = [
    ("Kodak (24 images, 768×512), k/n = 1/12, AWGN (simulation, models trained on CIFAR-10)",
     "Kodak (24 images, 768×512), k/n = 1/12, AWGN (simulation ; modèles entraînés sur CIFAR-10)"),
    ("Video: 3 CIF clips × 30 frames, frame-by-frame, k/n = 1/12, AWGN (simulation)",
     "Vidéo : 3 clips CIF × 30 trames, image par image, k/n = 1/12, AWGN (simulation)"),
    ("Extension: block Rayleigh fading, k/n = 1/12, perfect CSI at the receiver (simulation). ",
     "Extension : évanouissement de Rayleigh par blocs, k/n = 1/12, CSI parfaite au récepteur (simulation). "),
    ("Extension: temporal redundancy by latent conditional replenishment (CR), SNR 10 dB (simulation)",
     "Extension : redondance temporelle par renouvellement conditionnel des latents, SNR 10 dB (simulation)"),
    ("Training losses:", "Pertes d'entraînement :"),
]

# Titres construits par f-string dans demo.py : traduits par morceaux.
PIECES_FR = [
    ("sem. cos", "cos. sém."),
    (", bandwidth ratio k/n = ", ", rapport de bande k/n = "),
    (", simulated AWGN channel", ", canal AWGN simulé"),
]


def tr(s):
    """Traduit une chaîne affichée ; laisse inchangé ce qui n'est pas dans la table."""
    if not isinstance(s, str) or not s.strip():
        return s
    if s in FR:
        return FR[s]
    for en, fr in PREFIX_FR:
        if s.startswith(en):
            return fr + s[len(en):]
    out = s
    for en, fr in PIECES_FR:
        out = out.replace(en, fr)
    for en, fr in [(" dB", " dB"), ("Separated", "Séparé"), ("Semantic", "Sémantique"),
                   ("Syntactic", "Syntaxique"), ("capacity bound", "borne de capacité"),
                   ("best MCS", "meilleur MCS"), ("Task-oriented", "Orienté tâche"),
                   ("channel simulated inside the network", "canal simulé à l'intérieur du réseau"),
                   ("frames", "trames"), ("frame", "trame")]:
        out = out.replace(en, fr)
    return out


def patch():
    """Traduit les textes au moment où matplotlib les pose."""
    import matplotlib.axes as maxes
    import matplotlib.figure as mfigure

    for cls, name in [(maxes.Axes, "set_xlabel"), (maxes.Axes, "set_ylabel"), (maxes.Axes, "set_title"),
                      (mfigure.Figure, "suptitle")]:
        orig = getattr(cls, name)

        def wrapper(self, s, *a, _o=orig, **kw):
            return _o(self, tr(s), *a, **kw)
        setattr(cls, name, wrapper)

    orig_text = maxes.Axes.text

    def text(self, x, y, s, *a, **kw):
        return orig_text(self, x, y, tr(s), *a, **kw)
    maxes.Axes.text = text

    orig_plot = maxes.Axes.plot

    def plot(self, *a, **kw):
        if isinstance(kw.get("label"), str):
            kw["label"] = tr(kw["label"])
        return orig_plot(self, *a, **kw)
    maxes.Axes.plot = plot

    # Les légendes denses chevauchent les courbes : on leur donne un fond opaque et une place automatique.
    orig_legend = maxes.Axes.legend

    def legend(self, *a, **kw):
        if kw.get("fontsize", 10) <= 7:
            kw.setdefault("frameon", True)
            kw.setdefault("framealpha", 0.92)
            kw.setdefault("facecolor", "white")
            kw.setdefault("edgecolor", "#e4e3df")
            kw["loc"] = "best"
        return orig_legend(self, *a, **kw)
    maxes.Axes.legend = legend


def main():
    os.makedirs(OUT, exist_ok=True)
    patch()

    from scripts import make_figures as mf

    mf.FIG = OUT
    mf.LABEL = {"jscc_mse": "JSCC syntaxique (MSE)", "jscc_sem": "JSCC sémantique (proposé)",
                "sep": "Séparé : meilleur MCS", "cap": "Séparé : borne de capacité",
                "task": "Orienté tâche (ext.)"}

    an = pd.concat([mf.read("cifar_analog.csv"), mf.read("cifar_capacity.csv")], ignore_index=True)
    dg = mf.read("cifar_digital.csv")
    kd = pd.concat([mf.read("kodak_jscc.csv"), mf.read("kodak_sep.csv")], ignore_index=True)
    v = pd.concat([mf.read("video_jscc.csv"), mf.read("video_sep.csv")], ignore_index=True)
    vf = mf.read("video_frames.csv")

    mf.fig_architecture()
    mf.fig_learning()
    mf.fig_cifar(an, dg)
    mf.fig_bandwidth(an, dg)
    mf.fig_mismatch(an)
    mf.fig_task(an)
    mf.fig_natural(kd, "Kodak (24 images, 768×512), k/n = 1/12, AWGN (simulation, models trained on CIFAR-10)",
                   os.path.join(OUT, "kodak_snr.png"))
    mf.fig_rayleigh(an, dg, kd)
    mf.fig_natural(v, "Video: 3 CIF clips × 30 frames, frame-by-frame, k/n = 1/12, AWGN (simulation)",
                   os.path.join(OUT, "video_snr.png"))
    mf.fig_video_rate(v, vf)

    # Comparaison visuelle : produite par scripts/demo.py, qui a besoin des données Kodak et des modèles.
    try:
        from scripts import demo

        sys.argv = ["demo.py", "--out", os.path.join(OUT, "visual.png")]
        demo.main()
    except Exception as exc:  # données ou modèles absents : la figure anglaise existante reste en place
        print(f"visual.png non régénérée ({type(exc).__name__}: {exc})")

    print("figures françaises écrites dans", OUT)


if __name__ == "__main__":
    main()
