"""Defence presentation (English) generated from the evaluation results.   python article/make_presentation.py"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "article"))
os.chdir(ROOT)

from content import R, fmt  # noqa: E402
from slides_helper import Deck  # noqa: E402

FIG = "figures"


def main():
    r = R()
    c, k, v = r.cifar, r.kodak, r.video
    d = Deck("Future Mobile Networks — RT5 — Project 07: learned semantic communication for images and video")
    d.title("Learned semantic communication for image and video transmission",
            "Transmitting meaning instead of pixels: semantic deep JSCC vs separated and syntactic schemes "
            "(simulation study)",
            "Oussema Bouchaala · Mohamed Helmi Lakhdhar · Mohamed Amin Saddoud · Mouine Eldabbabi · Ghaith Gtari",
            "INSAT — RT5 — 2026/2027 — Supervisor: Dr Mériem Afif")
    d.bullets("Problem", [
        "Shannon: reproduce the bits exactly, whatever they mean; Weaver: technical, semantic and effectiveness levels",
        "Today's links stack an image/video codec (JPEG, H.264) with an independent channel code and modulation",
        "  wasteful on a noisy, band-limited channel: resources go to details that carry no meaning",
        "  cliff effect: below the SNR threshold of the chosen MCS the image is lost; above it, quality is capped",
        "Semantic communication (6G): a neural encoder sends a compact representation of the content, "
        "the channel is part of training",
        "Question: what does a semantic loss buy, in pixels and in meaning, against a separated chain and a syntactic "
        "JSCC, on images and on video?",
    ])
    d.image("Three systems, one simulated channel", os.path.join(FIG, "architecture.png"))
    d.bullets("Model", [
        "Image x ∈ [0,1]^(3×H×W), n = 3HW real values → k complex symbols, power (1/k)‖s‖² ≤ 1, ratio k/n",
        "Channel: AWGN y = s + w, or block Rayleigh y = h·s + w (one h per image, perfect CSI, ZF)",
        "Separated: JPEG (Kodak, video) or 4-bit quantised AE (CIFAR) + conv. code K=7 (1/2, 2/3, 3/4) + Gray QAM, "
        "soft Viterbi — simulated bit by bit",
        "  capacity bound: k·log₂(1+SNR) error-free bits for the same source coder",
        "Deep JSCC: CNN encoder (stride 4, 16 latent channels), first c channels sent (k/n = c/96), "
        "SNR ~ U[0, 20] dB in training",
        "  syntactic: MSE   |   semantic: MSE + 0.01·feature loss + 0.003·CE through a frozen classifier A",
        "Meaning judged by models never used in training: classifier B (CIFAR), ImageNet ResNet-18 (Kodak, video)",
    ])
    d.table("Experimental set-up", ["Item", "Value"], [
        ["Data", "CIFAR-10 (train 50k, test 5k), Kodak 24 images 768×512, 3 Xiph CIF clips × 30 frames"],
        ["Bandwidth ratio", "1/48 … 1/6 (main: 1/12)"],
        ["SNR", "0–20 dB (1-dB steps on CIFAR, 2-dB steps on Kodak/video)"],
        ["Digital MCS", "QPSK 1/2, 3/4 · 16-QAM 1/2, 3/4 · 64-QAM 2/3, 3/4; packets ≤ 1024 bits"],
        ["Pixel metrics", "PSNR, SSIM"],
        ["Meaning metrics", "accuracy of judge B; ImageNet feature cosine; top-1 label agreement"],
        ["Training", "PyTorch 2.9, CPU only, Adam + one-cycle, 25 epochs (JSCC), 9 models"],
    ], col_widths=[2.5, 9], size=14)
    rows = []
    for s in (0, 4, 10, 16, 20):
        rows.append([f"{s} dB", fmt(c("sep_best", s)), fmt(c("sep_ae_capacity", s)), fmt(c("jscc_mse", s)),
                     fmt(c("jscc_sem", s)), f"{100 * c('sep_best', s, 'acc'):.0f} %",
                     f"{100 * c('jscc_mse', s, 'acc'):.0f} %", f"{100 * c('jscc_sem', s, 'acc'):.0f} %"])
    d.table("CIFAR-10 results at k/n = 1/12 (simulated AWGN)",
            ["SNR", "Sep. best MCS PSNR", "Capacity bound PSNR", "Syntactic PSNR", "Semantic PSNR",
             "Sep. acc.", "Syntactic acc.", "Semantic acc."], rows, size=13,
            note=f"Accuracy of the independent judge B on clean images: {100 * c('original', 0, 'acc'):.1f} %")
    d.image("CIFAR-10: quality and meaning versus SNR", os.path.join(FIG, "cifar_snr.png"),
            caption="Thin grey curves: one fixed MCS each, showing the cliff effect")
    d.image("CIFAR-10: quality versus bandwidth ratio", os.path.join(FIG, "cifar_bandwidth.png"))
    d.image("Kodak (768×512, never seen in training)", os.path.join(FIG, "kodak_snr.png"))
    d.image("What the receiver sees", os.path.join(FIG, "visual.png"))
    d.image("Video, frame by frame (socle)", os.path.join(FIG, "video_snr.png"))
    d.bullets("Analysis", analysis_bullets(r))
    d.image("Train/test SNR mismatch", os.path.join(FIG, "snr_mismatch.png"), bullets=mismatch_bullets(r))
    d.section("Extensions", "Rayleigh fading · task-oriented transmission · temporal redundancy in video")
    d.image("Extension: block Rayleigh fading", os.path.join(FIG, "rayleigh.png"))
    d.image("Extension: task-oriented transmission", os.path.join(FIG, "task_oriented.png"),
            bullets=task_bullets(r))
    d.image("Extension: latent conditional replenishment", os.path.join(FIG, "video_rate.png"))
    d.bullets("Conclusion and future work", conclusion_bullets(r))
    d.bullets("Demonstration", [
        "python scripts/demo.py --image kodim23 --snr 2 10 18      (visual comparison of the three systems)",
        "python scripts/demo.py --image kodim05 --snr 0 6 12 --mcs \"QPSK 3/4\"   (another MCS: the cliff moves)",
        "python scripts/evaluate.py cifar ; python scripts/make_figures.py   (full re-evaluation)",
        "Code, results and figures: README.md; article: article/article.docx and article/main.tex",
        "All results come from simulated AWGN / Rayleigh channels, not from a radio link",
    ])
    path = os.path.join("article", "Presentation_Projet07_SemCom.pptx")
    d.save(path)
    print("wrote:", path)


# The analysis bullets are filled once the results are known (see content.py for the same reasoning in prose).
def analysis_bullets(r):
    from content import analysis_points
    return analysis_points(r)


def mismatch_bullets(r):
    from content import mismatch_points
    return mismatch_points(r)


def task_bullets(r):
    from content import task_points
    return task_points(r)


def conclusion_bullets(r):
    from content import conclusion_points
    return conclusion_points(r)


if __name__ == "__main__":
    main()
