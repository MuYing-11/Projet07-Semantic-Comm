"""Article content (single source for the Word and LaTeX versions). Numbers come from the summary table and the
evaluation CSVs (results/eval/*.csv). Every result is a simulation of a wireless channel."""
import os

import numpy as np
import pandas as pd

TITLE = ("Transmitting Meaning Instead of Pixels: A Simulation Study of Semantic Deep Joint Source-Channel Coding "
         "for Images and Video against Separated and Syntactic Schemes")
AUTHORS = ["Oussema Bouchaala", "Mohamed Helmi Lakhdhar", "Mohamed Amin Saddoud", "Mouine Eldabbabi", "Ghaith Gtari"]
AFFILIATION = ("RT5 — Future Mobile Networks, Institut National des Sciences Appliquées et de Technologie (INSAT), "
               "University of Carthage, Tunis, Tunisia. Supervisor: Dr Mériem Afif.")
KEYWORDS = ["Semantic communication", "Deep joint source-channel coding", "Cliff effect", "Wireless image and video "
            "transmission", "6G", "Deep learning"]

REFERENCES = [
    "C. E. Shannon, “A mathematical theory of communication,” *Bell Syst. Tech. J.*, vol. 27, pp. 379–423, 623–656, "
    "1948.",
    "W. Weaver, “Recent contributions to the mathematical theory of communication,” in C. E. Shannon and W. Weaver, "
    "*The Mathematical Theory of Communication*. Urbana, IL, USA: Univ. of Illinois Press, 1949.",
    "Z. Qin, X. Tao, J. Lu, W. Tong, and G. Y. Li, “Semantic communications: Principles and challenges,” "
    "arXiv:2201.01389, 2021.",
    "H. Xie, Z. Qin, G. Y. Li, and B.-H. Juang, “Deep learning enabled semantic communication systems,” *IEEE Trans. "
    "Signal Process.*, vol. 69, pp. 2663–2675, 2021.",
    "H. Zhang, S. Shao, M. Tao, X. Bi, and K. B. Letaief, “Deep learning-enabled semantic communication systems with "
    "task-unaware transmitter and dynamic data,” *IEEE J. Sel. Areas Commun.*, vol. 41, no. 1, pp. 170–185, 2023 "
    "(arXiv 2022).",
    "S. Wang, J. Dai, Z. Liang, K. Niu, Z. Si, C. Dong, X. Qin, and P. Zhang, “Wireless deep video semantic "
    "transmission,” *IEEE J. Sel. Areas Commun.*, vol. 41, no. 1, pp. 214–229, 2023.",
    "Q. Fu, H. Xie, Z. Qin, G. Slabaugh, and X. Tao, “Vector quantized semantic communication system,” *IEEE Wireless "
    "Commun. Lett.*, vol. 12, no. 6, pp. 982–986, 2023 (arXiv:2209.11519).",
    "A. Mercat, M. Viitanen, and J. Vanne, “UVG dataset: 50/120fps 4K sequences for video codec analysis and "
    "development,” in *Proc. ACM Multimedia Systems Conf. (MMSys)*, 2020, pp. 297–302.",
    "Eastman Kodak, “Kodak lossless true color image suite.” [Online]. Available: https://r0k.us/graphics/kodak/",
    "E. Bourtsoulatze, D. Burth Kurka, and D. Gündüz, “Deep joint source-channel coding for wireless image "
    "transmission,” *IEEE Trans. Cogn. Commun. Netw.*, vol. 5, no. 3, pp. 567–579, 2019.",
    "D. B. Kurka and D. Gündüz, “Bandwidth-agile image transmission with deep joint source-channel coding,” *IEEE Trans. "
    "Wireless Commun.*, vol. 20, no. 12, pp. 8081–8095, 2021.",
    "J. Xu, B. Ai, W. Chen, A. Yang, P. Sun, and M. Rodrigues, “Wireless image transmission using deep source channel "
    "coding with attention modules,” *IEEE Trans. Circuits Syst. Video Technol.*, vol. 32, no. 4, pp. 2315–2328, 2022.",
    "T.-Y. Tung and D. Gündüz, “DeepWiVe: Deep-learning-aided wireless video transmission,” *IEEE J. Sel. Areas "
    "Commun.*, vol. 40, no. 9, pp. 2570–2583, 2022.",
    "J. Shao, Y. Mao, and J. Zhang, “Learning task-oriented communication for edge inference: An information bottleneck "
    "approach,” *IEEE J. Sel. Areas Commun.*, vol. 40, no. 1, pp. 197–211, 2022.",
    "Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to "
    "structural similarity,” *IEEE Trans. Image Process.*, vol. 13, no. 4, pp. 600–612, 2004.",
    "A. Krizhevsky, “Learning multiple layers of features from tiny images,” Tech. Rep., Univ. of Toronto, 2009.",
    "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE CVPR*, 2016, "
    "pp. 770–778.",
    "J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in "
    "*Proc. ECCV*, 2016, pp. 694–711.",
    "A. J. Viterbi, “Error bounds for convolutional codes and an asymptotically optimum decoding algorithm,” *IEEE "
    "Trans. Inf. Theory*, vol. 13, no. 2, pp. 260–269, 1967.",
    "*IEEE Std 802.11a-1999: Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications — "
    "High-Speed Physical Layer in the 5 GHz Band*, IEEE, 1999.",
    "G. K. Wallace, “The JPEG still picture compression standard,” *IEEE Trans. Consum. Electron.*, vol. 38, no. 1, "
    "pp. xviii–xxxiv, 1992.",
    "K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in *Proc. "
    "ICLR*, 2015.",
    "D. Gündüz, Z. Qin, I. E. Aguerri, H. S. Dhillon, Z. Yang, A. Yener, K. K. Wong, and C.-B. Chae, “Beyond "
    "transmitting bits: Context, semantics, and task-oriented communications,” *IEEE J. Sel. Areas Commun.*, vol. 41, "
    "no. 1, pp. 5–41, 2023.",
    "J. Ballé, V. Laparra, and E. P. Simoncelli, “End-to-end optimized image compression,” in *Proc. ICLR*, 2017.",
    "J. Dai, S. Wang, K. Tan, Z. Si, X. Qin, K. Niu, and P. Zhang, “Nonlinear transform source-channel coding for "
    "semantic communications,” *IEEE J. Sel. Areas Commun.*, vol. 40, no. 8, pp. 2300–2316, 2022.",
    "Xiph.org, “Derf's test media collection.” [Online]. Available: https://media.xiph.org/video/derf/",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "results", "eval")
MAIN = 1 / 12


def fmt(x, d=1):
    return f"{x:.{d}f}"


class R:
    """Accessor on the evaluation CSVs (all at k/n = 1/12 and AWGN unless stated)."""

    def __init__(self, t=None):
        rd = lambda n: pd.read_csv(os.path.join(EV, n)) if os.path.exists(os.path.join(EV, n)) else pd.DataFrame()
        self.an = pd.concat([rd("cifar_analog.csv"), rd("cifar_capacity.csv")], ignore_index=True)
        self.dg = rd("cifar_digital.csv")
        self.kd = pd.concat([rd("kodak_jscc.csv"), rd("kodak_sep.csv")], ignore_index=True)
        self.vd = pd.concat([rd("video_jscc.csv"), rd("video_sep.csv")], ignore_index=True)
        self.vf = rd("video_frames.csv")
        self.t = t

    @staticmethod
    def _sel(df, scheme, snr, ratio=MAIN, channel="awgn"):
        return df[(df.scheme == scheme) & (df.snr == snr) & np.isclose(df.ratio, ratio, rtol=1e-3) &
                  (df.channel == channel)]

    def cifar(self, scheme, snr, k="psnr", ratio=MAIN, channel="awgn"):
        if scheme == "original":
            return float(self.an[self.an.scheme == "original"][k].iloc[0])
        if scheme == "sep_best":
            d = self._sel(self.dg, "sep_ae", snr, ratio, channel)
            return float(d[k].max())
        if scheme.startswith("mcs:"):
            d = self._sel(self.dg, "sep_ae", snr, ratio, channel)
            return float(d[d.mcs == scheme[4:]][k].iloc[0])
        d = self._sel(self.an, scheme, snr, ratio, channel)
        return float(d[k].iloc[0]) if len(d) else float("nan")

    def nat(self, df, scheme, snr, k="psnr", ratio=MAIN, channel="awgn", tags=None):
        if tags is not None:
            df = df[df.tag.isin(tags)]
        if scheme == "sep_best":
            d = self._sel(df, "sep_jpeg", snr, ratio, channel)
            return float(d.groupby("mcs")[k].mean().max()) if len(d) else float("nan")
        if scheme.startswith("mcs:"):
            d = self._sel(df, "sep_jpeg", snr, ratio, channel)
            return float(d[d.mcs == scheme[4:]][k].mean())
        d = self._sel(df, scheme, snr, ratio, channel)
        return float(d[k].mean()) if len(d) else float("nan")

    def kodak(self, scheme, snr, k="psnr", **kw):
        return self.nat(self.kd, scheme, snr, k, **kw)

    def video(self, scheme, snr, k="psnr", **kw):
        return self.nat(self.vd, scheme, snr, k, **kw)

    def crossover(self, a, b, k="psnr", data="cifar"):
        """Lowest SNR from which scheme b is at least as good as scheme a (None if never)."""
        f = self.cifar if data == "cifar" else self.kodak
        snrs = range(0, 21) if data == "cifar" else range(0, 21, 2)
        for s in snrs:
            if f(b, s, k) >= f(a, s, k):
                return s
        return None


def static_blocks():
    B = [("h1", "Introduction")]
    B.append(("p", "Shannon's theory treats communication as the exact reproduction, at the receiver, of a sequence of "
                   "symbols chosen at the transmitter, regardless of what they mean [1]. Weaver already distinguished "
                   "three levels of communication problems — technical, semantic and effectiveness — of which only the "
                   "first is addressed by today's systems [2]. A modern image or video link stacks a source codec "
                   "(JPEG, H.264) and an independent channel code and modulation, separately optimised. By the "
                   "separation theorem this is optimal for infinitely long blocks and ergodic channels, but in practice "
                   "it spends resources reproducing details that carry little meaning, and it suffers from the *cliff "
                   "effect*: once the SNR falls below the threshold of the chosen modulation and coding scheme (MCS), "
                   "the channel decoder fails and the image is lost, while above the threshold the quality stays capped "
                   "by the source codec."))
    B.append(("p", "Semantic communication, enabled by deep learning, is one of the pillars envisaged for 6G [3], [23]. "
                   "A neural encoder maps the source to a compact set of channel symbols that keeps what matters for "
                   "understanding it, the channel is simulated inside the network during training, and the decoder "
                   "reconstructs the content or directly extracts the needed information [4], [5]. Deep joint "
                   "source-channel coding (deep JSCC) [10] showed that such analog schemes degrade gracefully with the "
                   "SNR. Most JSCC work, however, is still *syntactic*: it optimises pixel fidelity (MSE/PSNR), so its "
                   "benefit in terms of meaning is rarely measured with an independent judge, and video studies rely on "
                   "heavy architectures [6], [13]."))
    B.append(("p", "This work implements and compares, under one simulation framework, the three paradigms named in the "
                   "project brief: (i) a separated digital chain with a real source code (JPEG [21] or a quantised "
                   "auto-encoder), a K=7 convolutional code with puncturing [20], Gray QAM and soft Viterbi decoding [19], "
                   "simulated bit by bit; (ii) a syntactic deep JSCC trained on MSE; and (iii) a semantic deep JSCC "
                   "with the same architecture, trained with a perceptual/semantic loss built on a frozen classifier "
                   "[18]. Meaning is measured by models never used in training: a second CIFAR-10 classifier with a "
                   "different architecture, and an ImageNet ResNet-18 [17] for high-resolution images and video."))
    B.append(("p", "Our contributions are: (1) a reproducible, CPU-only Python framework with a bit-exact digital "
                   "baseline and a capacity bound; (2) a comparison of the three schemes on CIFAR-10 [16], Kodak [9] and "
                   "three video clips with pixel (PSNR, SSIM [15]) and meaning metrics as a function of SNR and bandwidth "
                   "ratio, which exposes the cliff effect; (3) a study of the train/test SNR mismatch; and (4) the "
                   "extensions suggested by the brief: block Rayleigh fading, task-oriented transmission [14] and the "
                   "exploitation of temporal redundancy in video by latent conditional replenishment. Section II "
                   "reviews related work, Section III formalises the system, Section IV presents the implementation "
                   "and results, and Section V concludes."))

    B.append(("h1", "Related Work"))
    B.append(("p", "**Semantic communication.** Qin et al. [3] and Gündüz et al. [23] survey the shift from bit-level to "
                   "semantic and task-oriented communication and stress that a semantic system must be judged by the "
                   "meaning recovered at the receiver, not only by the bit or pixel error. DeepSC [4] introduced a "
                   "Transformer-based semantic transceiver for text, trained with a loss combining cross-entropy and "
                   "mutual information, and showed robustness at low SNR. Zhang et al. [5] considered a task-unaware "
                   "transmitter and dynamic data, transferring knowledge to the receiver side. VQ-DeepSC [7] makes "
                   "semantic features compatible with digital links by vector quantisation over a shared codebook, "
                   "i.e., a knowledge base shared by both ends."))
    B.append(("p", "**Deep JSCC for images.** Bourtsoulatze et al. [10] proposed the convolutional deep JSCC that our "
                   "syntactic baseline follows: a fully convolutional encoder whose output is power-normalised and "
                   "sent as analog complex symbols; it beats JPEG/JPEG2000 with capacity-achieving codes at low SNR and "
                   "avoids the cliff effect. Kurka and Gündüz [11] made the scheme bandwidth-agile by ordering latent "
                   "channels, which we adopt to obtain several bandwidth ratios from one model. ADJSCC [12] feeds the "
                   "SNR to attention modules to adapt one model to all SNRs, and NTSCC [25] couples a learned "
                   "nonlinear transform [24] with an entropy model to allocate symbols where the content is complex."))
    B.append(("p", "**Video and tasks.** DeepWiVe [13] and DVST [6] extend JSCC to video with motion compensation, "
                   "key frames and adaptive rate allocation and report large gains over H.264/H.265 with LDPC. "
                   "Task-oriented communication [14] removes the reconstruction altogether and transmits only the "
                   "features a receiver-side inference needs, trading reconstruction for drastically fewer symbols."))
    B.append(("p", "**Positioning.** Most JSCC papers compare against a separated scheme evaluated with a capacity "
                   "assumption and report PSNR only. We (i) simulate a practical digital chain bit by bit, next to its "
                   "capacity bound, (ii) separate the training loss (classifier A) from the semantic judge (classifier B "
                   "or ImageNet ResNet-18) to avoid a self-fulfilling metric, and (iii) evaluate the same models on "
                   "images and video, including fading and bandwidth trade-offs, with a CPU-scale budget."))

    B.append(("h1", "System Model and Proposed Approach"))
    B.append(("h2", "Channel model and bandwidth ratio"))
    B.append(("p", "An RGB image $x \\in [0,1]^{3 \\times H \\times W}$ has $n = 3HW$ real source values. A transmitter "
                   "maps it to $k$ complex channel symbols $s \\in \\mathbb{C}^k$ under the average power constraint "
                   "$\\frac{1}{k}\\|s\\|^2 \\le 1$; the bandwidth (compression) ratio is $k/n$. The channel is either AWGN "
                   "or slow block Rayleigh fading with one coefficient $h \\sim \\mathcal{CN}(0,1)$ per image, perfect "
                   "channel knowledge and zero-forcing equalisation at the receiver:"))
    B.append(("eq", r"y = h\,s + w,\quad w \sim \mathcal{CN}(0, \sigma^2 I_k),\quad \mathrm{SNR} = 1/\sigma^2"))
    B.append(("p", "with $h = 1$ for AWGN. All schemes are compared at the same $k/n$ and the same SNR."))
    B.append(("h2", "Separated digital chain (baseline)"))
    B.append(("p", "The source coder is JPEG for natural images, at the highest quality whose file fits in the bit "
                   "budget, and, for 32×32 CIFAR-10 images where JPEG headers alone exceed the budget, an auto-encoder "
                   "whose bounded latent is uniformly quantised on 4 bits (16 latent channels at stride 4, ordered by "
                   "importance, so the number of transmitted channels sets the bit rate). Bits are split into packets "
                   "of at most 1024 bits, each coded by the K=7 (133,171) convolutional code of IEEE 802.11a [20], "
                   "punctured to rates 2/3 or 3/4, Gray-mapped on QPSK, 16-QAM or 64-QAM, and decoded with max-log "
                   "LLRs and soft Viterbi decoding [19]. For a given MCS of spectral efficiency $\\eta$ the bit budget "
                   "is $B \\approx \\eta k$. We report each fixed MCS, the envelope obtained by a genie choosing the best "
                   "MCS for each SNR, and a *capacity bound* in which the source coder receives the error-free budget"))
    B.append(("eq", r"B^{\star} = k \log_2(1 + \mathrm{SNR})"))
    B.append(("p", "which no practical separated scheme with the same source coder can exceed."))
    B.append(("h2", "Syntactic and semantic deep JSCC"))
    B.append(("p", "The encoder $f_\\theta$ (two stride-2 5×5 convolutions, two residual blocks, PReLU) outputs a latent "
                   "$z \\in \\mathbb{R}^{16 \\times H/4 \\times W/4}$; the decoder $g_\\phi$ mirrors it with transposed "
                   "convolutions and a sigmoid. Pairs of latent values form complex symbols. Only the first $c$ "
                   "channels are sent, so $k/n = c/96$ ($c = 2,4,8,12,16$ gives 1/48 to 1/6), and the transmitted "
                   "symbols are normalised to unit average power:"))
    B.append(("eq", r"s = \sqrt{k}\,\frac{z_{1:c}}{\|z_{1:c}\|},\qquad \hat{x} = g_\phi(h^{-1} y)"))
    B.append(("p", "During training, $c$ is drawn at random for each mini-batch (bandwidth agility [11]) and the SNR of "
                   "each image is drawn uniformly in [0, 20] dB, so a single model serves all bandwidths and SNRs "
                   "without channel feedback. The *syntactic* JSCC minimises the pixel MSE. The *semantic* JSCC, with "
                   "the same architecture and budget, minimises"))
    B.append(("eq", r"\mathcal{L} = \|x-\hat{x}\|^2 + \lambda_f \sum_{l=1}^{3} \frac{\|\varphi_l(\hat{x})-\varphi_l(x)\|^2}{\|\varphi_l(x)\|^2} + \lambda_t\,\mathrm{CE}\left(A(\hat{x}), y\right)"))
    B.append(("p", "where $\\varphi_l$ are the feature maps of the three stages of a frozen CIFAR-10 classifier $A$ "
                   "(small ResNet), the second term is a perceptual loss [18] that preserves the content the "
                   "classifier relies on, and the third is the downstream task loss, with $\\lambda_f = 0.01$ and "
                   "$\\lambda_t = 0.003$ (chosen so that the semantic terms and the MSE have the same order of "
                   "magnitude at 10 dB). Meaning is judged by an independent classifier $B$ (VGG-type [22]) on CIFAR-10 "
                   "and, for Kodak and video, by the cosine similarity of the 512-d features of an ImageNet ResNet-18 "
                   "and by the agreement of its top-1 label between the original and the received image."))
    B.append(("fig", os.path.join("figures", "architecture.png"), "The three systems compared: (a) separated digital "
                     "chain simulated bit by bit, (b) syntactic deep JSCC trained on MSE, (c) semantic deep JSCC with "
                     "the same transceiver, trained with feature and task terms computed by a frozen classifier A.", True))
    B.append(("h2", "Extensions"))
    B.append(("bullets", [
        "**Train/test SNR mismatch.** Syntactic JSCC models trained at a single SNR (0, 10 or 20 dB) are tested from "
        "0 to 20 dB and compared with the model trained on the whole range.",
        "**Block Rayleigh fading.** The AWGN-trained models are evaluated on the fading channel; the digital chain "
        "keeps the MCS chosen for the *average* SNR, since a transmitter without instantaneous channel knowledge "
        "cannot adapt it.",
        "**Task-oriented transmission.** A stride-8 encoder sends $c \\in \\{1,..,16\\}$ channels (8 to 128 symbols "
        "for a CIFAR image) to a receiver-side classifier head, trained with cross-entropy only: the receiver obtains "
        "the class, not the image.",
        "**Temporal redundancy (video).** In *latent conditional replenishment* (CR), the first frame of each group of "
        "10 frames is a key frame sent with $c_k$ channels. For the next frames the transmitter computes, per 2×2 "
        "latent block (an 8×8-pixel macroblock), the energy of the change since the block was last sent, and sends "
        "only the fraction $f$ of blocks with the largest change. The receiver overwrites those blocks in its stored "
        "latent and decodes. The block map costs $N_b\\,H_2(f)$ bits, sent at 1 bit per symbol and counted in the "
        "rate. CR is compared with frame-by-frame transmission at the same average $k/n$.",
    ]))
    return B
