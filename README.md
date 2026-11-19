# Project 07 — Learned semantic communication for image and video transmission (deep JSCC · AWGN · Rayleigh · video)

An image or a video frame is sent over a **simulated** wireless channel by three systems, all simulated in Python and
PyTorch:

1. a **separated (Shannon) digital chain**: source coding (JPEG, or a 4-bit quantised auto-encoder on CIFAR-10),
   a K=7 convolutional code (rates 1/2, 2/3 and 3/4), Gray QAM (QPSK, 16-QAM, 64-QAM), soft demodulation and Viterbi
   decoding, simulated bit by bit;
2. a **syntactic deep JSCC** (CNN encoder → power-constrained complex symbols → channel layer → CNN decoder) trained on
   pixel MSE;
3. a **semantic deep JSCC** with the same architecture, trained with a loss that keeps the meaning: MSE plus a feature
   term and a task term computed by a frozen CIFAR-10 classifier.

Semantic preservation is judged by models that were **not** used for training: a second CIFAR-10 classifier
(different architecture) and an ImageNet ResNet-18 (feature cosine similarity, top-1 label agreement) for Kodak and
video. Pixel fidelity is measured by PSNR and SSIM. All results come from a simulated channel (AWGN or block Rayleigh),
not from a radio link.

## Installation

Python 3.10+. Tested on Windows 11 with Python 3.11, CPU only (no GPU):

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py   # CIFAR-10, Kodak, 3 Xiph CIF clips, ResNet-18 weights (about 250 MB)
```

## Reproducing the experiments

```bash
python scripts/run_all.py --workers 4 --threads 4   # trains the 9 models (can be relaunched, finished models are skipped)
python scripts/evaluate.py all --workers 6           # CIFAR-10, Kodak and video evaluation -> results/eval/*.csv
python scripts/make_figures.py                       # tables (results/tables) and figures (figures/)
python scripts/demo.py                               # visual comparison on one Kodak image -> figures/visual.png
python article/build_article.py                      # article .docx + IEEEtran LaTeX source
python article/make_presentation.py                  # defence slides (.pptx)
```

Individual pieces:

```bash
python -c "from semcom.train import train; train('jscc_mse', threads=8)"
python scripts/demo.py --image kodim05 --snr 0 6 12 --mcs "QPSK 3/4"
```

On a 16-thread CPU the training campaign takes about 2.5 h and the evaluation about 30 min.

## Interactive explorer

```bash
python scripts/explore.py          # opens http://127.0.0.1:8077/
python scripts/explore.py --port 9000 --no-browser --device cpu
```

A local page that walks through one real transmission, stage by stage: the transmitter, the channel and the
receiver are drawn as a map, and each stage shows what actually crossed that point — the latent tensor as a
stack of channel maps, the symbols in the I/Q plane before and after the noise, the coded bits with the wrong
ones in red, the LLRs, the decoded image and the judges' verdict. Every number comes from `semcom.*`; nothing
is re-simulated or pre-recorded, so a stage shows what `scripts/evaluate.py` measures.

| Tab | What it does |
|---|---|
| Explorer | Pick a system, image, SNR, channel, bandwidth and MCS; step through the stages (space, arrows, play/scrub) |
| Comparer | The same image and settings through several systems at once, received images and metrics side by side |
| Résultats mesurés | Curves read from `results/eval/*.csv` — the campaign, not a live run; thin lines are the fixed MCS |
| Concepts | The course notions used by each stage (SNR, AWGN, Rayleigh, separation, K=7 code, QAM, LLR, Viterbi, cliff effect) |

Systems offered: separated chain (bit-exact), its capacity bound, syntactic JSCC, semantic JSCC, the three
fixed-SNR models of the mismatch study, and the task-oriented transmitter. Trained models are required
(`scripts/run_all.py`); a system whose model is missing is greyed out. The page state lives in the URL, so a
particular stage of a particular transmission can be bookmarked or shared.

Requires nothing beyond `requirements.txt`: the server is `http.server` and the page is plain HTML, CSS and JS.

## Layout

| Path | Contents |
|---|---|
| `semcom/channel.py` | Power normalisation, AWGN, block Rayleigh fading (perfect CSI, ZF), bandwidth truncation |
| `semcom/models.py` | Deep JSCC encoder/decoder, 4-bit digital auto-encoder, CIFAR classifiers A (ResNet) and B (VGG), task-oriented model |
| `semcom/digital.py` | Convolutional code K=7 (133,171) with 802.11a puncturing, Gray QAM, max-log LLRs, batched Viterbi, JPEG rate fitting |
| `semcom/metrics.py` | PSNR, SSIM, ImageNet ResNet-18 semantic metrics |
| `semcom/data.py` | CIFAR-10, Kodak and Y4M video loaders |
| `semcom/train.py` | Training loops and the list of jobs |
| `semcom/video.py` | Frame-by-frame transmission and latent conditional replenishment (temporal extension) |
| `scripts/` | `download_data.py`, `run_all.py`, `evaluate.py`, `make_figures.py`, `demo.py`, `explore.py` |
| `explorer/` | Interactive explorer: `trace.py` (records every stage of one transmission), `server.py`, `static/` |
| `results/` | `models/<name>/{model.pt, train_log.csv, info.json}`, `eval/*.csv`, `tables/summary.{csv,md}`, `logs/` |
| `figures/` | Article figures |
| `article/` | Article (Word + LaTeX) and presentation |

## Systems compared

| System | Source / channel coding | Training loss | Role |
|---|---|---|---|
| Separated, AE (CIFAR-10) | 4-bit quantised auto-encoder + conv. code + QAM | MSE, no channel | classical digital baseline |
| Separated, JPEG (Kodak, video) | JPEG at the highest quality that fits + conv. code + QAM | — | classical digital baseline |
| Capacity bound | same source code, k·log2(1+SNR) error-free bits | — | best any separated scheme with this source coder could do |
| Syntactic JSCC | CNN, 16 latent channels at stride 4, analog symbols | MSE, SNR ~ U[0,20] dB | syntactic JSCC baseline (Bourtsoulatze et al.) |
| Semantic JSCC | same | MSE + 0.01·feature + 0.003·CE through frozen classifier A | proposed semantic system |
| JSCC at fixed SNR (0/10/20 dB) | same, ratio 1/12 | MSE | train/test SNR mismatch study |
| Task-oriented (extension) | stride-8 encoder → classifier head, no reconstruction | cross-entropy | transmit only what the task needs |
| Conditional replenishment (extension) | semantic JSCC; key frame + changed latent blocks only | — | temporal redundancy in video |

Bandwidth ratio k/n: k complex channel symbols for n = 3·H·W real source values. With c active latent channels at stride 4,
k/n = c/96 (c = 2, 4, 8, 12, 16 → 1/48 … 1/6). The main operating point is k/n = 1/12.

## Protocol

- CIFAR-10: 50,000 training images; evaluation on the first 5,000 test images. Judge B (VGG) accuracy measures meaning.
- Kodak: 24 images (portraits rotated to 768×512), never seen in training (models are trained on CIFAR-10 only).
- Video: the first 30 frames of three CIF (352×288) Xiph sequences, akiyo (static), foreman (moderate motion) and
  bus (fast pan). They stand in for the UVG clips suggested in the brief, whose raw 1080p files are several GB each.
- SNR from 0 to 20 dB (1-dB steps on CIFAR-10, 2-dB steps on Kodak and video). Digital points are simulated bit-exactly:
  packets of at most 1024 information bits are coded, modulated, sent over the channel and Viterbi-decoded.
  JPEG files that the decoder rejects become a mid-grey image.
- The "best MCS" curve is a genie that knows the SNR and picks the best modulation and coding scheme. The thin per-MCS curves
  show the cliff effect of a fixed design.
