"""semcom: learned semantic communication for images and video over simulated wireless channels.

Modules
    channel  power normalisation, AWGN and block Rayleigh channels, bandwidth truncation
    models   deep JSCC encoder/decoder (syntactic and semantic), digital auto-encoder, CIFAR classifiers,
             task-oriented transmitter/receiver
    digital  separated digital chain: JPEG / quantised AE source code, convolutional code (K=7) with puncturing,
             Gray QAM, soft Viterbi decoding
    metrics  PSNR, SSIM, semantic metrics (classifier accuracy, ImageNet feature similarity, label agreement)
    data     CIFAR-10, Kodak and Y4M video loaders
    train    training loops
    video    frame-by-frame transmission and latent conditional replenishment (temporal extension)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "data", "torch_home"))


def chdir_project_root():
    os.chdir(ROOT)
    return ROOT
