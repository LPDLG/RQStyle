# RQStyle: Site-Specific Structural Guidance and Reference Adaptation for Diffusion Style Transfer

**Xianlin Peng, Qinfan Ge, Zeyu Bai, Qiyao Hu, Moncef Gabbouj, and Jinye Peng**

RQStyle transfers reference appearance while preserving content structure by **injecting different information at different locations in a diffusion model**. Edges guide ResNet features, low-frequency grayscale refines self-attention Queries, and adapted reference features enter image cross-attention. Only **4.142M parameters** are trained; the SDXL backbone and native IP-Adapter components remain frozen. Inference generates **1024 × 1024** images without content inversion or test-time optimization.

> **Release status:** Code and pretrained weights are being prepared for public release. The commands below apply to the release package once these files are available.

[Method](#method) · [Quick start](#quick-start) · [Training](#training) · [Detailed usage](docs/usage.md) · [Evaluation](docs/evaluation.md)

![Style transfer comparisons. RQStyle is shown in the rightmost column.](assets/comparison.png)

## Method

![RQStyle framework: edge guidance, grayscale Query refinement, and reference Key/Value adaptation.](assets/framework.png)

- **Frequency-Decoupled Edge Guidance Network (FD-EdgeNet)** separates edge-conditioned updates into frequency components at selected ResNet layers, preserving contour guidance while reducing interference with reference textures.
- **Grayscale-Guided Query Refinement (GQR)** uses low-frequency grayscale to refine self-attention Queries for regional layout guidance, while retaining native Key and Value projections.
- **Style-Adaptive Adapter (SA-Adapter)** adds independent residual mappings to frozen IP-Adapter image Keys and Values, adapting reference injection to RQStyle's structural guidance.
- **Shuffled self-reference training** uses the same image as the reconstruction target and reference, with an 8 × 8 patch shuffle applied to the reference. The modules share one noise-prediction loss; inference uses an intact reference.

## Quick start

### Installation

Use a CUDA-capable NVIDIA GPU. The following setup uses Python 3.10 and PyTorch 2.3.0 with CUDA 12.1:

```bash
git clone https://github.com/LPDLG/RQStyle.git
cd RQStyle
conda create -n rqstyle python=3.10 -y
conda activate rqstyle
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

### Inference

The release package uses `checkpoints/rqstyle.pt`, trained for 15,000 iterations. Frozen SDXL, VAE, CLIP, and IP-Adapter weights are downloaded separately through Hugging Face. See [model sources and local paths](docs/usage.md#model-weights) for details.

```bash
python -m rqstyle.inference \
  --content data/content/01.jpg \
  --style data/style/01.jpg \
  --checkpoint checkpoints/rqstyle.pt \
  --output outputs/example.png \
  --prompt "an artwork" \
  --seed 1325412 \
  --device cuda:0
```

Replace the input paths with your own images and use `--prompt` to describe the content. The command saves the generated image and a JSON file recording its settings. Defaults are **50 UniPC steps** and **guidance scale 5.0**.

The example prompt is for quick use. Reproducing the paper's outputs requires its content-specific prompts and evaluation settings; see [evaluation details](docs/evaluation.md).

## Training

Prepare a separate folder of training artworks, excluding evaluation images. Each image supplies its own reconstruction target and shuffled reference; paired stylized targets and captions are not required.

```bash
python -m rqstyle.training \
  --train-dir /path/to/training/artworks \
  --output-dir outputs/train \
  --iterations 15000 \
  --device cuda:0
```

This uses batch size 1 and AdamW with learning rate and weight decay both set to `1e-4`. Checkpoints are saved every 1,000 iterations. See [training details](docs/usage.md#training) for caching, resuming, and initializing from existing weights. Both entry points list available options with `--help`.

## Acknowledgments

Our implementation builds on [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [Diffusers](https://github.com/huggingface/diffusers), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), and [InstantStyle](https://github.com/instantX-research/InstantStyle). We thank the creators of [COCO](https://cocodataset.org/) and [WikiArt](https://www.wikiart.org/) for the source image collections. Third-party code, weights, and images retain their respective usage terms.
