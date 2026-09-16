# RQStyle

**ResNet-Query Modulation for Structure-Aware Diffusion Style Transfer**

RQStyle transfers the appearance of a reference image while conditioning generation
on the structure of a content image. It trains **4.14M parameters** on top of frozen
SDXL and IP-Adapter backbones and generates at **1024 x 1024** without content
inversion, test-time optimization, or image post-processing.

This compact release contains the final model, training and inference code, and
the inputs from our final COCO-to-WikiArt comparison. Experiment schedulers,
ablation variants, metric scripts, logs, and generated results are not included.

## Method

- **FDRM:** edge-conditioned ResNet gain and bias, null-response calibration,
  and low/high-frequency routing at six decoder sites.
- **LFQM:** low-frequency grayscale guidance for gain-and-bias modulation of
  11 aligned self-attention Query projections; native Key and Value are retained.
- **RKVCA:** independent residual adaptation of image-condition Key and Value.
- **Training:** the target artwork is also the reference, with an 8 x 8 patch
  shuffle applied only to the training reference. The only objective is diffusion
  noise-prediction MSE. At inference, the reference is not shuffled.

The included checkpoint is the original final 15,000-iteration model, with both
gain and bias and both Key and Value adaptation. It is not a retrained control
or a gain-only / Value-only variant.

## Installation

Use Python 3.9-3.12 and a CUDA-capable NVIDIA GPU (the release smoke tests used
Python 3.9.20). For a Python 3.10 environment, run from the repository root:

```bash
conda create -n rqstyle python=3.10 -y
conda activate rqstyle
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The implementation pins Diffusers 0.30.0, Transformers 4.44.0, and Accelerate
0.33.0. Native Diffusers attention uses PyTorch SDPA; xFormers is not required.

## Weights

The small RQStyle checkpoint is included at `checkpoints/rqstyle.pt`. Frozen
backbone weights are downloaded separately from Hugging Face on first use:

| Component | Source |
|---|---|
| SDXL | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| VAE | [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) |
| Image encoder | [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) |
| IP-Adapter | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter), `sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` |

Only load checkpoints from trusted sources. Backbone weights are not included in
this repository.

## Inference

```bash
python -m rqstyle.inference \
  --content data/content/01.jpg \
  --style data/style/01.jpg \
  --output outputs/example.png \
  --prompt "an artwork" \
  --device cuda:0
```

The default checkpoint is `checkpoints/rqstyle.pt`. To use a newly trained model,
add `--checkpoint outputs/train/checkpoints/rqstyle_iter15000.pt`.

Defaults are 50 UniPC steps, guidance scale 5.0, and seed 1325412. The command
writes the image and a JSON sidecar with the inference settings. Run commands
from the repository root, or provide an explicit checkpoint path.

`an artwork` is a quick-start prompt, not the archived content-specific captions
used for comparison tables. Exact archival reproduction additionally requires
the original prompts, negative prompts, seed, preprocessing, software and
hardware settings. The 60 images alone do not reproduce a table.

## Training

Place training artworks in a separate folder; subfolders are supported. Every
image provides its own reconstruction target, edge/grayscale conditions and
shuffled reference. No paired stylized targets or captions are required.
**Do not train on the bundled comparison inputs.**

```bash
python -m rqstyle.training \
  --train-dir /path/to/training/artworks \
  --output-dir outputs/train \
  --iterations 15000 \
  --save-every 1000 \
  --device cuda:0
```

Training starts from randomly initialized RQStyle modules and frozen pretrained
backbones. It uses batch size 1, AdamW, learning rate 1e-4, weight decay 1e-4,
gradient clipping at 1.0, and the original five-stratum 50-position DDIM training
schedule. Generic artwork prompts are sampled during training.

The entry point automatically prepares reusable target and edge VAE latents in
`outputs/train/cache/`. An optional `--cache-dir /path/to/cache` shares these
between runs; image hashes and preprocessing settings are checked. Patch
permutations are sampled afresh each iteration, not frozen in the cache.

Resume a training checkpoint, including optimizer and RNG states:

```bash
python -m rqstyle.training \
  --train-dir /path/to/training/artworks \
  --output-dir outputs/train \
  --resume-checkpoint outputs/train/checkpoints/rqstyle_iter01000.pt \
  --iterations 15000
```

`--iterations` is the final iteration number, not an additional step count.
Resume requires the same training images and configuration. Alternatively,
`--init-checkpoint checkpoints/rqstyle.pt` loads weights but starts a new
optimizer and iteration count. The bundled inference checkpoint has no optimizer
state, so use initialization, not resume, for that file.

The original model was trained on 160 WikiArt artworks from 20 categories; that
training set is not bundled. Training on a different image folder uses the same
method but does not reproduce the published checkpoint. The new cache builder
uses deterministic per-image seeds, not the original research cache's class
indexing.

Training retains FP32 U-Net computation and activation gradients through the
frozen backbone. It therefore requires substantially more VRAM than inference.
This release is single-GPU; it does not implement DDP or model sharding. A 4.14M
trainable parameter count is not the total model size or a training VRAM estimate.

## Comparison Inputs

```text
data/
  content/    # 20 COCO images: 01.jpg ... 20.jpg
  style/      # 40 WikiArt references: 01.jpg ... 40.jpg
```

Use the complete Cartesian product: **20 contents x 40 styles = 800 pairs**.
Only the 60 source images are stored, without duplicated inputs, generated
outputs, metrics, or selection manifests. Images retain their original bytes;
only their filenames are simplified.

This is the final **model-aware, post-selected diagnostic subset** used in the
COCO-to-WikiArt comparisons (TASK241/TASK242). It is **not** StyleID's official
800-pair set, a random sample, or an unbiased benchmark. Its selection used
model outputs; results on it should not be presented as evidence of general
superiority. It is separate from the fresh random qualitative example pools.

## Offline Use

Both entry points accept `--base-model`, `--vae`, `--image-encoder`,
`--ip-adapter-root`, and `--ip-adapter-weight`. The same model locations can be
set through `RQSTYLE_BASE_MODEL`, `RQSTYLE_VAE`, `RQSTYLE_IMAGE_ENCODER`, and
`RQSTYLE_IP_ADAPTER`. Local IP-Adapter roots should contain the `sdxl_models/`
subfolder. Use the same model revisions when reproducing an existing result.

## Acknowledgments and Usage Terms

Built on [Diffusers](https://github.com/huggingface/diffusers),
[SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), and
[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter). Comparison inputs
originate from [COCO](https://cocodataset.org/) and
[WikiArt](https://www.wikiart.org/).

Third-party code, weights, and images remain subject to their respective terms
and rights. Bundling research inputs does not assign a new license to those
images. No new project-wide code license is assigned by this snapshot; the
authors should choose one before offering a licensed public release.
