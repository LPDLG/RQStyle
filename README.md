# RQStyle

Official code for **RQStyle: ResNet and Attention-Query Modulation for Structure-Preserving Style Transfer**. RQStyle is a diffusion-based style transfer system that preserves content structure while transferring a reference style. The codebase includes SDXL inference pipelines, ResNet/attention-query modulation experiments, adapter training scripts, baseline runners, and the evaluation protocol used for the paper experiments.

> Paper, project page, and released checkpoints will be added after publication.

## Highlights

- Structure-preserving stylization with Stable Diffusion XL.
- ResNet feature and attention-query modulation for balancing structure and style.
- Support for WikiArt and Chinese paper-cut style-transfer protocols.
- Baseline runners for StyleSSP-style generation and StyleID.
- Adapter and ablation code for UNet attention/residual feature injection.
- Official evaluation wrappers for LPIPS, FID, and ArtFID.

## Installation

The main experiments were run with Python 3.9, PyTorch 2.3, Diffusers 0.30, and CUDA GPUs.

```bash
git clone <repo-url>
cd StyleSSP-main

conda env create -f environment.yaml
conda activate StyleSSP
```

Alternatively, install the Python dependencies in an existing environment:

```bash
pip install -r requirements.txt
```

For reproducible offline runs, point Hugging Face caches to your local model directory:

```bash
export HF_HOME=/path/to/huggingface
export HUGGINGFACE_HUB_CACHE=/path/to/huggingface/hub
export TRANSFORMERS_CACHE=/path/to/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

If CUDA memory fragmentation occurs, this setting is often helpful:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## Models and Data

Large pretrained models, generated images, and checkpoints are not included in the repository. Depending on the experiment, prepare:

- Stable Diffusion XL base 1.0.
- IP-Adapter / IP-Adapter-Instruct weights for SDXL.
- ControlNet weights for SDXL baselines.
- DINOv2 weights for self-similarity analysis.
- Evaluation manifests under `results/task027_v1b_eval_protocol*`.
- Optional RQStyle/StyleSSP adapter checkpoints under `checkpoints/`.

Many scripts contain local default paths from the original experiment machine. Use command-line arguments or environment variables to override them on a new system.

## Repository Layout

```text
StyleSSP-main/
  infer_style.py                         # single-image inference entry
  pipeline_*.py                          # modified SDXL/style-transfer pipelines
  train_task027_d5_zero_control.py       # adapter training entry
  tools/                                 # protocol, baseline, metric, and analysis tools
  evaluation/                            # FID / LPIPS / ArtFID utilities
  configs/                               # experiment configs
  checkpoints/                           # local checkpoints, not tracked
  results/                               # generated images and metrics, not tracked
  run_*.sh                               # reproducibility scripts
```

## Quick Start

### StyleSSP-Style Baseline

WikiArt protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/task027_v1b_stylessp_ori_runner.py \
  --protocol-dir results/task027_v1b_eval_protocol \
  --output-dir results/stylessp_ori_v1b/wikiart_apply20_sdxl_s50_r512 \
  --domains anime photo \
  --resolution 512 \
  --run-metrics
```

Chinese paper-cut protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/task027_v1b_stylessp_ori_runner.py \
  --protocol-dir results/task027_v1b_eval_protocol_papercut_styleid \
  --output-dir results/stylessp_ori_v1b/papercut_apply20_sdxl_s50_r512 \
  --domains anime photo \
  --resolution 512 \
  --run-metrics
```

The convenience wrapper runs both protocols:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_stylessp_ori_v1b_wikiart_papercut_r512.sh
```

### StyleID Baseline

The persistent runner keeps the diffusion pipeline loaded and reuses style inversion features across content images:

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/task027_v1b_styleid_persistent_runner.py \
  --protocol-dir results/task027_v1b_eval_protocol \
  --output-dir results/styleid_v1b/wikiart_apply20_sd15_s50 \
  --domains anime photo \
  --resolution 512 \
  --run-metrics
```

For paper-cut style references:

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/task027_v1b_styleid_persistent_runner.py \
  --protocol-dir results/task027_v1b_eval_protocol_papercut_styleid \
  --style-manifest results/task027_v1b_eval_protocol_papercut_styleid/eval_style_apply20.txt \
  --fid-ref-manifest results/task027_v1b_eval_protocol_papercut_styleid/eval_style_papercut_fidref2048.txt \
  --output-dir results/styleid_v1b/papercut_apply20_sd15_s50 \
  --domains anime photo \
  --resolution 512 \
  --run-metrics
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/task027_d27_official_artfid.py \
  --items-json results/stylessp_ori_v1b/wikiart_apply20_sdxl_s50_r512/summary/stylessp_ori_v1b_items_all.json \
  --output-dir results/metrics/stylessp_ori_wikiart \
  --method stylessp_ori_wikiart \
  --domains anime photo \
  --device cuda \
  --batch-size 8 \
  --num-workers 1
```

Each run writes both JSON and Markdown metric summaries to the selected output directory.

## Reproducing Paper Experiments

The paper experiments use frozen evaluation manifests under:

- `results/task027_v1b_eval_protocol`
- `results/task027_v1b_eval_protocol_papercut_styleid`

Common full-grid entry points:

```bash
# StyleSSP-style baseline on WikiArt and paper-cut protocols
CUDA_VISIBLE_DEVICES=0 bash run_stylessp_ori_v1b_wikiart_papercut_r512.sh

# A1 reference / adapter ablation with official ArtFID
CUDA_VISIBLE_DEVICES=0 bash run_task027_d30_partb_a1ref_apply40_official.sh
```

Before launching a large run, inspect the script and update:

- GPU id: `CUDA_VISIBLE_DEVICES`
- model cache paths
- protocol directories
- output directories
- checkpoint paths

Most full-grid runners support resume behavior by checking for existing output files.

