# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ONLY (One-Layer Intervention Sufficiently Mitigates Hallucinations) is an ICCV 2025 research codebase that addresses hallucinations in Large Vision-Language Models (LVLMs). It implements a contrastive decoding method that intervenes at a single transformer layer during inference — no fine-tuning required.

This repo is configured for the **CASTOR** shipwreck image inference task, running LLaVA-1.5-7B on maritime disaster images on the pleiades SLURM cluster (AART Lab).

## Environment Setup

```bash
conda create -n ONLY python=3.10
conda activate ONLY
pip install -r requirements.txt
python -m pip install -e transformers   # installs the local patched transformers fork
```

Requires PyTorch 2.0.1 + matching CUDA. LLaVA-1.5-7b checkpoint must be downloaded separately (auto-downloaded by the cluster job if missing).

## CASTOR: Shipwreck Inference

All commands run from the repo root (`~/ONLY/`).

### Prepare dataset (one-time per prompt)

```bash
python CASTOR/prepare_dataset.py \
    --image-dir  CASTOR/shipwreck_wiki_images/sorted_images \
    --output     CASTOR/shipwreck_wiki_images/questions.jsonl \
    --prompt-file CASTOR/prompts/shipwreck.txt
```

### Submit cluster jobs

```bash
# Both modes (ONLY + baseline) — one array task per mode per prompt file
bash CASTOR/submit.sh

# ONLY mode only
bash CASTOR/submit.sh --use-only

# Baseline only (no layer intervention)
bash CASTOR/submit.sh --no-only

# With a run tag (appended to output filename)
bash CASTOR/submit.sh --use-only --run-name layer2_g025

# Monitor
squeue -u $USER
tail -f /data/$USER/logs/castor_<ARRAYJOBID>_<TASKID>.out
```

### Run inference directly (interactive node or local)

```bash
python CASTOR/run_inference.py --use-only
python CASTOR/run_inference.py --no-only
python CASTOR/run_inference.py --use-only --enhance-layer-index 2 --js-gamma 0.25
python CASTOR/run_inference.py --help    # full flag list
```

Results are written to `/data/$USER/castor_results/answers_{mode}[_{run_name}].jsonl`.
Runs are resumable: re-running skips already-written lines.

## Architecture

### Core Method

The ONLY intervention lives in two places:

1. **`only_utils/only_sample.py`** — Monkey-patches `transformers.generation.utils.GenerationMixin.sample` at import time via `evolve_only_sampling()`. The patched `sample()` implements contrastive decoding: on each generation step it compares full-model logits against "contrast" logits from a single suppressed layer, then applies an adaptive plausibility constraint (`ritual_beta`) before sampling.

2. **The patched `transformers/` fork** (`transformers/src/`) — A copy of HuggingFace `transformers==4.31.0` with model-specific modifications. The LLaVA `generate()` method is extended to accept `use_only`, `enhance_layer_index`, `js_gamma`, and related kwargs, and returns a second value `logits_cd` (contrast logits from the intervened layer).

### Decoding Logic (in `only_utils/only_sample.py`)

When `use_only=True`, for each token step:
- The model's normal forward pass returns `(outputs, logits_cd)` where `logits_cd` are logits computed with one layer's output altered at `enhance_layer_index`.
- Total Variation Distance (TVD) between the two distributions is computed.
- If `TVD < js_gamma`: boost the contrast logits positively (`diffs = logits + alpha_pos * logits_cd`).
- If `TVD >= js_gamma`: subtract the contrast logits (`diffs = (1+alpha_neg)*logits - alpha_neg*logits_cd`).
- An adaptive plausibility mask (`ritual_beta`) filters out tokens too far below the max logit.

### Key Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `enhance_layer_index` | 0 | Which transformer layer to suppress for the contrast forward pass |
| `js_gamma` | 0.2 | TVD threshold switching between additive/subtractive branch |
| `ritual_alpha_pos` | 3.0 | Scaling factor for the additive (low-divergence) branch |
| `ritual_alpha_neg` | 1.0 | Scaling factor for the subtractive (high-divergence) branch |
| `ritual_beta` | 0.1 | Adaptive plausibility cutoff |

All parameters are set in `CASTOR/config.json` and overridable via CLI.

### Directory Layout

```
only_utils/                   # Core ONLY implementation
  only_sample.py              #   monkey-patches GenerationMixin.sample
  vcd_add_noise.py            #   diffusion noise utility (for baseline flags)
CASTOR/                       # Shipwreck inference pipeline
  run_inference.py            #   main inference script
  prepare_dataset.py          #   builds questions.jsonl from image directory
  config.json                 #   default paths + hyperparameters
  submit.sh                   #   cluster wrapper (creates log dir, calls sbatch)
  submit_job.sh               #   SLURM batch script (builds SIF, runs inference)
  container.def               #   Apptainer container definition
  prompts/                    #   .txt prompt files (one task per file in array job)
  shipwreck_wiki_images/
    sorted_images/            #   images by category (aground, capsized, on_fire, sunken)
    questions.jsonl           #   generated by prepare_dataset.py
experiments/
  llava/                      # LLaVA-1.5 model code (imported by run_inference.py)
transformers/                 # Patched HuggingFace transformers fork
utils/                        # dist_util.py, logger.py
```

### Cluster Infrastructure

- **Container**: `CASTOR/container.def` builds `castor_only.sif` via Apptainer (auto-built on first job, cached by sha256 hash of container.def).
- **Storage**: All user-writable paths under `/data/$USER/` (model cache, results, logs, SIF).
- **Array jobs**: `submit.sh` submits N×2 tasks (N prompts × 2 modes) by default; interleaved even=baseline / odd=ONLY.
- **Resumable**: `run_inference.py` counts existing lines in the output file and skips already-processed images.
