# [ICCV 2025] ONLY

[![Website](https://img.shields.io/badge/Project-Website-green)](https://zifuwan.github.io/ONLY/) [![arXiv](https://img.shields.io/badge/arXiv-2507.00898-red)](http://arxiv.org/abs/2507.00898) [![Conference](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv.thecvf.com/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 👀Introduction

This repository contains the code for our ICCV 2025 paper `ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models`. 

<div align="center">
  <img src="figs/efficiency.png" height="250"/>
  <img src="figs/overview.png"/>
</div>



## 💡Environment

We test our codebase with PyTorch 2.0.1. Please install the corresponding PyTorch and CUDA versions according to your computational resources.

```
conda create -n ONLY python=3.10
conda activate ONLY
git clone https://github.com/zifuwan/ONLY.git
cd ONLY
pip install -r requirements.txt
python -m pip install -e transformers
```

Please also download the model checkpoints:

- [**LLaVA-1.5**](https://github.com/haotian-liu/LLaVA): Download [LLaVA-1.5 merged 7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- [**InstructBLIP**](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip): Download [InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b)
- [**Qwen-VL-Chat**](https://huggingface.co/Qwen/Qwen-VL-Chat): Download [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat/tree/main)

As for the datasets and benchmarks:

- For **MSCOCO** dataset, see [this link](https://cocodataset.org/).
- For **MME**, see [this link](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).

## 📦Usage

We provide the code for evaluating our ONLY on POPE, CHAIR, and MME-Hallucination benchmark. You can simply run the following code to run the experiments:

- POPE: `bash eval_bench/scripts/pope_eval.sh`
- CHAIR:`bash eval_bench/scripts/chair_eval.sh`
- MME:`bash experiments/cd_scripts/mme_eval.sh`

## 🙏Acknowledgements

Our codebase is adapted from  [RITUAL](https://github.com/sangminwoo/RITUAL), [VCD](https://github.com/DAMO-NLP-SG/VCD), [OPERA](https://github.com/shikiw/OPERA), [LLaVA](https://github.com/haotian-liu/LLaVA), and [DeGF](https://github.com/zhangce01/DeGF/tree/main). We thank the authors for releasing their code!

## 📧Contact

If you have any questions, please  contact [zifuw@andrew.cmu.edu](mailto:zifuw@andrew.cmu.edu).

## 📌 BibTeX & Citation

If you find this code useful, please consider citing our work:

```bibtex
@article{wan2025only,
  title={ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models},
  author={Wan, Zifu and Zhang, Ce and Yong, Silong and Ma, Martin Q and Stepputtis, Simon and Morency, Louis-Philippe and Ramanan, Deva and Sycara, Katia and Xie, Yaqi},
  journal={arXiv preprint arXiv:2507.00898},
  year={2025}
}
```

## CASTOR Application (ONR Research)

This repository is also configured for the **CASTOR** maritime disaster classification task: classifying shipwreck images (`aground / capsized / on_fire / sunken`) using LLaVA-1.5-7B on the AART Lab `pleiades` SLURM cluster.

### Cluster Setup

```bash
ssh head1.condo.cs.cmu.edu
# Interactive GPU node (RTX6000Ada required)
srun -p pleiades --time=1:00:00 --cpus-per-task=4 --gpus=1 --mem=40G --constraint=RTX6000ADA --pty bash
```

### Running CASTOR Inference

From `~/ONLY/` on the cluster:

```bash
# Full sweep — ONLY method + baseline, all prompts
bash CASTOR/submit.sh

# ONLY method only
bash CASTOR/submit.sh --use-only

# Baseline only (no layer intervention)
bash CASTOR/submit.sh --no-only

# With a run tag (appended to output filenames)
bash CASTOR/submit.sh --use-only --run-name my_run

# Monitor
squeue -u $USER
tail -f /data/$USER/logs/castor_<ARRAYJOBID>_<TASKID>.out
```

Results land in `/data/$USER/castor_results/answers_{mode}[_{run_name}].jsonl`. Runs are resumable — resubmitting skips already-written lines.

### Pipeline Architecture

```
CASTOR/submit.sh                  (creates log dir, counts prompts, calls sbatch)
  └─ sbatch CASTOR/submit_job.sh  (SLURM array: one task per prompt × mode)
       ├─ builds/reuses castor_ONLY.sif (hashed against container.def)
       ├─ runs prepare_dataset.py  (builds per-prompt questions.jsonl)
       └─ apptainer exec → run_inference.py
            ├─ loads LLaVA-1.5-7B  (experiments/llava/ — vendored, not pip)
            └─ per image: forward pass → contrast layer → TVD-gated decoding
```

The ONLY method lives in `only_utils/only_sample.py`, which monkey-patches
`GenerationMixin.sample` at import time. On each token step it compares full-model
logits against a single suppressed-layer contrast forward pass, then applies an
adaptive plausibility mask (`ritual_beta`) before sampling.

### Key Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `enhance_layer_index` | 0 | Layer to suppress for the contrast pass |
| `js_gamma` | 0.2 | TVD threshold: additive (low) vs. subtractive (high) branch |
| `ritual_alpha_pos` | 3.0 | Scale factor for the additive branch |
| `ritual_alpha_neg` | 1.0 | Scale factor for the subtractive branch |
| `ritual_beta` | 0.1 | Adaptive plausibility cutoff |

All parameters live in `CASTOR/config.json` and are overridable via CLI flags.

### Hard Constraints

- **Never upgrade pinned packages** — `transformers==4.31.0`, `torch==2.0.1`, `peft==0.4.0`, `bitsandbytes==0.41.0`. These are interdependent.
- `experiments/llava/` is the vendored LLaVA source — `run_inference.py` inserts it into `sys.path`; there is no `pip install llava`.
- `transformers/` is a patched fork (installed via `pip install -e transformers`); do not replace with the upstream package.
- `$USER` in config paths is expanded at runtime — never hardcode a username.

### Cluster Storage

| What | Path |
|------|------|
| LLaVA-1.5-7B weights | `/data/$USER/llava-v1.5-7b/` |
| Apptainer container | `/data/$USER/castor_ONLY.sif` |
| HF cache | `/data/$USER/.cache/huggingface/` |
| Results | `/data/$USER/castor_results/` |
| Logs | `/data/$USER/logs/` |
