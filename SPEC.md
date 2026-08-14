# ONLY CASTOR — Spec

> **Note on image paths (2026-08-14).** This spec was written while ONLY was a
> standalone checkout at `~/ONLY`, and it describes images living inside the
> repo at `CASTOR/shipwreck_wiki_images/`. That is no longer the layout. ONLY is
> now a submodule of BenchyBench, and the image set is a single shared copy at
> the BenchyBench root:
>
> ```
> ~/BenchyBench/shipwreck_wiki_images/sorted_images/
> ```
>
> Paths are resolved at runtime by `CASTOR/benchybench_paths.sh`; nothing needs
> to be symlinked or copied into this repo. See `CASTOR/README.md` for current
> usage. The design intent below is otherwise unchanged.

Port the CASTOR shipwreck inference pipeline from DeGF to ONLY, replacing
DeGF's diffusion-based contrastive decoding with ONLY's single-layer
intervention. Remove all benchmark eval infrastructure that is irrelevant
to the shipwreck task.

---

## 1. Objective

Run ONLY-enhanced LLaVA-1.5-7B inference over the
`CASTOR/shipwreck_wiki_images/sorted_images` dataset on the pleiades SLURM
cluster (AART Lab, head1.condo.cs.cmu.edu) via Apptainer.

Target user: sole researcher submitting SLURM array jobs; results written to
`/data/$USER/castor_results/`.

---

## 2. Commands

```bash
# One-time: prepare the questions file for a given prompt
python CASTOR/prepare_dataset.py \
    --image-dir  CASTOR/shipwreck_wiki_images/sorted_images \
    --output     CASTOR/shipwreck_wiki_images/questions.jsonl \
    --prompt-file CASTOR/prompts/prompt.txt

# Submit ONLY inference (default — use_only=True)
bash CASTOR/submit.sh

# Submit baseline (ONLY disabled)
bash CASTOR/submit.sh --no-only

# Submit with run tag
bash CASTOR/submit.sh --run-name exp1

# Monitor
squeue -u $USER
tail -f /data/$USER/logs/castor_<ARRAYJOBID>_<TASKID>.out

# Interactive debug on a compute node
srun -p pleiades --time=1:00:00 --cpus-per-task=4 --gpus=1 \
     --mem=40G --constraint=RTX6000ADA --pty bash
# inside the node:
cd ~/BenchyBench/ONLY
apptainer exec --containall --nv \
    --bind /data/$USER:/data/$USER --bind ~/BenchyBench/ONLY:~/BenchyBench/ONLY --bind /tmp:/tmp \
    /data/$USER/castor_only.sif \
    /opt/conda/bin/python3 CASTOR/run_inference.py --use-only
```

---

## 3. Files Changed / Added

### Add (new)

| File | Description |
|------|-------------|
| `CASTOR/run_inference.py` | Main inference script — adapted from DeGF's, uses `evolve_only_sampling()` instead of `evolve_degf_sampling()`; no SD pipeline; ONLY-specific CLI flags |
| `CASTOR/prepare_dataset.py` | Copied verbatim from `../DeGF/CASTOR/prepare_dataset.py` |
| `CASTOR/config.json` | ONLY-specific config (ONLY hyperparams replace DeGF's; no `use_diffusion`/SD fields) |
| `CASTOR/submit.sh` | Copied verbatim from `../DeGF/CASTOR/submit.sh`; `--use-diffusion`/`--no-diffusion` replaced with `--use-only`/`--no-only` |
| `CASTOR/submit_job.sh` | Adapted from `../DeGF/CASTOR/submit_job.sh`; references `castor_only.sif`, ONLY flags |
| `CASTOR/container.def` | Adapted from `../DeGF/CASTOR/container.def`; adds `pip install -e /ONLY/transformers`; no diffusers/SD deps |
| `CASTOR/prompts/` | Directory with `.txt` prompt files — copy from `../DeGF/CASTOR/` prompts if they exist, or create fresh |
| `CASTOR/shipwreck_wiki_images/` | Symlink or copy from `../DeGF/CASTOR/shipwreck_wiki_images/` |
| `experiments/llava/` | Copied verbatim from `../DeGF/experiments/llava/` — provides the `llava` package that `run_inference.py` imports via `sys.path` |

### Remove (delete from ONLY repo)

| Path | Reason |
|------|--------|
| `eval_bench/` | POPE and CHAIR eval scripts — not needed for shipwreck inference |
| `experiments/eval/` | MME evaluation scripts |
| `experiments/Qwen_VL/` | Qwen-VL model files — only LLaVA is used here |
| `experiments/lavis/` | InstructBLIP / LAVIS framework — not used |
| `experiments/cd_scripts/` | MME launch scripts |
| `chair.pkl` | CHAIR eval data |
| `figs/` | Paper figures |

### Keep untouched

| Path | Reason |
|------|--------|
| `only_utils/only_sample.py` | Core ONLY decoding — not modified |
| `only_utils/vcd_add_noise.py` | Used by baseline comparison flags |
| `utils/dist_util.py`, `utils/logger.py` | Utilities |
| `transformers/` | Patched HF fork — installed via `pip install -e transformers` inside the container |
| `requirements.txt` | Reference; deps are installed in container via `container.def`, not via this file directly |

---

## 4. Key Adaptation: `CASTOR/run_inference.py`

Start from `../DeGF/CASTOR/run_inference.py` and apply these diffs:

| DeGF | ONLY |
|------|------|
| `from degf_utils.degf_sample import evolve_degf_sampling` | `from only_utils.only_sample import evolve_only_sampling` |
| `evolve_degf_sampling()` | `evolve_only_sampling()` |
| `use_diffusion` flag + SD pipeline | `use_only` flag; no SD pipeline |
| `degf_alpha_pos`, `degf_alpha_neg`, `degf_beta` params | `ritual_alpha_pos`, `ritual_alpha_neg`, `ritual_beta`, `js_gamma`, `enhance_layer_index` |
| `model.generate(..., use_diffusion=..., degf_alpha_pos=..., ...)` | `model.generate(..., use_only=..., enhance_layer_index=..., js_gamma=..., ritual_alpha_pos=..., ritual_alpha_neg=..., ritual_beta=...)` |
| `images_neg` set when `use_diffusion` | `images_neg=None`, `images_pos=None` always (ONLY doesn't need alternate images) |

Preserve from DeGF's `run_inference.py` verbatim:
- All torch compatibility patches at the top (xpu/mps stubs, float8 types, compiler mock, distributed device_mesh stubs, transformers class stubs)
- CUDA device detection and fatal-exit-if-no-GPU guard
- `config.json` loading + CLI `--merge` pattern
- Resumable output (line-count resume)
- OOM recovery retry (`_run_generate_safe`)
- Per-image GPU memory cleanup in `finally` block
- Timing telemetry and throughput summary

---

## 5. `CASTOR/config.json`

```json
{
    "paths": {
        "model_path": "/data/$USER/llava-v1.5-7b",
        "model_base": null,
        "image_folder": "CASTOR/shipwreck_wiki_images/sorted_images",
        "question_file": "CASTOR/shipwreck_wiki_images/questions.jsonl",
        "answers_file": "/data/$USER/castor_results/answers.jsonl"
    },
    "hyperparameters": {
        "conv_mode": "llava_v1",
        "use_only": true,
        "enhance_layer_index": 0,
        "js_gamma": 0.2,
        "ritual_alpha_pos": 3.0,
        "ritual_alpha_neg": 1.0,
        "ritual_beta": 0.1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": null,
        "seed": 42,
        "max_new_tokens": 1024
    }
}
```

---

## 6. `CASTOR/container.def`

Based on `../DeGF/CASTOR/container.def` with two changes:

1. **Remove** `diffusers==0.21.4` and SD-related packages (no image generation needed).
2. **Add** after the main `pip install` block:
   ```
   pip install -e /ONLY/transformers
   ```
   The SLURM submit job must bind-mount the repo at `/ONLY` so the editable install works inside the container.

The `%environment` block (env vars, CUDA alloc conf, thread hygiene, LD_LIBRARY_PATH) is copied verbatim.

---

## 7. `CASTOR/submit_job.sh`

Same structure as DeGF's, with:
- `SIF="$DATA_DIR/castor_only.sif"` (separate SIF from DeGF's `castor.sif`)
- `DEF` hash references `CASTOR/container.def` in ONLY repo
- `APPTAINER_BASE` bind-mount adds `--bind $REPO:/ONLY` so `pip install -e /ONLY/transformers` resolves
- Mode flags changed: `--use-only` / `--no-only` instead of `--use-diffusion` / `--no-diffusion`
- SBATCH label: `-J castor_only`
- Baseline mode tag: `only` vs `baseline`

---

## 8. Code Style

- Match DeGF's `run_inference.py` style exactly for the adapted sections.
- No type annotations added beyond what's already present.
- No new abstraction layers or helper modules.
- Each change is surgical — only lines that differ from DeGF's version are touched.

---

## 9. Testing Strategy

1. **Submit test** — `bash CASTOR/submit.sh` completes without permission errors; `squeue` shows job queued.
2. **Log file** — `ls /data/$USER/logs/` shows `castor_<ARRAYJOBID>_<TASKID>.out/err`.
3. **Container build** — first run builds `castor_only.sif`; second run skips (hash match).
4. **Inference smoke test** — interactive SLURM session; run `python CASTOR/run_inference.py --use-only` on 1-image subset of `questions.jsonl`; confirm output in `answers_only_<tag>.jsonl`.
5. **Baseline comparison** — run `python CASTOR/run_inference.py --no-only` on same 1-image subset; verify different output (confirms ONLY is actually doing something).
6. **GPU utilization** — `nvidia-smi` inside job shows memory used.

No automated test suite is added.

---

## 10. Boundaries

| Category | Rule |
|----------|------|
| **Always do** | GPU required — if `torch.cuda.is_available()` is `False`, log a clear error and exit. |
| **Always do** | Preserve all ONLY decoding flags: `use_only`, `enhance_layer_index`, `js_gamma`, `ritual_alpha_pos`, `ritual_alpha_neg`, `ritual_beta`. |
| **Always do** | Preserve baseline comparison flags: `use_ritual`, `use_vcd`, `use_m3id` (pass through to `model.generate` unchanged). |
| **Always do** | Keep `pip install -e transformers` as the install mechanism for the patched fork. |
| **Ask first** | Any change to `only_utils/only_sample.py` logic. |
| **Ask first** | Any change to `config.json` default hyperparameter values. |
| **Ask first** | Adding or removing `container.def` package pins. |
| **Never** | Modify `shipwreck_wiki_images/` data or `questions.jsonl`. |
| **Never** | Upgrade `transformers`, `tokenizers`, `peft`, `torch`, `bitsandbytes` versions. |
| **Never** | Add new Python dependencies without checking against existing pin constraints. |
| **Never** | Touch `only_utils/only_sample.py` — it is the core method and must remain intact. |
