# CASTOR — ONLY

Maritime disaster classification with ONLY-enhanced LLaVA-1.5-7B on the AART Lab
`pleiades` cluster. Classifies shipwreck images as
`aground / capsized / on_fire / sunken` and extracts vessel type, size and cargo.

ONLY (ICCV 2025) mitigates hallucination with a **single transformer-layer
intervention**. Unlike DeGF it needs no Stable Diffusion reference image, so a
run costs roughly one forward pass per token instead of two plus image
generation — substantially cheaper, and the reason it is worth comparing against
DeGF on identical inputs.

## Quick Start

```bash
# 1. Log into the cluster
ssh <username>@head1.condo.cs.cmu.edu

# 2. Clone BenchyBench, which carries this repo as a submodule.
#    Images are NOT in this repo — they live once at the BenchyBench root
#    and are shared by every method.
cd ~
git clone --recurse-submodules https://github.com/M4-on-Github/BenchyBench.git

# 3. Build the container (one time, ~15 min)
cd ~/BenchyBench/ONLY
sbatch CASTOR/build_container.sh

# 4. Baseline run (intervention disabled)
bash CASTOR/submit.sh --no-only

# 5. ONLY run
bash CASTOR/submit.sh --use-only
```

Always submit through `CASTOR/submit.sh`, never `sbatch CASTOR/submit_job.sh`
directly — the wrapper creates `/data/$USER/logs/` before SLURM opens the log
file, and counts prompts to size the array.

## Commands

```bash
bash CASTOR/submit.sh                  # every prompt × both modes
bash CASTOR/submit.sh --use-only       # ONLY mode only
bash CASTOR/submit.sh --no-only        # baseline only
bash CASTOR/submit.sh --run-name exp1  # tag output and log files
```

With no mode flag, one array task is submitted per (prompt × mode) pair, so
baseline and ONLY run in parallel on separate GPUs. Task IDs interleave: even →
baseline, odd → ONLY, with each pair sharing a prompt.

Monitor:

```bash
squeue -u $USER
tail -f /data/$USER/logs/castor_*<ARRAYJOBID>*.out
```

## Prompts

`CASTOR/prompts/` holds one file per experiment, discovered by glob and indexed
by SLURM array task ID:

| Prompt | Approach |
|---|---|
| `promptv1_1shot_noCoT.txt` | One worked example, no reasoning steps |
| `promptv2_CoT+1shot.txt` | Chain-of-thought plus one example |
| `promptv4.1.txt` | Staged: evidence catalog → grounding questions → JSON |
| `promptv5.txt` | Single dense descriptive paragraph |

Adding a `.txt` here grows the array automatically. **Treat the directory as
frozen once a run starts** — changing it invalidates comparisons against earlier
runs. The shared library these are drawn from is
`BenchyBench/all_maritime_prompts/`.

## Storage Layout

| What | Where | Why |
|---|---|---|
| Code | `~/BenchyBench/ONLY/` (home, 500 GB) | Fast access, backed up |
| Images | `~/BenchyBench/shipwreck_wiki_images/` | One shared copy at the root, not per-repo |
| LLaVA-1.5-7B weights | `/data/$USER/llava-v1.5-7b/` | Too large for the home quota |
| Apptainer container | `/data/$USER/castor_ONLY.sif` | ~6 GB |
| Results | `/data/$USER/castor_results/` | Grows across experiments |
| Logs | `/data/$USER/logs/` | Per array task |

Weights are downloaded automatically on first run if the directory is missing.

## Image Path Resolution

Images are located by `CASTOR/benchybench_paths.sh`, which resolves the
BenchyBench root by probing each candidate and **erroring rather than guessing**:

1. `$BENCHYBENCH_ROOT` — explicit; invalid is a hard error, not a fallback
2. `$SLURM_SUBMIT_DIR` and its parent — validated, never blindly trusted
3. The script's own location — parent (nested), then repo (standalone)

`$SLURM_SUBMIT_DIR` is consulted before the script location because SLURM copies
a batch script to a spool directory, so `$0` inside a running job may point at
`/var/spool/...` rather than this repo.

Every job log records the resolved root via `bb_report`, so a run against the
wrong tree leaves evidence. To run outside BenchyBench, set `BENCHYBENCH_ROOT`
or pass `--image-folder`.

This file is byte-identical across DeGF, ONLY and QWEN-Maritime, enforced by
`BenchyBench/tests/test_paths.sh`. Do not edit one copy in isolation.

## Output Format

One JSONL record per image, at
`/data/$USER/castor_results/answers_{mode}[_{run_name}]_{stem}_j{ArrayJobID}.jsonl`:

```json
{
  "question_id": 17,
  "image": "aground/00017.jpg",
  "prompt": "...the full prompt text...",
  "text": "...model output...",
  "model_id": "llava-v1.5-7b",
  "use_only": true,
  "timing": {"infer_s": 4.21, "total_s": 4.98}
}
```

`use_only` records which mode produced the record, so baseline and ONLY output
remain distinguishable after files are merged.

Failures are written rather than dropped, keeping one line per image:

```json
{"question_id": 17, "image": "aground/00017.jpg", "error": "oom-skip"}
```

An image that OOMs even after retry is skipped with `error: "oom-skip"`; other
exceptions record the message. Downstream evaluation must tolerate records with
`error` and no `text`.

`image` is the join key against
`Eval_CASTOR/human_ground_truth_label/human_gt.csv`. Runs resume: if the answers
file already holds N records, processing restarts at image N + 1.

## Overriding Config

`CASTOR/config.json` holds defaults; every field has a CLI flag that overrides it.

```bash
# Tune the intervention
sbatch CASTOR/submit.sh --use-only --enhance-layer-index 2 --ritual-beta 0.2

# Write to a custom path
sbatch CASTOR/submit.sh --answers-file /data/$USER/castor_results/exp1.jsonl

# Point at a different image set
sbatch CASTOR/submit.sh --image-folder /data/$USER/my_subset
```

| Hyperparameter | Default | Effect |
|---|---|---|
| `enhance_layer_index` | 0 | Which transformer layer carries the intervention |
| `ritual_alpha_pos` | 3.0 | Positive amplification weight |
| `ritual_alpha_neg` | 1.0 | Negative suppression weight |
| `ritual_beta` | 0.1 | Blending strength against base logits |
| `js_gamma` | 0.2 | JS-divergence gate threshold |
| `max_new_tokens` | 1024 | Truncates long outputs |

Run `python CASTOR/run_inference.py --help` for the full list.

## Architecture

```
submit.sh              wrapper: makes log dir, counts prompts, sizes the array
  └─ submit_job.sh     SLURM body: resolves paths, builds the apptainer command
       ├─ prepare_dataset.py   images + prompt → questions.jsonl
       └─ run_inference.py     questions.jsonl → answers.jsonl
```

`run_inference.py` is passed `--image-folder` explicitly, so `config.json`'s
relative `"../shipwreck_wiki_images/sorted_images"` is never consulted — that
path resolved against the current working directory and therefore depended on
where the job was launched from.

The vendored LLaVA source is at `experiments/llava/`; there is no
`pip install llava`. `run_inference.py` inserts `experiments/` into `sys.path`
at startup.

## Gotchas

- **Submit through `submit.sh`.** Calling `sbatch CASTOR/submit_job.sh` directly
  fails when `/data/$USER/logs/` does not exist yet.
- **`--containall` is required.** Without it the cluster's `apptainer.conf`
  bind-mounts the host `/opt` over the container's `/opt/conda`.
- **Package pins are hard.** `transformers==4.31.0`, `torch==2.0.1`,
  `peft==0.4.0`. They are interdependent and shared with DeGF; upgrading one
  breaks the vendored LLaVA.
- **Runs resume rather than overwrite.** Delete or rename the answers file to
  force a clean re-run.
- **Prompts are frozen mid-experiment.** Editing `prompts/` between runs makes
  results incomparable.

## Evaluation

Inference output feeds the Eval_CASTOR pipelines. P1 is the fast no-backend
sanity check.

`eval_castor.py` takes no input path — it globs `*.jsonl` from a fixed handoff
directory at the BenchyBench root, so copy results there first:

```bash
mkdir -p ~/BenchyBench/results/castor_results
cp /data/$USER/castor_results/answers_*.jsonl ~/BenchyBench/results/castor_results/

cd ~/BenchyBench/Eval_CASTOR
python pipelines/eval_castor.py              # P1, regex only
python pipelines/eval_castor.py --pre-parsed # P2, Gemma-extracted fields
```

`results/` is gitignored — it is the handoff point between inference and
evaluation, not tracked output.

See `BenchyBench/PIPELINES.md` for the full pipeline directory.

## Related

- `../README.md` — upstream ONLY paper README (method, citation)
- `../SPEC.md` — CASTOR port design intent
- `BenchyBench/PIPELINES.md` — every pipeline across all four repos
