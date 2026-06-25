# ONLY CASTOR — Implementation Plan

See full plan at: `.claude/plans/dapper-cuddling-nebula.md`

## Summary

Port the CASTOR shipwreck inference pipeline from `../DeGF/CASTOR/` into this
repo, replacing DeGF's diffusion-based decoding with ONLY's single-layer
intervention. Remove all COCO benchmark eval infrastructure.

## Slices Completed

| Slice | Description | Status |
|-------|-------------|--------|
| A | Copy `experiments/llava/` from DeGF | ✅ Done |
| B | CASTOR data layer (prepare_dataset.py, prompts/, images) | ✅ Done |
| C | `CASTOR/config.json` + `CASTOR/run_inference.py` | ✅ Done |
| D | `CASTOR/container.def`, `submit.sh`, `submit_job.sh` | ✅ Done |
| E | Delete eval_bench/, experiments/eval+Qwen_VL+lavis+cd_scripts, chair.pkl, figs/ | ✅ Done |
| F | Update CLAUDE.md | ✅ Done |

## Key Design Decisions

- `run_inference.py` uses `evolve_only_sampling()` — swapped from DeGF's `evolve_degf_sampling()`
- No SD/diffusion pipeline; `images_neg=None`, `images_pos=None` always
- ONLY params in config: `use_only`, `enhance_layer_index`, `js_gamma`, `ritual_alpha_pos/neg/beta`
- Baseline comparison flags (`use_ritual`, `use_vcd`, `use_m3id`) passed through as `False`
- Separate SIF: `castor_only.sif` (doesn't conflict with DeGF's `castor.sif`)
- Container bind-mounts repo at `/ONLY` so `pip install -e /ONLY/transformers` resolves
- Array job: even task ID = baseline, odd = ONLY (same interleaved pattern as DeGF)
