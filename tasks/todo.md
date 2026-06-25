# CASTOR ONLY — TODO

## Completed

- [x] Copy `experiments/llava/` from DeGF
- [x] Create `CASTOR/prepare_dataset.py` (verbatim copy from DeGF)
- [x] Create `CASTOR/prompts/shipwreck.txt`
- [x] Copy `CASTOR/shipwreck_wiki_images/` from DeGF
- [x] Create `CASTOR/config.json` (ONLY hyperparams)
- [x] Create `CASTOR/run_inference.py` (adapted from DeGF, ONLY decoding)
- [x] Create `CASTOR/container.def` (no diffusers, pip install -e /ONLY/transformers)
- [x] Create `CASTOR/submit.sh` (--use-only / --no-only flags)
- [x] Create `CASTOR/submit_job.sh` (castor_only.sif, /ONLY bind, no SD download)
- [x] Delete eval_bench/, experiments/eval/, experiments/Qwen_VL/, experiments/lavis/, experiments/cd_scripts/, chair.pkl, figs/
- [x] Update CLAUDE.md

## Pending (cluster-side, after push)

- [ ] Push branch ONLY_CASTOR to GitHub
- [ ] On cluster: `git clone` or `git pull` ONLY_CASTOR branch
- [ ] First run: `bash CASTOR/submit.sh --use-only` — verify container builds, model downloads, inference runs
- [ ] Check `/data/$USER/castor_results/answers_only_*.jsonl` for output
- [ ] Run baseline for comparison: `bash CASTOR/submit.sh --no-only`
