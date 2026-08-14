#!/usr/bin/env python3
"""
CASTOR inference for maritime disaster images using the ONLY method.

Loads LLaVA-1.5 once and processes all images in-process.
Resumable: re-running skips already-written results.

Usage (from repo root):
    python CASTOR/run_inference.py
    python CASTOR/run_inference.py --use-only
    python CASTOR/run_inference.py --no-only
    python CASTOR/run_inference.py --enhance-layer-index 2 --js-gamma 0.25
    python CASTOR/run_inference.py --config path/to/other.json --answers-file results/run2.jsonl

All CLI flags override their config.json counterpart.
Run `python CASTOR/run_inference.py --help` for the full list.
"""
import argparse
import gc
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# ── Torch compatibility patches ───────────────────────────────────────────────
# Must run before any torch/transformers/diffusers import.
import torch

try:
    torch.library.define("torchvision::nms",
        "(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    torch.library.define("torchvision::roi_align",
        "(Tensor input, Tensor rois, float spatial_scale, int pooled_height, "
        "int pooled_width, int sampling_ratio, bool aligned) -> Tensor")
except Exception:
    pass

for _attr in ("xpu", "mps"):
    if not hasattr(torch, _attr):
        setattr(torch, _attr, type(f"Mock{_attr.upper()}", (),
            {"__getattr__": lambda self, n: (lambda *a, **kw: None)})())

for _dtype in ("float8_e4m3fn", "float8_e5m2"):
    if not hasattr(torch, _dtype):
        setattr(torch, _dtype, type("MockDtype", (), {})())

if not hasattr(torch, "compiler"):
    torch.compiler = type("MockCompiler", (), {
        "disable": lambda self, *a, **kw: (lambda f: f) if not a else a[0]
    })()

try:
    from types import ModuleType
    import torch.distributed
    if not hasattr(torch.distributed, "device_mesh"):
        _dm = ModuleType("torch.distributed.device_mesh")
        sys.modules["torch.distributed.device_mesh"] = _dm
        torch.distributed.device_mesh = _dm
        _dm.DeviceMesh = type("DeviceMesh", (), {})
    try:
        import torch.distributed._functional_collectives as _fc
        if not hasattr(_fc, "AsyncCollectiveTensor"):
            _fc.AsyncCollectiveTensor = type("AsyncCollectiveTensor", (), {})
    except ImportError:
        _fc = ModuleType("torch.distributed._functional_collectives")
        sys.modules["torch.distributed._functional_collectives"] = _fc
        _fc.AsyncCollectiveTensor = type("AsyncCollectiveTensor", (), {})
except ImportError:
    pass

# Vendored transformers must be on sys.path before `import transformers` so that
# Python loads the custom LlamaModel (which accepts use_only/enhance_layer_index)
# rather than the stock one from the container's /opt/conda site-packages.
_REPO_EARLY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_EARLY, "transformers", "src"))

try:
    import transformers
    for _name in ("Cache", "DynamicCache", "EncoderDecoderCache",
                  "Dinov2WithRegistersConfig", "Dinov2WithRegistersModel",
                  "SiglipVisionConfig", "SiglipVisionModel", "SiglipImageProcessor",
                  "ViTMAEConfig", "ViTMAEModel"):
        if not hasattr(transformers, _name):
            setattr(transformers, _name, type(_name, (), {}))
except Exception:
    pass

# ── Device selection ─────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("ERROR: CUDA not available — this job requires a GPU. Exiting.", flush=True)
    sys.exit(1)
_DEVICE = "cuda"
_gpu = torch.cuda.get_device_properties(0)
print(f"Device     : {_gpu.name}  ({_gpu.total_memory // 1024**2} MiB)", flush=True)

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)                              # run_config.py
sys.path.insert(0, _REPO)                              # only_utils/

# Config handling lives in its own module so it can be tested without the
# vendored LLaVA stack, which pins transformers==4.31.0 and only imports
# inside the container.
from run_config import RunConfig
sys.path.insert(0, os.path.join(_REPO, "experiments")) # llava/

# ── Project imports ───────────────────────────────────────────────────────────
from PIL import Image
from tqdm import tqdm
from transformers import set_seed

from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                              DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from only_utils.only_sample import evolve_only_sampling
evolve_only_sampling()


# ─────────────────────────────────────────────────────────────────────────────

def _run_generate(model, input_ids, image_tensor, hp: dict, use_only: bool,
                  use_cache: bool = True):
    with torch.inference_mode():
        output_ids, _ = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(_DEVICE),
            images_pos=None,
            images_neg=None,
            do_sample=True,
            temperature=hp["temperature"],
            top_p=hp["top_p"],
            top_k=hp["top_k"],
            max_new_tokens=hp["max_new_tokens"],
            use_cache=use_cache,
            use_ritual=False,
            use_vcd=False,
            use_m3id=False,
            use_only=use_only,
            enhance_layer_index=hp["enhance_layer_index"],
            js_gamma=hp["js_gamma"],
            ritual_alpha_pos=hp["ritual_alpha_pos"],
            ritual_alpha_neg=hp["ritual_alpha_neg"],
            ritual_beta=hp["ritual_beta"],
        )
    return output_ids


def _run_generate_safe(model, input_ids, image_tensor, hp: dict, use_only: bool,
                       use_cache: bool = True):
    """Calls _run_generate with one OOM-recovery retry before giving up."""
    try:
        return _run_generate(model, input_ids, image_tensor, hp, use_only, use_cache)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        tqdm.write("[OOM] Clearing GPU cache and retrying once...")
        gc.collect()
        torch.cuda.empty_cache()
        return _run_generate(model, input_ids, image_tensor, hp, use_only, use_cache)


def run(cfg: dict):
    paths = cfg["paths"]
    hp    = cfg["hyperparameters"]

    set_seed(hp["seed"])
    disable_torch_init()

    use_only = hp["use_only"]
    job_start = time.perf_counter()
    print(f"Mode       : {'ONLY (layer intervention)' if use_only else 'Baseline (no intervention)'}")
    print(f"Model      : {paths['model_path']}")
    print(f"Questions  : {paths['question_file']}")
    print(f"Output     : {paths['answers_file']}")

    def _p(key):
        return os.path.expandvars(os.path.expanduser(paths[key]))

    t0 = time.perf_counter()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        _p("model_path"),
        paths.get("model_base"),
        get_model_name_from_path(_p("model_path")),
    )
    print(f"[timing] Model load     : {time.perf_counter() - t0:.1f}s")

    with open(_p("question_file"), encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    answers_file = _p("answers_file")
    os.makedirs(os.path.dirname(os.path.abspath(answers_file)), exist_ok=True)

    start_idx = 0
    if os.path.exists(answers_file):
        with open(answers_file, encoding="utf-8") as f:
            for line in f:
                try:
                    json.loads(line)
                    start_idx += 1
                except json.JSONDecodeError:
                    break  # truncated tail from a killed job — will be re-processed

    if start_idx >= len(questions):
        print(f"All {len(questions)} images already processed.")
        return

    print(f"Processing : {len(questions)} images (resuming from {start_idx})")
    conv_mode    = hp["conv_mode"]
    image_folder = _p("image_folder")
    model_name   = get_model_name_from_path(_p("model_path"))

    question_times = []
    _time_total = 0.0

    with open(answers_file, "a", encoding="utf-8") as out_f:
        for item in tqdm(questions[start_idx:], initial=start_idx, total=len(questions)):
            raw_image = image_tensor = None
            input_ids = output_ids = None
            q_start = time.perf_counter()
            t_main = 0.0

            try:
                raw_image = Image.open(
                    os.path.join(image_folder, item["image"])
                ).convert("RGB")
                image_tensor = image_processor.preprocess(
                    raw_image, return_tensors="pt"
                )["pixel_values"][0]

                # ── Main inference pass ───────────────────────────────────────
                t0 = time.perf_counter()
                question_text = item["text"]
                qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
                      if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN + "\n"
                      ) + question_text

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

                input_ids = tokenizer_image_token(
                    conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).cuda()
                output_ids = _run_generate_safe(model, input_ids, image_tensor,
                                                hp, use_only, use_cache=True)

                answer = tokenizer.batch_decode(
                    output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
                )[0].strip().rstrip(stop_str).strip()
                t_main = time.perf_counter() - t0

                q_total = time.perf_counter() - q_start
                question_times.append(q_total)
                _time_total += q_total
                done = len(question_times)
                avg  = _time_total / done
                eta  = (len(questions) - start_idx - done) * avg
                tqdm.write(
                    f"[{item['image']}] "
                    f"infer={t_main:.1f}s  total={q_total:.1f}s  "
                    f"avg={avg:.1f}s  eta={eta/60:.1f}min"
                )

                out_f.write(json.dumps({
                    "question_id" : item["question_id"],
                    "image"       : item["image"],
                    "prompt"      : question_text,
                    "text"        : answer,
                    "model_id"    : model_name,
                    "use_only"    : use_only,
                    "timing": {
                        "infer_s" : round(t_main, 3),
                        "total_s" : round(time.perf_counter() - q_start, 3),
                    },
                }) + "\n")
                out_f.flush()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                is_oom = "out of memory" in str(e).lower()
                if is_oom:
                    tqdm.write(f"[OOM-SKIP] {item['image']}: skipped (OOM even after retry)")
                    error_tag = "oom-skip"
                else:
                    tqdm.write(f"[ERROR] {item['image']}: {e}")
                    error_tag = f"error: {e}"
                out_f.write(json.dumps({
                    "question_id": item["question_id"],
                    "image":       item["image"],
                    "error":       error_tag,
                }) + "\n")
                out_f.flush()

            finally:
                del raw_image, image_tensor
                del input_ids, output_ids
                gc.collect()
                torch.cuda.empty_cache()

    job_elapsed = time.perf_counter() - job_start
    n = len(question_times)
    print(f"\n{'='*50}")
    print(f"  Images processed : {n}")
    print(f"  Total job time   : {job_elapsed/60:.1f} min  ({job_elapsed:.0f}s)")
    if n:
        print(f"  Mean per image   : {sum(question_times)/n:.1f}s")
        print(f"  Min / Max        : {min(question_times):.1f}s / {max(question_times):.1f}s")
        print(f"  Throughput       : {3600/(sum(question_times)/n):.0f} images/hr")
    print(f"  Results file     : {answers_file}")
    print(f"{'='*50}")


# ─────────────────────────────────────────────────────────────────────────────

# Argparse destinations that may override config.json. Declared as data rather
# than as a wall of near-identical if-statements, so adding a flag means adding
# a name here instead of another line that is easy to mistype or omit.
# The merge itself lives in run_config.py, which imports nothing beyond the
# standard library and is therefore testable outside the container.
_CFG_PATH_KEYS = (
    "model_path", "model_base", "image_folder", "question_file", "answers_file",
)
_CFG_HP_KEYS = (
    "conv_mode", "use_only", "enhance_layer_index", "js_gamma",
    "ritual_alpha_pos", "ritual_alpha_neg", "ritual_beta",
    "temperature", "top_p", "top_k", "seed", "max_new_tokens",
)


def _load_config(path: str) -> dict:
    """Load config.json. Facade over RunConfig.load()."""
    return RunConfig.load(path).to_dict()


def _make_answers_path(cfg: dict, run_name, model_tag: str = "llava") -> str:
    """Return the auto-suffixed answers file path: base_{model_tag}_{mode}[_{run_name}].ext"""
    base, ext = os.path.splitext(cfg["paths"]["answers_file"])
    mode_tag = "only" if cfg["hyperparameters"]["use_only"] else "baseline"
    name_tag = f"_{run_name}" if run_name else ""
    return f"{base}_{model_tag}_{mode_tag}{name_tag}{ext}"


def _merge(cfg: dict, args) -> dict:
    """Apply any explicitly-set CLI args over the config.

    Only arguments that are not None are applied, so argparse defaults never
    silently beat config.json. See run_config.RunConfig.apply_overrides().
    """
    RunConfig(cfg).apply_overrides(args, _CFG_PATH_KEYS, _CFG_HP_KEYS)

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="CASTOR: LLaVA-1.5 inference on maritime disaster images with ONLY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", default=os.path.join(_HERE, "config.json"),
                        metavar="PATH", help="Base config.json; CLI args override it.")
    parser.add_argument("--run-name", default=None, metavar="TAG",
                        help="Label appended to the auto-generated output filename. "
                             "Ignored when --answers-file is set explicitly.")
    parser.add_argument("--model-tag", default="llava", metavar="TAG",
                        help="Model identifier prefix in the output filename "
                             "(e.g. 'qwen3vl8b' → answers_qwen3vl8b_only.jsonl). "
                             "Ignored when --answers-file is set explicitly.")

    g = parser.add_argument_group("paths (override config.json)")
    g.add_argument("--model-path",    default=None, metavar="PATH")
    g.add_argument("--model-base",    default=None, metavar="PATH")
    g.add_argument("--image-folder",  default=None, metavar="PATH")
    g.add_argument("--question-file", default=None, metavar="PATH")
    g.add_argument("--answers-file",  default=None, metavar="PATH")

    g = parser.add_argument_group("mode")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--use-only", dest="use_only", action="store_true",
                    default=None, help="Enable ONLY layer intervention (overrides config).")
    mx.add_argument("--no-only",  dest="use_only", action="store_false",
                    help="Disable ONLY / baseline mode (overrides config).")

    g = parser.add_argument_group("ONLY hyperparameters (override config.json)")
    g.add_argument("--enhance-layer-index", type=int,   default=None, metavar="N",
                   help="Which transformer layer to suppress for the contrast pass.")
    g.add_argument("--js-gamma",            type=float, default=None, metavar="F",
                   help="TVD threshold switching additive/subtractive branch.")
    g.add_argument("--ritual-alpha-pos",    type=float, default=None, metavar="F",
                   help="α+ weight: additive branch (TVD < js_gamma).")
    g.add_argument("--ritual-alpha-neg",    type=float, default=None, metavar="F",
                   help="α- weight: subtractive branch (TVD ≥ js_gamma).")
    g.add_argument("--ritual-beta",         type=float, default=None, metavar="F",
                   help="Adaptive plausibility filter cutoff.")

    g = parser.add_argument_group("sampling (override config.json)")
    g.add_argument("--conv-mode",      default=None)
    g.add_argument("--temperature",    type=float, default=None, metavar="F")
    g.add_argument("--top-p",          type=float, default=None, metavar="F")
    g.add_argument("--top-k",          type=int,   default=None, metavar="N")
    g.add_argument("--seed",           type=int,   default=None, metavar="N")
    g.add_argument("--max-new-tokens", type=int,   default=None, metavar="N")

    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg = _merge(cfg, args)

    if args.answers_file is None:
        cfg["paths"]["answers_file"] = _make_answers_path(cfg, args.run_name, args.model_tag)

    run(cfg)


if __name__ == "__main__":
    main()
