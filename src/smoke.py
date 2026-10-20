"""Tiny end-to-end run: 24 prompts, K=3 samples, 20 training steps, 8 eval prompts.

Run this first on Kaggle (takes a few minutes on a T4 with the 0.5B model).  It
exercises every code path: build -> epsilon -> train (dpo, rdpo, auxdpo,
restricted capacity) -> eval -> analyze.  Nothing it produces is a result.
"""
from __future__ import annotations

import json
import os
import tempfile

from .common.config import resolve, run_dir
from .common.explog import finish_entry, start_entry


def smoke_config(model: str, tmp: str) -> dict:
    return {
        "experiment_id": "SMOKE",
        "model": {"name": model, "dtype": "auto", "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}, "capacity": "full"},
        "data": {"prompt_source": "alpaca_cleaned", "n_prompts": 24, "n_eval_prompts": 8, "seed": 0, "sampling": {"n_samples_per_prompt": 3, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 48, "batch_size": 8}},
        "annotator": {"reward_scale": 2.0, "axis_weight": 1.0, "gamma": 0.0, "mixture": {"K": 2, "minority_mass": 0.3}, "label_flip_eta": 0.0},
        "train": {"algorithm": "dpo", "beta": 0.1, "epochs": 1, "learning_rate": 5e-5, "lr_schedule": "cosine", "warmup_ratio": 0.1, "per_device_batch_size": 2, "gradient_accumulation_steps": 2, "max_length": 192, "max_prompt_length": 128, "checkpoint_every_steps": 0, "seed": 0, "gradient_checkpointing": False, "displacement_pairs": 16},
        "eval": {"max_new_tokens": 48, "batch_size": 8, "n_margin_pairs": 24},
        "logging": {"output_dir": os.path.join(tmp, "results")},
    }


def run_smoke(model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
    from .analysis.epsilon import epsilon_for_cell
    from .analysis.figures import analyze
    from .eval.evaluate import evaluate_cell
    from .prefgen.build import build_cache, load_cache
    from .train.trainer import train_cell

    tmp = tempfile.mkdtemp(prefix="dpo_smoke_")
    os.environ["RESULTS_ROOT"] = os.path.join(tmp, "results")
    cfg = smoke_config(model, tmp)
    build_cache(cfg, force=True)
    eps = epsilon_for_cell(load_cache(cfg), cfg["annotator"], max_prompts=24)
    print("epsilon (mixture, should be > 0):", eps)
    eps0 = epsilon_for_cell(load_cache(cfg), resolve(cfg, {"minority_mass": 0.0})["annotator"], max_prompts=24)
    print("epsilon (single annotator, should be ~0):", eps0)
    assert eps0["epsilon"] < 1e-4, eps0
    for cell in ({"algorithm": "dpo"}, {"algorithm": "rdpo"}, {"algorithm": "auxdpo"}, {"algorithm": "dpo", "capacity": "restricted"}):
        c = resolve(cfg, cell)
        d = run_dir(c)
        meta = start_entry(c, d, "smoke")
        m = train_cell(c)
        m.update(evaluate_cell(c))
        finish_entry(meta, m)
        print(cell, json.dumps({k: v for k, v in m.items() if isinstance(v, (int, float))}, default=float))
    analyze(os.path.join(tmp, "results"))
    print("SMOKE OK ->", tmp)
