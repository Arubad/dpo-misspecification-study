"""E1 + E2: build the response cache once.

Output: ``results/cache/responses.parquet`` with one row per (prompt, sample):
    prompt_idx, split, sample_idx, prompt_text, response, n_tokens,
    rstar_raw, rstar_z, f_raw, f_z, ref_logp
and ``results/cache/prompts.json`` with the train / eval prompt lists.

Every training cell reads this file and only re-samples *labels*, so the cache
is the expensive part (generation + two scoring passes + one teacher-forcing
pass) and everything else is cheap.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from ..common.kaggle_env import resolve_dtype, results_root, torch_device
from ..common.seed import derive_seed, seed_everything
from .prompts import load_prompts
from .refprobs import reference_logprobs
from .reward import DEFAULT_AXIS_MODEL, DEFAULT_REWARD_MODEL, score_reward_model, score_sentiment_axis, zscore
from .sample import load_causal_lm, sample_responses


def cache_paths(cfg: Dict) -> Dict[str, str]:
    root = os.path.join(results_root(), "cache")
    os.makedirs(root, exist_ok=True)
    return {"responses": os.path.join(root, "responses.parquet"), "prompts": os.path.join(root, "prompts.json")}


def build_cache(cfg: Dict, force: bool = False) -> str:
    paths = cache_paths(cfg)
    if os.path.exists(paths["responses"]) and not force:
        print(f"cache exists: {paths['responses']}")
        return paths["responses"]

    seed = int(cfg["data"].get("seed", 0))
    seed_everything(seed)
    device = torch_device()
    dtype = resolve_dtype(cfg["model"].get("dtype", "auto"))

    # 1. prompts
    d = cfg["data"]
    prompts = load_prompts(d["prompt_source"], int(d["n_prompts"]), int(d["n_eval_prompts"]), derive_seed(seed, "prompts"))
    with open(paths["prompts"], "w") as f:
        json.dump(prompts, f)

    # 2. K samples per prompt from the reference model (train split) + 1 greedy reference answer on eval prompts
    tok, model = load_causal_lm(cfg["model"]["name"], dtype, device)
    s = d["sampling"]
    rows = []
    for split in ("train", "eval"):
        res = sample_responses(
            model, tok, prompts[split], k=int(s["n_samples_per_prompt"]), temperature=float(s["temperature"]), top_p=float(s["top_p"]),
            max_new_tokens=int(s["max_new_tokens"]), batch_size=int(s.get("batch_size", 8)), seed=derive_seed(seed, f"sample_{split}"), system=d.get("system_prompt"),
        )
        for r in res:
            r["split"] = split
        rows.extend(res)
    df = pd.DataFrame(rows)
    df = df[df["response"].str.len() > 0].reset_index(drop=True)

    # 3. reference log-probs (same model, teacher forcing)
    t = cfg["train"]
    df["ref_logp"] = reference_logprobs(
        model, tok, df["prompt_text"].tolist(), df["response"].tolist(), int(t["max_prompt_length"]), int(t["max_length"]),
        batch_size=int(s.get("logprob_batch_size", 8)), system=d.get("system_prompt"),
    )
    del model
    import torch

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 4. ground-truth reward r* and the second axis f
    a = cfg["annotator"]
    df["rstar_raw"] = score_reward_model(a.get("true_reward_model", DEFAULT_REWARD_MODEL), df["prompt_text"].tolist(), df["response"].tolist(), device)
    df["f_raw"] = score_sentiment_axis(a.get("axis_model", DEFAULT_AXIS_MODEL), df["response"].tolist(), device)
    train_mask = df["split"] == "train"
    mu_r, sd_r = df.loc[train_mask, "rstar_raw"].mean(), df.loc[train_mask, "rstar_raw"].std() or 1.0
    mu_f, sd_f = df.loc[train_mask, "f_raw"].mean(), df.loc[train_mask, "f_raw"].std() or 1.0
    df["rstar_z"] = (df["rstar_raw"] - mu_r) / sd_r
    df["f_z"] = (df["f_raw"] - mu_f) / sd_f
    df.to_parquet(paths["responses"], index=False)
    print(f"wrote {len(df)} responses to {paths['responses']}")
    print("r* vs f correlation on train (should be small):", float(np.corrcoef(df.loc[train_mask, "rstar_z"], df.loc[train_mask, "f_z"])[0, 1]))
    return paths["responses"]


def load_cache(cfg: Dict) -> pd.DataFrame:
    return pd.read_parquet(cache_paths(cfg)["responses"])


__all__ = ["build_cache", "load_cache", "cache_paths", "zscore"]
