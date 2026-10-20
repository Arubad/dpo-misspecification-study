"""Evaluate one trained cell.

Dependent variables (all paired over the same held-out prompts, bootstrap CIs):
  win_rate            P(r_1(policy answer) > r_1(reference answer)) under the *majority* annotator
  mean_length, p90    tokens of greedy answers
  kl_to_ref           E_{y ~ pi}[log pi(y|x) - log pi_ref(y|x)] from one temperature-1 sample per prompt
  margin_iqr          IQR of beta*h on held-out labelled pairs (implicit-reward margin compression, H2)
  calibration_rho     Spearman( beta * log pi/pi_ref (y),  r*(y) ) over held-out responses
  rstar_mean          mean true reward of the greedy answers (the quantity DPO is supposed to raise)

The reference answers and their r* are generated once (cached in results/cache/eval_ref.parquet)
so every cell is compared against the same baseline.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ..common.config import run_dir
from ..common.kaggle_env import resolve_dtype, results_root, torch_device
from ..common.seed import derive_seed
from ..common.stats import bootstrap_ci, iqr, paired_bootstrap_diff, spearman, win_rate
from ..prefgen.annotate import annotator_rewards
from ..prefgen.build import cache_paths, load_cache
from ..prefgen.reward import DEFAULT_REWARD_MODEL, score_reward_model
from ..prefgen.sample import load_causal_lm, sample_responses
from ..train.dataset import collate, encode_prompt_response, sequence_logprobs
from ..train.pairs import build_pairs


def _eval_prompts(cfg: Dict) -> List[str]:
    with open(cache_paths(cfg)["prompts"]) as f:
        p = json.load(f)["eval"]
    n = int(cfg["data"].get("n_eval_prompts", len(p)))
    return p[:n]


def reference_answers(cfg: Dict, device, dtype) -> pd.DataFrame:
    """Greedy answers of the reference model on eval prompts, scored once and cached."""
    path = os.path.join(results_root(), "cache", "eval_ref.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    prompts = _eval_prompts(cfg)
    tok, model = load_causal_lm(cfg["model"]["name"], dtype, device)
    res = sample_responses(model, tok, prompts, k=1, temperature=1.0, top_p=1.0, max_new_tokens=int(cfg["eval"]["max_new_tokens"]), batch_size=int(cfg["eval"].get("batch_size", 8)), seed=0, greedy=True)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    df = pd.DataFrame(res)
    df["rstar_raw"] = score_reward_model(cfg["annotator"].get("true_reward_model", DEFAULT_REWARD_MODEL), df["prompt_text"].tolist(), df["response"].tolist(), device)
    df.to_parquet(path, index=False)
    return df


def _zstats(cfg: Dict) -> Dict[str, float]:
    tr = load_cache(cfg)
    tr = tr[tr["split"] == "train"]
    return {"mu_r": float(tr["rstar_raw"].mean()), "sd_r": float(tr["rstar_raw"].std() or 1.0)}


@torch.no_grad()
def _implicit_rewards(model, tok, rows: pd.DataFrame, beta: float, mp: int, ml: int, device, batch: int = 4) -> np.ndarray:
    """beta * (log pi(y|x) - log pi_ref(y|x)) for cached responses; pi_ref is the model with adapters disabled."""
    enc = [encode_prompt_response(tok, p, r, mp, ml) for p, r in zip(rows["prompt_text"], rows["response"])]
    out = np.zeros(len(enc))
    for s in range(0, len(enc), batch):
        b = collate(enc[s : s + batch], tok.pad_token_id, device)
        lp = sequence_logprobs(model, b)
        with model.disable_adapter():
            lr = sequence_logprobs(model, b)
        out[s : s + batch] = (beta * (lp - lr)).float().cpu().numpy()
    return out


def evaluate_cell(cfg: Dict, adapter_dir: str | None = None) -> Dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch_device()
    dtype = resolve_dtype(cfg["model"].get("dtype", "auto"))
    out_dir = run_dir(cfg)
    adapter_dir = adapter_dir or os.path.join(out_dir, "adapter")
    beta = float(cfg["train"]["beta"])
    mp, ml = int(cfg["train"]["max_prompt_length"]), int(cfg["train"]["max_length"])
    ev = cfg["eval"]

    ref = reference_answers(cfg, device, dtype)
    prompts = ref["prompt_text"].tolist()

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()

    # greedy answers
    pol = pd.DataFrame(sample_responses(model, tok, prompts, k=1, temperature=1.0, top_p=1.0, max_new_tokens=int(ev["max_new_tokens"]), batch_size=int(ev.get("batch_size", 8)), seed=0, greedy=True))
    # one temperature-1 sample per prompt for the KL estimate
    smp = pd.DataFrame(sample_responses(model, tok, prompts, k=1, temperature=1.0, top_p=1.0, max_new_tokens=int(ev["max_new_tokens"]), batch_size=int(ev.get("batch_size", 8)), seed=derive_seed(int(cfg["train"]["seed"]), "kl")))
    tok.padding_side = "right"
    kl_terms = _implicit_rewards(model, tok, smp, 1.0, mp, ml, device)  # log pi - log pi_ref on pi's own samples

    # true reward of the greedy answers, same z-units as training
    zs = _zstats(cfg)
    pol["rstar_raw"] = score_reward_model(cfg["annotator"].get("true_reward_model", DEFAULT_REWARD_MODEL), pol["prompt_text"].tolist(), pol["response"].tolist(), device)
    pol_r = (pol["rstar_raw"].values - zs["mu_r"]) / zs["sd_r"]
    ref_r = (ref["rstar_raw"].values - zs["mu_r"]) / zs["sd_r"]
    s = float(cfg["annotator"].get("reward_scale", 2.0))
    pol_r1, ref_r1 = s * pol_r, s * ref_r  # majority annotator r_1 = s * z(r*)

    # held-out labelled pairs for margins and calibration
    cache = load_cache(cfg)
    cache["row_id"] = np.arange(len(cache))
    held = build_pairs(cache, cfg["annotator"], int(cfg["train"]["seed"]), "eval")
    held = held.iloc[: int(ev.get("n_margin_pairs", 600))]
    hw = _implicit_rewards(model, tok, held.rename(columns={"chosen": "response"})[["prompt_text", "response"]], beta, mp, ml, device)
    hl = _implicit_rewards(model, tok, held.rename(columns={"rejected": "response"})[["prompt_text", "response"]], beta, mp, ml, device)
    margins = hw - hl
    calib_rho = spearman(np.concatenate([hw, hl]), np.concatenate([held["rstar_w"].values, held["rstar_l"].values]))

    pol.to_parquet(os.path.join(out_dir, "eval_generations.parquet"), index=False)
    wr = win_rate(pol_r1, ref_r1)
    metrics = {
        "win_rate": wr,
        "win_rate_ci": bootstrap_ci((pol_r1 > ref_r1).astype(float) + 0.5 * (pol_r1 == ref_r1)),
        "rstar_mean": float(pol_r.mean()),
        "rstar_gain_vs_ref": paired_bootstrap_diff(pol_r, ref_r),
        "mean_length": bootstrap_ci(pol["n_tokens"].values.astype(float)),
        "p90_length": float(np.quantile(pol["n_tokens"].values, 0.9)),
        "ref_mean_length": float(ref["n_tokens"].mean()),
        "kl_to_ref": bootstrap_ci(kl_terms),
        "margin_iqr": iqr(margins),
        "margin_mean": float(margins.mean()),
        "margin_sign_agreement": float((margins > 0).mean()),
        "calibration_rho": calib_rho,
        "n_eval_prompts": int(len(prompts)),
    }
    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
    return metrics
