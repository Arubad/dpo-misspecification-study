"""Minimal DPO trainer on a LoRA policy with cached reference log-probs.

One call to ``train_cell(cfg)`` trains one cell end to end and returns the
adapter directory.  Checkpoints are LoRA adapters only (never merged models).
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..common.config import run_dir
from ..common.kaggle_env import resolve_dtype, torch_device
from ..common.seed import derive_seed, seed_everything
from ..prefgen.build import load_cache
from ..prefgen.prompts import chat_format
from .dataset import collate, encode_prompt_response, sequence_logprobs
from .dpo_loss import compute_loss
from .pairs import build_pairs, label_sanity


def lora_targets(capacity: str) -> Dict:
    """``full``: r16 on q/k/v/o.  ``restricted``: r2 on q/v only (policy-class-misspecification cell for H3)."""
    if capacity == "restricted":
        return {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
    return {}


def load_policy(cfg: Dict, device, dtype, adapter: Optional[str] = None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.config.use_cache = False
    if cfg["train"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        l = dict(cfg["model"]["lora"])
        l.update(lora_targets(cfg["model"].get("capacity", "full")))
        lcfg = LoraConfig(r=int(l["r"]), lora_alpha=int(l["alpha"]), lora_dropout=float(l["dropout"]), target_modules=list(l["target_modules"]), task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
    model.to(device)
    return tok, model


class AuxOffsets(torch.nn.Module):
    """One free scalar per training response (AuxDPO), centred within each prompt's response group."""

    def __init__(self, n_responses: int, group_ids: torch.Tensor):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(n_responses))
        self.register_buffer("group_ids", group_ids)
        self.n_groups = int(group_ids.max().item()) + 1

    def centred(self) -> torch.Tensor:
        sums = torch.zeros(self.n_groups, device=self.delta.device).index_add(0, self.group_ids, self.delta)
        counts = torch.zeros(self.n_groups, device=self.delta.device).index_add(0, self.group_ids, torch.ones_like(self.delta))
        return self.delta - (sums / counts.clamp(min=1))[self.group_ids]


def _lr_at(step: int, total: int, base: float, warmup: float, schedule: str) -> float:
    w = max(1, int(total * warmup))
    if step < w:
        return base * (step + 1) / w
    if schedule == "constant":
        return base
    prog = (step - w) / max(1, total - w)
    return base * 0.5 * (1 + math.cos(math.pi * prog))


def train_cell(cfg: Dict, expected_note: str = "") -> Dict:
    t = cfg["train"]
    seed = int(t["seed"])
    seed_everything(derive_seed(seed, "lora"))
    device = torch_device()
    dtype = resolve_dtype(cfg["model"].get("dtype", "auto"))
    out_dir = run_dir(cfg)

    # ---- data
    responses = load_cache(cfg)
    responses["row_id"] = np.arange(len(responses))
    pairs = build_pairs(responses, cfg["annotator"], seed, "train", cfg["data"].get("pairs_per_prompt"))
    n_limit = cfg["data"].get("max_train_pairs")
    if n_limit:
        pairs = pairs.iloc[: int(n_limit)]
    sanity = label_sanity(pairs)
    print("label sanity:", sanity)
    pairs.to_parquet(os.path.join(out_dir, "train_pairs.parquet"), index=False)

    tok, model = load_policy(cfg, device, dtype)
    system = cfg["data"].get("system_prompt")
    mp, ml = int(t["max_prompt_length"]), int(t["max_length"])
    enc_w = [encode_prompt_response(tok, p, r, mp, ml, system) for p, r in zip(pairs["prompt_text"], pairs["chosen"])]
    enc_l = [encode_prompt_response(tok, p, r, mp, ml, system) for p, r in zip(pairs["prompt_text"], pairs["rejected"])]
    ref_w = torch.tensor(pairs["ref_w"].values, dtype=torch.float32)
    ref_l = torch.tensor(pairs["ref_l"].values, dtype=torch.float32)
    len_w = torch.tensor(pairs["len_w"].values, dtype=torch.float32)
    len_l = torch.tensor(pairs["len_l"].values, dtype=torch.float32)

    algorithm = t["algorithm"]
    params: List[torch.nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    aux: Optional[AuxOffsets] = None
    if algorithm == "auxdpo":
        gid = torch.tensor(responses["prompt_idx"].values, dtype=torch.long)
        gid = torch.unique(gid, return_inverse=True)[1]
        aux = AuxOffsets(len(responses), gid).to(device)
        params = params + [aux.delta]
    opt = torch.optim.AdamW(params, lr=float(t["learning_rate"]), weight_decay=float(t.get("weight_decay", 0.0)))

    bs = int(t["per_device_batch_size"])
    accum = int(t["gradient_accumulation_steps"])
    epochs = int(t.get("epochs", 1))
    steps_per_epoch = math.ceil(len(pairs) / (bs * accum))
    total_steps = steps_per_epoch * epochs
    use_amp = dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ckpt_every = int(t.get("checkpoint_every_steps", 0) or 0)

    history: List[Dict] = []
    model.train()
    step = 0
    micro = 0
    t0 = time.time()
    order_rng = np.random.default_rng(derive_seed(seed, "order"))
    for ep in range(epochs):
        order = order_rng.permutation(len(pairs))
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            bw = collate([enc_w[i] for i in idx], tok.pad_token_id, device)
            bl = collate([enc_l[i] for i in idx], tok.pad_token_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                lp_w = sequence_logprobs(model, bw)
                lp_l = sequence_logprobs(model, bl)
            idx_t = torch.tensor(idx)
            kw = dict(
                logp_w=lp_w, logp_l=lp_l, ref_w=ref_w[idx_t].to(device), ref_l=ref_l[idx_t].to(device), beta=float(t["beta"]),
                len_w=len_w[idx_t].to(device), len_l=len_l[idx_t].to(device), alpha=float(t.get("rdpo_alpha", 0.0)),
            )
            if aux is not None:
                d = aux.centred()
                cw = torch.tensor(pairs["chosen_id"].values[idx], device=device)
                cl = torch.tensor(pairs["rejected_id"].values[idx], device=device)
                kw.update(delta_w=d[cw], delta_l=d[cl], delta_all=aux.delta, lam=float(t.get("auxdpo_lambda", 0.1)))
            res = compute_loss(algorithm, **kw)
            loss = res["loss"] / accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % accum == 0:
                lr = _lr_at(step, total_steps, float(t["learning_rate"]), float(t.get("warmup_ratio", 0.05)), t.get("lr_schedule", "cosine"))
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, float(t.get("max_grad_norm", 1.0)))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "loss": float(res["loss"].item()), "margin": float(res["margin_mean"].item()), "reward_acc": float(res["reward_acc"].item()), "lr": lr, "elapsed_s": time.time() - t0}
                history.append(rec)
                if step % 10 == 0 or step == 1:
                    print(json.dumps(rec))
                if not math.isfinite(rec["loss"]):
                    raise RuntimeError("non-finite loss: switch dtype to float32 (T4 has no bf16; fp16 Qwen can overflow)")
                if ckpt_every and step % ckpt_every == 0:
                    model.save_pretrained(os.path.join(out_dir, f"ckpt_step{step:05d}"))
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    if aux is not None:
        torch.save(aux.delta.detach().cpu(), os.path.join(out_dir, "aux_delta.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_history.csv"), index=False)

    # ---- likelihood displacement (Razin et al.): did log pi(y_w) go *down* during training?
    disp = likelihood_displacement(model, tok, pairs, ref_w, mp, ml, device, n=int(t.get("displacement_pairs", 256)), system=system)
    summary = {
        "n_pairs": int(len(pairs)),
        "steps": step,
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "final_reward_acc": history[-1]["reward_acc"] if history else float("nan"),
        "train_gpu_hours": (time.time() - t0) / 3600.0,
        "adapter": adapter_dir,
        **{f"label_{k}": v for k, v in sanity.items()},
        **disp,
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@torch.no_grad()
def likelihood_displacement(model, tok, pairs: pd.DataFrame, ref_w: torch.Tensor, mp: int, ml: int, device, n: int = 256, system: str | None = None) -> Dict[str, float]:
    model.eval()
    sub = pairs.iloc[:n]
    enc = [encode_prompt_response(tok, p, r, mp, ml, system) for p, r in zip(sub["prompt_text"], sub["chosen"])]
    lps = []
    for s in range(0, len(enc), 4):
        b = collate(enc[s : s + 4], tok.pad_token_id, device)
        lps.append(sequence_logprobs(model, b).float().cpu())
    lp = torch.cat(lps)
    delta = (lp - ref_w[: len(lp)]).numpy()
    model.train()
    return {"displacement_mean_logp_delta_chosen": float(delta.mean()), "displacement_frac_chosen_decreased": float((delta < 0).mean())}
