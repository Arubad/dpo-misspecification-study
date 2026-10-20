"""DPO loss and its variants, written against pre-computed reference log-probs.

Notation: h = log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x).

dpo     : -log sigmoid(beta * h)                                 (Rafailov et al. 2023)
rdpo    : -log sigmoid(beta * h - alpha * (|y_w| - |y_l|))       (Park et al. 2024, length regulariser)
auxdpo  : -log sigmoid(beta * h + delta_w - delta_l) + lam * ||delta||^2
          where delta is one free scalar per response in the training set
          (Gopalan, Chowdhury, Banerjee, ICLR 2026).  delta lets the implicit
          reward represent targets the policy class cannot; the penalty keeps it
          from doing all the work.  Our implementation also centres delta within
          each prompt, which removes the prompt-level offset c(x) that DPO is
          invariant to.  Check against the paper's definition before relying on
          it (open item in REVIEW.md).
soft    : cross-entropy against a *probability* label p (used by the online
          two-timescale variant, where the label is the learned reward model's
          sigmoid(r_hat_w - r_hat_l)).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def dpo_logits(logp_w: torch.Tensor, logp_l: torch.Tensor, ref_w: torch.Tensor, ref_l: torch.Tensor, beta: float) -> torch.Tensor:
    return beta * ((logp_w - ref_w) - (logp_l - ref_l))


def loss_dpo(z: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(z).mean()


def loss_rdpo(z: torch.Tensor, len_w: torch.Tensor, len_l: torch.Tensor, alpha: float) -> torch.Tensor:
    return -F.logsigmoid(z - alpha * (len_w - len_l)).mean()


def loss_auxdpo(z: torch.Tensor, delta_w: torch.Tensor, delta_l: torch.Tensor, delta_all_for_penalty: torch.Tensor, lam: float) -> torch.Tensor:
    return -F.logsigmoid(z + delta_w - delta_l).mean() + lam * (delta_all_for_penalty**2).mean()


def loss_soft(z: torch.Tensor, p_label: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of sigmoid(z) against a soft label p in [0, 1]."""
    return -(p_label * F.logsigmoid(z) + (1 - p_label) * F.logsigmoid(-z)).mean()


def compute_loss(
    algorithm: str,
    logp_w: torch.Tensor,
    logp_l: torch.Tensor,
    ref_w: torch.Tensor,
    ref_l: torch.Tensor,
    beta: float,
    len_w: Optional[torch.Tensor] = None,
    len_l: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    delta_w: Optional[torch.Tensor] = None,
    delta_l: Optional[torch.Tensor] = None,
    delta_all: Optional[torch.Tensor] = None,
    lam: float = 0.0,
    p_label: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    z = dpo_logits(logp_w, logp_l, ref_w, ref_l, beta)
    if algorithm == "dpo":
        loss = loss_dpo(z)
    elif algorithm in ("rdpo", "dpo_lennorm"):
        loss = loss_rdpo(z, len_w, len_l, alpha)
    elif algorithm == "auxdpo":
        loss = loss_auxdpo(z, delta_w, delta_l, delta_all, lam)
    elif algorithm == "soft":
        loss = loss_soft(z, p_label)
    else:
        raise ValueError(f"unknown algorithm {algorithm}")
    with torch.no_grad():
        margin = z.detach()
        acc = (margin > 0).float().mean()
    return {"loss": loss, "margin_mean": margin.mean(), "reward_acc": acc, "margin": margin}
