"""epsilon = min_r E_x KL( p*(. | x) || BT_r(. | x) ), computed numerically.

With only two responses per prompt any pairwise probability is a BT model, so
epsilon would be identically zero; that is why the generator samples K >= 3
responses per prompt and labels all K(K-1)/2 pairs (amendment of 2026-10-20).
For each prompt we solve the convex problem

    min_{r in R^K}  sum_{i<j} KL( p*_ij || sigmoid(r_i - r_j) )

by L-BFGS (the objective is a logistic-regression cross-entropy with soft
targets, convex in r), and average the minimum over prompts and pairs.  For a
single-annotator generator (minority_mass = 0, any gamma) the minimum is 0 up
to numerical tolerance, which the unit test checks.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


def _kl_bern(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def bt_projection_one_prompt(pair_probs: Dict[Tuple[int, int], float], n_items: int) -> Tuple[float, np.ndarray]:
    """Return (mean KL over pairs at the optimum, optimal r)."""
    pairs = list(pair_probs.keys())
    I = np.array([i for i, _ in pairs])
    J = np.array([j for _, j in pairs])
    P = np.array([pair_probs[k] for k in pairs], dtype=float)

    def f(r):
        d = r[I] - r[J]
        q = 1 / (1 + np.exp(-d))
        kl = _kl_bern(P, q).sum()
        # gradient of sum_pairs [-P log q - (1-P) log(1-q)] wrt r (the KL differs by a constant)
        g_d = q - P
        grad = np.zeros(n_items)
        np.add.at(grad, I, g_d)
        np.add.at(grad, J, -g_d)
        return kl, grad

    res = minimize(f, np.zeros(n_items), jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-10})
    return float(res.fun / len(pairs)), res.x


def epsilon_from_groups(groups: Sequence[Dict[Tuple[int, int], float]], n_items: int) -> Dict[str, float]:
    vals = [bt_projection_one_prompt(g, n_items)[0] for g in groups]
    v = np.array(vals)
    return {"epsilon": float(v.mean()), "epsilon_median": float(np.median(v)), "epsilon_std": float(v.std()), "n_prompts": int(len(v))}


def epsilon_for_cell(responses, annotator_cfg: Dict, max_prompts: int = 2000) -> Dict[str, float]:
    """Compute epsilon on the cached training responses for one annotator configuration (exact p*, no label noise)."""
    from ..prefgen.annotate import annotator_rewards, exact_pair_probs, mixture_weights

    df = responses[responses["split"] == "train"]
    groups: List[Dict[Tuple[int, int], float]] = []
    w = mixture_weights(annotator_cfg)
    gamma = float(annotator_cfg.get("gamma", 0.0))
    K = None
    for _, g in df.groupby("prompt_idx", sort=True):
        if len(g) < 3:
            continue
        g = g.sort_values("sample_idx")
        R = annotator_rewards(g["rstar_z"].values, g["f_z"].values, annotator_cfg)
        groups.append(exact_pair_probs(R, g["n_tokens"].values.astype(float), gamma, w))
        K = len(g) if K is None else min(K, len(g))
        if len(groups) >= max_prompts:
            break
    if not groups:
        raise ValueError("epsilon needs >= 3 responses per prompt in the cache")
    # all groups must have the same number of items for the solver; drop extras
    groups = [{k: v for k, v in g.items() if max(k) < K} for g in groups]
    return epsilon_from_groups(groups, K)
