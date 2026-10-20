"""The annotator model p*(y_i > y_j | x) and sampled labels.

For a prompt with K responses we know, for every response, z-scored r*(y), the
second axis f(y) and the token length |y|.  The data-generating preference is

    p*(i > j) = sum_k w_k * sigmoid( r_k(y_i) - r_k(y_j) + gamma * (|y_i| - |y_j|) )

with annotator rewards (all in the same z-units times ``reward_scale``):

    r_1 = s * z(r*)                      (majority annotator; the "true" reward)
    r_2 = s * (z(r*) + lam * z(f))       (minority annotator: also likes cheerful answers)
    r_3 = s * (z(r*) - lam * z(f))       (only when K_annotators = 3)

gamma != 0 is *length contamination*: still inside the BT family (the BT reward
is r* + gamma |y|), so DPO's optimum is predictable (H1).  A mixture with
minority mass > 0 is outside the BT family whenever lam != 0 (H2); its distance
epsilon is computed numerically in analysis/epsilon.py.  Label-flip noise eta
replaces p by eta + (1 - 2 eta) p: it is the control that should not matter.

All probabilities below are *exact*; sampled labels are Bernoulli draws from
them with the cell's seed, so every cell can also be analysed against the exact
p* rather than only against its noisy labels.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def annotator_rewards(rstar_z: np.ndarray, f_z: np.ndarray, cfg: Dict) -> np.ndarray:
    """Return an array of shape (K_annotators, n_responses) of per-annotator rewards (no length term)."""
    s = float(cfg.get("reward_scale", 2.0))
    lam = float(cfg.get("axis_weight", 1.0))
    K = int(cfg.get("mixture", {}).get("K", 1))
    rows = [s * rstar_z]
    if K >= 2:
        rows.append(s * (rstar_z + lam * f_z))
    if K >= 3:
        rows.append(s * (rstar_z - lam * f_z))
    return np.stack(rows[:K], axis=0)


def mixture_weights(cfg: Dict) -> np.ndarray:
    K = int(cfg.get("mixture", {}).get("K", 1))
    m = float(cfg.get("mixture", {}).get("minority_mass", 0.0))
    if K == 1 or m == 0.0:
        return np.array([1.0] + [0.0] * (K - 1))
    w = np.full(K, m / (K - 1))
    w[0] = 1.0 - m
    return w


def exact_pair_probs(rewards: np.ndarray, lengths: np.ndarray, gamma: float, weights: np.ndarray) -> Dict[tuple, float]:
    """``rewards``: (K, n) per-annotator rewards for the n responses of one prompt.

    Returns ``{(i, j): p*(i > j)}`` for all i < j.
    """
    n = rewards.shape[1]
    out = {}
    for i, j in combinations(range(n), 2):
        d = rewards[:, i] - rewards[:, j] + gamma * (lengths[i] - lengths[j])
        out[(i, j)] = float(np.dot(weights, sigmoid(d)))
    return out


def apply_flip_noise(p: float, eta: float) -> float:
    return eta + (1.0 - 2.0 * eta) * p


def annotate_prompt_group(group: List[Dict], cfg: Dict, rng: np.random.Generator) -> List[Dict]:
    """``group``: the K response rows of one prompt (dicts with rstar_z, f_z, n_tokens, ...).

    Returns one dict per pair with the exact probability, the noised probability and a sampled label.
    """
    rz = np.array([g["rstar_z"] for g in group])
    fz = np.array([g["f_z"] for g in group])
    ln = np.array([g["n_tokens"] for g in group], dtype=float)
    R = annotator_rewards(rz, fz, cfg)
    w = mixture_weights(cfg)
    gamma = float(cfg.get("gamma", 0.0))
    eta = float(cfg.get("label_flip_eta", 0.0))
    probs = exact_pair_probs(R, ln, gamma, w)
    pairs = []
    for (i, j), p in probs.items():
        p_noisy = apply_flip_noise(p, eta)
        label_i_wins = bool(rng.random() < p_noisy)
        wi, li = (i, j) if label_i_wins else (j, i)
        pairs.append(
            {
                "prompt_idx": group[0]["prompt_idx"],
                "i": i,
                "j": j,
                "p_exact": p,
                "p_noisy": p_noisy,
                "chosen": wi,
                "rejected": li,
                "true_majority_margin": float(R[0, wi] - R[0, li]),  # majority annotator's view of the sampled label
            }
        )
    return pairs
