"""Pre-registered pass/fail tests, applied mechanically to the collected metrics.

Each function returns ``{"verdict": "confirmed"|"refuted"|"under-powered"|"inconclusive", ...evidence}``.
The thresholds are copied from PREREGISTRATION.md (v0.2) and must not be edited
here without a dated amendment there.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.stats import bootstrap_ci, linear_fit, spearman


def _cell_table(rows: List[Dict]) -> pd.DataFrame:
    """Flatten metrics.json records into one row per cell with scalar columns."""
    flat = []
    for r in rows:
        d = {k: v for k, v in r.items() if not k.startswith("_") and not isinstance(v, (dict, list))}
        for k, v in r.items():
            if isinstance(v, dict) and "mean" in v:
                d[k] = v["mean"]
                d[f"{k}_lo"], d[f"{k}_hi"] = v.get("lo"), v.get("hi")
        d.update(r.get("_meta", {}).get("cell", {}))
        flat.append(d)
    return pd.DataFrame(flat)


def test_h1(rows: List[Dict], beta: float = 0.1) -> Dict:
    """H1: mean length at gamma=0.3 > 1.25 x length at gamma=0 in >= 4 of 5 (gamma-level x seed) comparisons,
    and KL rises > 2x over the range.  Refuted if change < 5% or Spearman(gamma, length) < 0.6."""
    df = _cell_table(rows)
    df = df[(df.get("beta", beta) == beta)] if "beta" in df else df
    if df.empty or "gamma" not in df:
        return {"verdict": "inconclusive", "reason": "no gamma cells"}
    base = df[df["gamma"] == 0].groupby("seed")["mean_length"].mean()
    top = df[df["gamma"] == df["gamma"].max()].groupby("seed")["mean_length"].mean()
    ratios = (top / base).dropna()
    rho = spearman(df["gamma"].values, df["mean_length"].values)
    kl_base = df[df["gamma"] == 0]["kl_to_ref"].mean()
    kl_top = df[df["gamma"] == df["gamma"].max()]["kl_to_ref"].mean()
    kl_ratio = float(kl_top / kl_base) if kl_base > 0 else float("inf")
    n_pass = int((ratios > 1.25).sum())
    change = float(ratios.mean() - 1)
    if change < 0.05 or rho < 0.6:
        verdict = "refuted"
    elif n_pass >= min(4, len(ratios)) and kl_ratio > 2:
        verdict = "confirmed"
    else:
        verdict = "under-powered"
    slope = linear_fit(df["gamma"].values / beta, df["mean_length"].values)
    return {"verdict": verdict, "length_ratio_by_seed": ratios.to_dict(), "n_seeds_over_1p25": n_pass, "spearman_gamma_length": rho, "kl_ratio": kl_ratio, "length_vs_gamma_over_beta_slope": slope}


def test_h2(rows: List[Dict], epsilons: Dict[float, float]) -> Dict:
    """H2: win-rate loss monotone in eps with linear R^2 > 0.8 for eps <= 0.1, and margin IQR shrinks > 30% at the largest eps.
    Refuted if win-rate loss at the largest eps is within the 95% CI of zero.  ``epsilons`` maps minority_mass -> epsilon."""
    df = _cell_table(rows)
    if df.empty or "minority_mass" not in df:
        return {"verdict": "inconclusive", "reason": "no mixture cells"}
    df["eps"] = df["minority_mass"].map(epsilons)
    base = df[df["minority_mass"] == 0]
    wr0 = base["win_rate"].mean()
    iqr0 = base["margin_iqr"].mean()
    agg = df.groupby("eps").agg(win_rate=("win_rate", "mean"), margin_iqr=("margin_iqr", "mean")).reset_index().sort_values("eps")
    agg["loss"] = wr0 - agg["win_rate"]
    top = df[df["eps"] == agg["eps"].max()]
    loss_top = bootstrap_ci((wr0 - top["win_rate"]).values)
    small = agg[agg["eps"] <= 0.1]
    fit = linear_fit(small["eps"].values, small["loss"].values) if len(small) >= 3 else {"r2": float("nan"), "slope": float("nan")}
    rho = spearman(agg["eps"].values, agg["loss"].values)
    iqr_shrink = float(1 - top["margin_iqr"].mean() / iqr0) if iqr0 > 0 else float("nan")
    if loss_top["lo"] <= 0 <= loss_top["hi"]:
        verdict = "refuted"
    elif rho > 0.9 and fit["r2"] > 0.8 and iqr_shrink > 0.30:
        verdict = "confirmed"
    else:
        verdict = "under-powered"
    return {"verdict": verdict, "curve": agg.to_dict("records"), "loss_at_max_eps_ci": loss_top, "linear_fit_small_eps": fit, "spearman_eps_loss": rho, "margin_iqr_shrink": iqr_shrink}


def test_h3(rows_dpo: List[Dict], rows_aux: List[Dict], rows_full_capacity: List[Dict]) -> Dict:
    """H3 (v0.2): AuxDPO recovers >= 50% of the policy-class gap (restricted vs full capacity, well-specified labels)
    in >= 2 of 3 seeds AND < 20% of the largest-eps mixture gap.  ``rows_*`` are the metrics of the relevant cells."""
    d = _cell_table(rows_dpo)
    a = _cell_table(rows_aux)
    f = _cell_table(rows_full_capacity)
    out: Dict = {}
    # policy-class cell: capacity == restricted, minority_mass == 0
    pc_d = d[(d.get("capacity") == "restricted") & (d.get("minority_mass", 0) == 0)] if "capacity" in d else d.iloc[0:0]
    pc_a = a[(a.get("capacity") == "restricted") & (a.get("minority_mass", 0) == 0)] if "capacity" in a else a.iloc[0:0]
    full = f[(f.get("minority_mass", 0) == 0)] if not f.empty else f
    rec_pc = []
    for seed in sorted(set(pc_d.get("seed", [])) & set(pc_a.get("seed", []))):
        gap = full["win_rate"].mean() - pc_d[pc_d["seed"] == seed]["win_rate"].mean()
        got = pc_a[pc_a["seed"] == seed]["win_rate"].mean() - pc_d[pc_d["seed"] == seed]["win_rate"].mean()
        rec_pc.append(got / gap if gap > 0 else float("nan"))
    # mixture cell: capacity full, largest minority_mass
    mx_d = d[(d.get("capacity", "full") == "full")] if "minority_mass" in d else d.iloc[0:0]
    rec_mx = float("nan")
    if not mx_d.empty and "minority_mass" in a:
        m = mx_d["minority_mass"].max()
        gap = full["win_rate"].mean() - mx_d[mx_d["minority_mass"] == m]["win_rate"].mean()
        got = a[(a.get("capacity", "full") == "full") & (a["minority_mass"] == m)]["win_rate"].mean() - mx_d[mx_d["minority_mass"] == m]["win_rate"].mean()
        rec_mx = float(got / gap) if gap > 0 else float("nan")
    n_pc = int(np.sum(np.array(rec_pc) >= 0.5)) if rec_pc else 0
    if not rec_pc or np.isnan(rec_mx):
        verdict = "inconclusive"
    elif rec_mx > 0.5 or n_pc == 0:
        verdict = "refuted"
    elif n_pc >= 2 and rec_mx < 0.2:
        verdict = "confirmed"
    else:
        verdict = "under-powered"
    out.update({"verdict": verdict, "recovery_policy_class_by_seed": rec_pc, "recovery_mixture_at_max_eps": rec_mx})
    return out


def control_eta(rows_control: List[Dict], rows_base: List[Dict]) -> Dict:
    """Expected-to-fail control: the eta=0.1 cell must not change length by > 5% or win-rate beyond the seed CI."""
    c, b = _cell_table(rows_control), _cell_table(rows_base)
    if c.empty or b.empty:
        return {"verdict": "inconclusive"}
    len_change = float(c["mean_length"].mean() / b["mean_length"].mean() - 1)
    wr_ci = bootstrap_ci(b["win_rate"].values) if len(b) > 1 else {"lo": b["win_rate"].min(), "hi": b["win_rate"].max()}
    wr_c = float(c["win_rate"].mean())
    ok = abs(len_change) < 0.05 and wr_ci["lo"] - 0.02 <= wr_c <= wr_ci["hi"] + 0.02
    return {"verdict": "passed" if ok else "FAILED: harness is broken, fix before reporting anything", "length_change": len_change, "win_rate_control": wr_c, "win_rate_base_ci": wr_ci}
