"""Collect metrics from results/ and draw the paper's figures.

    python -m src.run analyze --results results

Figures (PNG + CSV of the plotted numbers, in results/figures/):
  fig1_length_vs_gamma.png      H1: mean length (CI) vs gamma/beta, KL vs gamma
  fig2_winrate_vs_epsilon.png   H2: win-rate loss vs numerical epsilon, margin IQR
  fig3_auxdpo_recovery.png      H3: recovery in the policy-class cell vs the mixture cell
  fig4_convergence.png          E6: ||theta_n - theta_T||^2 vs a_n with the C a_n + b fit
  control_eta.json              expected-to-fail control verdict
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.stats import linear_fit
from .hypotheses import _cell_table, control_eta, test_h1, test_h2, test_h3


def load_all_metrics(results_root: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for path in glob.glob(os.path.join(results_root, "*", "*", "metrics.json")):
        with open(path) as f:
            m = json.load(f)
        exp = m.get("_meta", {}).get("experiment_id") or path.split(os.sep)[-3]
        out.setdefault(exp, []).append(m)
    return out


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_length_vs_gamma(rows: List[Dict], beta: float, out: str) -> None:
    plt = _plt()
    df = _cell_table(rows)
    if df.empty or "gamma" not in df:
        return
    df = df[df.get("beta", beta) == beta] if "beta" in df else df
    g = df.groupby("gamma").agg(length=("mean_length", "mean"), lo=("mean_length_lo", "mean"), hi=("mean_length_hi", "mean"), kl=("kl_to_ref", "mean")).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].errorbar(g["gamma"] / beta, g["length"], yerr=[g["length"] - g["lo"], g["hi"] - g["length"]], marker="o")
    ax[0].set_xlabel("gamma / beta (nats per token)")
    ax[0].set_ylabel("mean response length (tokens)")
    ax[1].plot(g["gamma"], g["kl"], marker="o")
    ax[1].set_xlabel("gamma")
    ax[1].set_ylabel("KL(pi || pi_ref)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    g.to_csv(out.replace(".png", ".csv"), index=False)


def fig_winrate_vs_epsilon(rows: List[Dict], epsilons: Dict[float, float], out: str) -> None:
    plt = _plt()
    df = _cell_table(rows)
    if df.empty or "minority_mass" not in df:
        return
    df["eps"] = df["minority_mass"].map(epsilons)
    g = df.groupby("eps").agg(wr=("win_rate", "mean"), lo=("win_rate_lo", "mean"), hi=("win_rate_hi", "mean"), iqr=("margin_iqr", "mean"), rho=("calibration_rho", "mean")).reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].errorbar(g["eps"], g["wr"], yerr=[g["wr"] - g["lo"], g["hi"] - g["wr"]], marker="o")
    ax[0].set_xlabel("epsilon (KL to BT family)")
    ax[0].set_ylabel("win-rate vs reference (majority reward)")
    ax[1].plot(g["eps"], g["iqr"], marker="o")
    ax[1].set_xlabel("epsilon")
    ax[1].set_ylabel("implicit-reward margin IQR")
    ax[2].plot(g["eps"], g["rho"], marker="o")
    ax[2].set_xlabel("epsilon")
    ax[2].set_ylabel("calibration Spearman(implicit, r*)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    g.to_csv(out.replace(".png", ".csv"), index=False)


def fig_convergence(results_root: str, out: str) -> Dict:
    plt = _plt()
    curves = glob.glob(os.path.join(results_root, "E6*", "*", "convergence_curve.csv"))
    if not curves:
        return {}
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fits = {}
    for c in curves:
        df = pd.read_csv(c)
        fit = linear_fit(df["a_n"].values, df["sq_dist_to_final"].values)  # sq_dist ~ C a_n + b
        name = os.path.basename(os.path.dirname(c))
        fits[name] = fit
        ax.plot(df["a_n"], df["sq_dist_to_final"], marker="o", label=f"{name} (b={fit['intercept']:.3g})")
    ax.set_xlabel("a_n (fast step size)")
    ax.set_ylabel("||theta_n - theta_T||^2")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fits


def analyze(results_root: str, epsilons: Dict[float, float] | None = None, beta: float = 0.1) -> Dict:
    figdir = os.path.join(results_root, "figures")
    os.makedirs(figdir, exist_ok=True)
    allm = load_all_metrics(results_root)
    eps_path = os.path.join(results_root, "cache", "epsilon_by_minority_mass.json")
    if epsilons is None and os.path.exists(eps_path):
        with open(eps_path) as f:
            epsilons = {float(k): v["epsilon"] for k, v in json.load(f).items()}
    epsilons = epsilons or {}
    report: Dict = {}
    e3 = allm.get("E3_gamma_sweep", [])
    e4 = allm.get("E4_epsilon_sweep", [])
    e5 = allm.get("E5_auxdpo", [])
    ctrl = allm.get("E0_control_eta", [])
    if e3:
        fig_length_vs_gamma(e3, beta, os.path.join(figdir, "fig1_length_vs_gamma.png"))
        report["H1"] = test_h1(e3, beta)
    if e4 and epsilons:
        fig_winrate_vs_epsilon(e4, epsilons, os.path.join(figdir, "fig2_winrate_vs_epsilon.png"))
        report["H2"] = test_h2(e4, epsilons)
    if e5:
        aux = [m for m in e5 if m.get("_meta", {}).get("cell", {}).get("algorithm") == "auxdpo"]
        dpo = [m for m in e5 if m.get("_meta", {}).get("cell", {}).get("algorithm") == "dpo"]
        full = [m for m in dpo if m.get("_meta", {}).get("cell", {}).get("capacity", "full") == "full" and m.get("_meta", {}).get("cell", {}).get("minority_mass", 0) == 0]
        report["H3"] = test_h3(dpo, aux, full)
    if ctrl and e3:
        base = [m for m in e3 if m.get("_meta", {}).get("cell", {}).get("gamma") == 0]
        report["control_eta"] = control_eta(ctrl, base)
    report["E6_convergence_fits"] = fig_convergence(results_root, os.path.join(figdir, "fig4_convergence.png"))
    with open(os.path.join(figdir, "hypothesis_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in report.items()}, indent=2, default=str))
    return report
