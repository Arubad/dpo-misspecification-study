"""dpo-misspecification-sa — DPO under Bradley–Terry misspecification, analysed as stochastic approximation.

Planned modules (see src/README.md):
    prefgen   — synthetic preference generator with exactly known ground-truth reward
    train     — DPO variants (vanilla, length-normalised, AuxDPO, two-timescale stretch)
    eval      — win-rate, length, KL, margins, calibration
    analysis  — ε computation, convergence-curve fits, figures
    reproduce — one-command reproduction of the main plot
"""

__version__ = "0.0.0-prereg"
