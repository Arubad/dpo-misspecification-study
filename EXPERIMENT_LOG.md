# Experiment log — dpo-misspecification-study

Every run gets an entry **before** it is launched (fill `Expected`) and is completed **after** it finishes (fill `Observed`, `Next change`). Entries are append-only; never rewrite history. The W&B project `dpo-misspec-sa` mirrors these entries, keyed by `config_hash`. Progress at the milestone level is tracked in [`ROADMAP.md`](ROADMAP.md).

Config hash = first 12 hex characters of `sha256` over the resolved YAML (`python -m src.utils.config_hash configs/<file>.yaml`).

## Entry format

```
## YYYY-MM-DD — <experiment id> — <one-line title>
- Config: configs/<file>.yaml  (hash: <12 hex>)
- Git commit: <short sha>
- Seeds: [..]
- Hardware: <GPU, platform>
- Wall-clock / GPU-h:
- Expected: <what the pre-registration predicts for this cell, with numbers>
- Observed: <numbers with 95 % CI; link to results/<run>/metrics.json and figure>
- Control status: <did the expected-to-fail control pass on this harness version?>
- Next change: <exactly one thing changed before the next run, and why>
```

---

## 2026-10-13 — TEMPLATE — E3 γ sweep, γ = 0.3, β = 0.1 (example entry, not a real run)

> **This entry is a template.** It illustrates the required fields with plausible placeholder values; nothing below was run.

- Config: configs/gamma_sweep_0p5b.yaml  (hash: `0000deadbeef`)
- Git commit: `abc1234`
- Seeds: [0, 1, 2]
- Hardware: 1× T4 16 GB, Colab Pro
- Wall-clock / GPU-h: 0.4 h per seed, 1.2 h total
- Expected: mean length at γ = 0.3 > 1.25 × mean length at γ = 0 (H1); KL(π‖π_ref) ≥ 2× the γ = 0 value; win-rate vs r* within CI of γ = 0 (length term not yet dominant at β = 0.1).
- Observed: _(to be filled) mean length X ± CI vs Y ± CI; KL A vs B; win-rate W ± CI; results/E3_g0p3_b0p1/metrics.json; figures/length_vs_gamma.png_
- Control status: η = 0.1 label-flip control run on commit `abc1234`: length change _%, win-rate change _ points (must be within CI).
- Next change: _(to be filled) e.g. "none — proceed to γ = 0.2" or "reduce LR 1e-5 → 5e-6 because training loss diverged at seed 1"._

---

_(real entries start below)_
