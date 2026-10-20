# Pre-registration — DPO under Bradley–Terry misspecification

**Project:** `dpo-misspecification-study` (local working name: `dpo-misspecification-sa`)
**Author:** Arush Badhe
**Date frozen:** 2026-10-13
**Status:** Frozen before the first sweep. This file is not edited after the freezing commit; any change goes in the dated "Amendments" appendix at the bottom with a justification.

## 1. Problem statement

Measure how, and explain why, DPO's learned policy degrades when the preference data violate the Bradley–Terry assumption it encodes.

## 2. Research question

Under what kinds and magnitudes of preference misspecification does DPO's learned policy degrade, and is the degradation graceful (bias proportional to the misspecification level ε, as a projected-fixed-point analysis predicts) or catastrophic (reward hacking such as verbosity growth that compounds with training)?

## 3. Hypotheses and pre-specified numeric tests

### H1 — Length is absorbed into the implicit reward
Prediction: under p*(y_w ≻ y_l | x) = σ(r*(y_w) − r*(y_l) + γ(|y_w| − |y_l|)), DPO converges to π ∝ π_ref · exp((r* + γ|y|)/β); mean length rises monotonically in γ/β.

- **Confirmed if:** mean response length at γ = 0.3 exceeds γ = 0 by more than 25 % in at least 4 of 5 (γ-level × seed) comparisons, AND KL(π‖π_ref) rises more than 2× over the same range.
- **Refuted if:** the length change is under 5 %, OR length is non-monotone in γ (Spearman ρ < 0.6).

### H2 — Mixture misspecification degrades gracefully then breaks
Prediction: under p* = Σ_k w_k σ(r_k(y_w) − r_k(y_l)), DPO converges to the BT information projection; implicit-reward margins compress; win-rate loss grows ≈ linearly in ε = min_r E_x KL(p*(·|x) ‖ BT_r(·|x)) for small ε, then nonlinearly.

- **Confirmed if:** win-rate loss against the majority annotator is monotone in ε with a linear fit R² > 0.8 for ε ≤ 0.1, AND the implicit-reward margin IQR shrinks by more than 30 % at the largest ε.
- **Refuted if:** win-rate loss at the largest ε is within the 95 % CI of zero.

### H3 — The ICLR-2026 fix works
Prediction: the correction of Gopalan et al. (ICLR 2026) recovers most of the lost win-rate in the mixture setting at the cost of higher variance.

- **Confirmed if:** the fix recovers at least 50 % of the win-rate gap at the largest ε in at least 2 of 3 seeds.
- **Refuted if:** recovery is under 20 %, OR the fix loses more than 3 win-rate points at ε = 0.

### Secondary, pre-specified analyses (not pass/fail)
- Fit E‖θ_n − θ_T‖² ≈ C·a_n + b(ε) on the online two-timescale runs and report whether b(ε) is linear in ε (slope, R²).
- Spearman calibration of the implicit reward against the true reward r*, per ε level.
- Federated ablation: whether the server policy's win-rate against the majority annotator drops relative to centralised DPO on the same data.

## 4. Expected-to-fail control

BT-consistent label-flip noise η = 0.1 (labels flipped independently of x, y). This is well-specified noise and **must not** change mean length by more than 5 % or win-rate by more than the seed-to-seed 95 % CI. If it does, the harness is declared broken and no sweep result is reported until the control passes.

## 5. Models, data and fixed settings

| Item | Value |
|---|---|
| Primary model | Qwen2.5-0.5B-Instruct, LoRA rank 16 (alpha 32, dropout 0.05, q/k/v/o projections) |
| Replication model (full budget only) | Qwen2.5-1.5B-Instruct |
| Fallback if too slow | SmolLM-360M or Pythia-410M |
| Prompts | 10k fixed prompts from a public instruction set (5k in the minimum version); 500 held-out evaluation prompts (300 in the minimum version) |
| Pair sampling | 2 samples per prompt from the reference model, temperature 0.8, max 256 new tokens |
| Ground-truth reward r* | small public reward model (sentiment/helpfulness scorer) |
| γ levels | {0, 0.05, 0.1, 0.2, 0.3} |
| ε levels | 5 mixture weights with minority mass in {0, 0.1, 0.2, 0.35, 0.5}; K = 2 (K = 3 as ablation) |
| β | 0.1 unless swept over {0.05, 0.1, 0.5} |
| Epochs | 1 |
| Decoding at evaluation | greedy |
| Seeds | {0, 1, 2} in every cell (5 seeds added only if CIs overlap on the headline cell) |

## 6. Statistics

- Every reported number is a mean over the 3 seeds with a 95 % bootstrap CI (10,000 resamples).
- Win-rate comparisons are paired over the same 500 evaluation prompts.
- Monotonicity is tested with Spearman ρ over γ (or ε) levels pooled across seeds.
- Linear-fit R² for H2 uses the three smallest ε levels (ε ≤ 0.1 after numerical computation of ε).

## 7. What will be reported regardless of outcome

The negative-results section lists every hypothesis with its outcome (confirmed / refuted / under-powered) and the mechanistic explanation we now believe. Under-powered is used only when the pre-specified CI contains both the confirm and refute thresholds.

## 8. Amendments

Format: `YYYY-MM-DD — what changed — why — git commit`. All amendments below were made **before any experiment was run** (Week 0), following an independent review recorded in [`REVIEW.md`](REVIEW.md).

- 2026-10-18 — **H3 rewritten.** Old: "the ICLR-2026 fix recovers most of the lost win-rate in the mixture setting." New: "AuxDPO recovers ≥ 50 % of the gap in a policy-class-misspecification cell (rank-2 LoRA on q/v only, well-specified labels) in ≥ 2 of 3 seeds, **and** recovers < 20 % of the gap in the largest-ε mixture cell. Refuted if AuxDPO recovers > 50 % of the mixture gap or < 20 % of the policy-class gap." — Why: Gopalan et al. (ICLR 2026, arXiv 2510.20413) study misspecification of the *policy class* under BT-distributed labels; AuxDPO still fits a BT model to the labels, so under a length-biased or mixture annotator it learns the biased target. The old H3 was likely false by construction. The new H3 is a decomposition that separates the two kinds of misspecification. — commit: this one.
- 2026-10-18 — **Added independent variable: policy-class capacity** {full LoRA r16 on q/k/v/o; restricted LoRA r2 on q/v}. — Why: needed for the new H3. — commit: this one.
- 2026-10-18 — **Added dependent variable: likelihood displacement** (log-probability of y_w before vs after training). — Why: Razin et al. (ICLR 2025) show DPO can reduce the probability of preferred responses; we must rule this out as the cause of win-rate loss before attributing it to misspecification. — commit: this one.
- 2026-10-18 — **H1 reframed as a calibration check.** The test is unchanged; the text now says the γ/β-nats-per-token constant is a closed-form corollary of DPO's optimum and that with LoRA and one epoch only the slope and monotonicity are tested. — Why: R-DPO (Park et al. 2024) and CoDPO (NeurIPS 2026) already establish length exploitation; the contribution is the calibration against a known γ, not the existence of the effect. — commit: this one.
- 2026-10-18 — **Dropped the federated-preference ablation** (secondary analysis 3). — Why: Distributed DPO (arXiv 2605.20696) and FedPDPO (arXiv 2603.19741) already cover federated DPO with heterogeneous users. — commit: this one.
- 2026-10-18 — **Online two-timescale variant demoted to stretch.** Kept only if it yields a testable prediction (bias floor vs step-size ratio b_n/a_n). — Why: no DPO-specific stochastic-approximation analysis was found in search, which makes it interesting, but it is the most expensive and least certain arm. — commit: this one.
- 2026-10-18 — **dtype changed from bfloat16 to float32** for the 0.5B model. — Why: T4 has no bf16 and Qwen2.5 in fp16 can produce NaNs; the 0.5B model fits in fp32. — commit: this one.
- 2026-10-18 — **Per-run GPU-hour estimate raised from 0.4 h to 0.8 h** and evaluation budget from 3 h to 12 h. — Why: review back-of-envelope (≈ 6M tokens through policy forward+backward plus reference forward, plus TRL overhead) and the cost of generation + reward scoring per checkpoint. — commit: this one.
- 2026-10-18 — **Algorithm name `dpo_fix_iclr2026` renamed `auxdpo`.** — Why: the fix has a name. — commit: this one.
- 2026-10-20 — **Responses per prompt raised from 2 to K = 4; all 6 pairs per prompt labelled, 3 used for training (`pairs_per_prompt`).** — Why: with only two responses per prompt *every* pairwise probability is representable by some Bradley–Terry reward, so the misspecification distance ε would be identically zero and H2 untestable. With K ≥ 3 the BT projection is a real constraint; ε is computed per prompt by convex minimisation (`src/analysis/epsilon.py`, unit-tested: ε = 0 for any single annotator at any γ, ε > 0 and increasing in minority mass for the mixture). Generation cost of E1 doubles. — commit: code drop.
- 2026-10-20 — **Second annotator axis fixed as sentiment (SST-2 log-odds), z-scored; r_2 = s·(z(r*) + λ·z(f)).** — Why: the pre-registration said "formality versus brevity" without defining it; sentiment is cheap, independent of helpfulness, and gives a known axis that no single BT reward can absorb across prompts. — commit: code drop.
- 2026-10-20 — **The flip-noise control is reported with its own ε.** — Why: η-flips map p to η + (1−2η)p, which is only approximately inside the BT family for K ≥ 3 (unit test: its ε is far below the mixture's). The control's job is unchanged: it must not move length or win-rate. — commit: code drop.
- 2026-10-20 — **Length-normalised baseline implemented as R-DPO (Park et al. 2024, α = 0.005).** — Why: it is the published length regulariser; "length-normalised DPO" was unspecified. — commit: code drop.
- 2026-10-20 — **Training uses our own minimal DPO trainer, not TRL.** — Why: AuxDPO, R-DPO, the restricted-capacity cell and the online variant need the loss and parameter set under our control; TRL internals change between versions. — commit: code drop.
