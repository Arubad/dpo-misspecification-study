# When preferences lie: DPO under Bradley–Terry misspecification, analysed as stochastic approximation

**Status:** Pre-registration v0.2 · Week 0 of 10 · Pre-registered 2026-10-13, amended 2026-10-18 before any run · Models: Qwen2.5-0.5B-Instruct (primary), Qwen2.5-1.5B-Instruct (replication) · Seeds: 0, 1, 2

| Document | What it is |
|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Frozen hypotheses, numeric pass/fail tests, models, seeds. Changes only via the dated Amendments appendix. |
| [`REVIEW.md`](REVIEW.md) | Independent review of 2026-10-18: citation corrections, novelty against the literature, design flaws, what changed and why. Read this before the pre-registration. |
| [`ROADMAP.md`](ROADMAP.md) | Milestones as GitHub task lists. Tick them off; this is the progress tracker. |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | Append-only log, one entry per run, expectation written before launch. |
| [`READING.md`](READING.md) | Ordered reading list with the reason for each item. |

**Problem in one sentence.** We measure how, and explain why, DPO's learned policy degrades when the preference data violate the Bradley–Terry (BT) assumption that DPO encodes, and we separate that failure from the different failure Gopalan et al. (ICLR 2026) study, in which the policy class cannot express the true reward.

**Why this matters.** DPO is the default way to align small open models, and it assumes the annotator's preferences come from a single Bradley–Terry reward. Real preference data do not: annotators disagree, and they favour longer answers. Each of those failures has been studied separately (see [`REVIEW.md`](REVIEW.md) §3), but there is no controlled study on a small open LLM where the ground-truth reward is known exactly, each kind of misspecification is dialled in by hand on the same prompts, and the degradation is compared against a closed-form prediction. Without one we cannot say whether DPO fails gracefully or catastrophically, or which proposed fixes can work even in principle.

---

## 1. Research proposition

### The gap

Three strands of work at IISc point at this question from different directions, and none of them closes it.

The **estimation-theoretic strand** is Aditya Gopalan's. His ICLR 2026 oral, *"Why DPO is a Misspecified Estimator and How to Fix It"* ([arXiv 2510.20413](https://arxiv.org/abs/2510.20413)), shows that DPO is a reverse-KL projection of the true reward onto the set of rewards the policy class can express, that this projection can reverse preference order when the policy class is too small, and proposes AuxDPO, which adds a per-response auxiliary offset with a null-space penalty so the implicit reward can represent r* even when the policy cannot. **Note carefully: the misspecification in that paper is of the policy class, under the assumption that labels are Bradley–Terry.** This study asks the complementary question: what happens when the labels themselves are not Bradley–Terry. His AISTATS 2026 paper *"Bad Values but Good Behavior"* ([PMLR v300](https://proceedings.mlr.press/v300/banerjee26b.html)) shows, for bandits, that badly misspecified value estimates can coexist with good behaviour; it is motivation, not a method we use.

The **stochastic-approximation strand** supplies the analytical tools but has not been applied to preference optimisation. Gugan Thoppe's work on finite-time rates for TD and Q-learning and on federated stochastic approximation ([Thoppe research page](https://sites.google.com/site/gugancth/research); [arXiv 2510.07436](https://arxiv.org/abs/2510.07436)), and Shalabh Bhatnagar's stochastic-approximation lineage ([Bhatnagar homepage](https://www.csa.iisc.ac.in/~shalabh/)), give the ODE and two-timescale machinery. Borkar's *Stochastic Approximation: A Dynamical Systems Viewpoint* is the textbook.

The **empirical alignment literature** has each arm of this study separately: length exploitation and its regularisers (Park et al., R-DPO; CoDPO, NeurIPS 2026), annotator mixtures and their impossibility results (MaxMin-RLHF, ICML 2024; EM-DPO), DPO under label-flip noise (Provably Robust DPO, ICML 2024), and the Borda-count reading of DPO under hidden context (Siththaranjan et al.). None of them puts the three kinds of misspecification side by side on a known reward with a computed misspecification distance.

### The research question

Under what kinds and magnitudes of preference-model misspecification does DPO's learned policy degrade; is the degradation **graceful** (bias proportional to the misspecification level, as a projected-fixed-point analysis predicts) or **catastrophic** (reward hacking such as verbosity growth that compounds with training); and does a fix designed for policy-class misspecification (AuxDPO) help, hurt, or do nothing under preference-model misspecification?

### Why synthetic preferences, and why a sub-2B model

Because the ground-truth reward $r^*$ must be known exactly to measure anything. Our generator scores $r^*$ with a small reward model and injects nuisance terms with known coefficients: a length coefficient $\gamma$, a mixture over $K$ annotator rewards with weights $w_k$, and BT-consistent label-flip noise $\eta$ as the control that should not matter. A 0.5B model with LoRA fits a single T4; a 1.5B replication checks that the shape is not a small-model artefact.

---

## 2. Pre-registered hypotheses (v0.2)

Full text with dates and seeds is in [`PREREGISTRATION.md`](PREREGISTRATION.md); the 2026-10-18 amendments are in its appendix. The numerical tests are copied verbatim here.

**H1 — length is absorbed into the implicit reward (calibration check, not a discovery).** If the annotator model is $p^*(y_w \succ y_l \mid x) = \sigma\big(r^*(y_w) - r^*(y_l) + \gamma(|y_w| - |y_l|)\big)$, the BT class still contains the data-generating reward $r^* + \gamma|y|$, so the population optimum is a policy that pays exactly $\gamma/\beta$ nats per extra token. With LoRA and one epoch we do not reach the population optimum, so the test is on the *slope and monotonicity*, not the exact constant.
*Confirmed* if mean response length at $\gamma = 0.3$ exceeds $\gamma = 0$ by more than 25% in at least 4 of 5 ($\gamma$-level × seed) comparisons and KL-to-reference rises more than 2× over the same range. *Refuted* if the length change is under 5% or non-monotone (Spearman $\rho < 0.6$).

**H2 — mixture misspecification degrades gracefully, then breaks.** If $p^* = \sum_k w_k\, \sigma\big(r_k(y_w) - r_k(y_l)\big)$, no single BT reward reproduces $p^*$; DPO converges to the BT information projection, implicit-reward margins compress, and win-rate loss against the majority annotator grows approximately linearly in $\varepsilon = \min_r \mathbb{E}_x\, \mathrm{KL}\big(p^*(\cdot \mid x)\,\|\,\mathrm{BT}_r(\cdot \mid x)\big)$ for small $\varepsilon$, then nonlinearly.
*Confirmed* if win-rate loss is monotone in $\varepsilon$ with a linear fit $R^2 > 0.8$ for $\varepsilon \le 0.1$ and the implicit-reward margin IQR shrinks by more than 30% at the largest $\varepsilon$. *Refuted* if win-rate loss at the largest $\varepsilon$ is within the 95% CI of zero.

**H3 — AuxDPO separates the two kinds of misspecification (amended 2026-10-18).** AuxDPO (Gopalan et al., ICLR 2026) repairs the policy class's inability to express $r^*$; it still fits a BT model to the labels. Prediction: in a cell where the policy is deliberately too small to express $r^*$ (rank-2 LoRA on attention only, labels well-specified), AuxDPO recovers most of the win-rate gap to a full-capacity policy; in the mixture cells (full-capacity policy, labels misspecified), AuxDPO recovers none of the gap and may slightly widen it.
*Confirmed* if AuxDPO recovers at least 50% of the win-rate gap in the policy-class cell in at least 2 of 3 seeds **and** recovers less than 20% in the largest-$\varepsilon$ mixture cell. *Refuted* if AuxDPO recovers more than 50% of the mixture gap (the fix would then be doing something the derivation does not predict, which is itself worth reporting) or fails to recover 20% in the policy-class cell.

**Most interesting possible outcome.** If win-rate loss is flat in $\varepsilon$ while implicit-reward calibration (Spearman against $r^*$) collapses, DPO's policy is robust to misspecification even though its implicit reward is useless as a reward model. That would be a direct empirical instance of "bad values but good behaviour" and matters to anyone reusing DPO implicit rewards for data selection.

---

## 3. Experimental design

| Role | Variable | Levels |
|---|---|---|
| Independent | Length-contamination coefficient $\gamma$ | {0, 0.05, 0.1, 0.2, 0.3} |
| Independent | Annotator-mixture misspecification $\varepsilon$ | $K$ = 2 (3 as ablation); 5 mixture weights with minority mass in {0, 0.1, 0.2, 0.35, 0.5} |
| Independent | Policy-class capacity (for H3) | full LoRA r16 on q/k/v/o; restricted LoRA r2 on q/v only |
| Independent | Algorithm | DPO, length-normalised DPO, AuxDPO; online two-timescale DPO as stretch |
| Dependent | Win-rate vs true majority reward | 500 held-out prompts |
| Dependent | Response length | mean and 90th percentile |
| Dependent | $\mathrm{KL}(\pi \,\|\, \pi_{\mathrm{ref}})$ | per prompt, averaged |
| Dependent | Implicit-reward margin distribution | IQR |
| Dependent | Calibration of implicit vs true reward | Spearman $\rho$ |
| Dependent | Likelihood displacement | log-prob of $y_w$ before vs after training (to rule out Razin et al.'s mechanism) |
| Controlled | Base model | Qwen2.5-0.5B-Instruct, fp32 on T4 (no bf16; fp16 Qwen can NaN) |
| Controlled | Prompt set | 10k fixed prompts from a public instruction set |
| Controlled | $\beta$ | 0.1 unless swept ({0.05, 0.1, 0.5}) |
| Controlled | Training | 1 epoch, same pair-sampling temperature, same 3 seeds in every cell |
| Controlled | Evaluation decoding | greedy |
| Control (expected to fail) | BT-consistent label-flip noise $\eta = 0.1$ | should produce no systematic change |

**Models.** Qwen2.5-0.5B-Instruct for all sweeps; Qwen2.5-1.5B-Instruct for headline cells. Qwen3-0.6B (what Gopalan et al. use) is the preferred swap if TRL supports it cleanly at the time of the first run; decide in W1 and record in the log.

**Data.** 5k prompts (10k full) from alpaca-cleaned; **four** responses per prompt sampled from the reference model so that $\varepsilon$ is well defined (amendment 2026-10-20); all six pairs labelled by the exact $p^*$, three used for training. The annotator rewards disagree on a known axis: the minority annotator also rewards positive sentiment (SST-2 log-odds, z-scored).

**Baselines.** DPO at $\beta \in \{0.05, 0.1, 0.5\}$; length-normalised DPO; AuxDPO. The online two-timescale variant is kept only if it yields a testable stochastic-approximation prediction (bias floor as a function of the step-size ratio $b_n/a_n$); otherwise it is dropped.

**Dropped (2026-10-18).** The federated-preference ablation (E7): Distributed DPO ([arXiv 2605.20696](https://arxiv.org/abs/2605.20696)) and FedPDPO already cover federated DPO with heterogeneous users. The Prathosh subset-selection and Pruthi evaluation citations: neither paper is about preferences.

**The expected-to-fail control.** BT-consistent label flips at $\eta = 0.1$ keep the data inside the BT class. If this cell shows an effect of the size we see under $\gamma$ or $\varepsilon$, the harness is broken and nothing else is reported until it is fixed.

---

## 4. The reasoning component

We derive first and test second. The DPO loss is

$$\mathcal{L}(\theta) = -\mathbb{E}\big[\log \sigma(\beta h_\theta)\big], \qquad h_\theta = \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)},$$

with gradient

$$\nabla \mathcal{L} = -\beta\, \mathbb{E}\Big[\sigma(-\beta h_\theta)\big(\nabla \log \pi_\theta(y_w \mid x) - \nabla \log \pi_\theta(y_l \mid x)\big)\Big].$$

The weight $\sigma(-\beta h_\theta)$ is largest on pairs the model currently gets wrong. Under annotator disagreement the model keeps pushing on pairs that no BT reward can explain; that is the mechanism behind margin compression in H2.

At the population level, if the data follow BT with reward $r^*$ and the policy class is rich enough, the minimiser satisfies $\beta \log \pi_\theta / \pi_{\mathrm{ref}} = r^* + c(x)$, so $\pi_\theta \propto \pi_{\mathrm{ref}} \exp(r^*/\beta)$. Under length contamination this becomes $\pi_\theta \propto \pi_{\mathrm{ref}} \exp\big((r^* + \gamma|y|)/\beta\big)$, the quantitative prediction behind H1. Under a mixture the minimiser is the KL projection of $p^*$ onto the BT family; we compute $\varepsilon$ numerically on the synthetic data and plot win-rate loss against it.

Gopalan et al. show that when the policy class is *not* rich enough, DPO's implicit reward is the reverse-KL projection of $r^*$ onto the expressible set, and AuxDPO adds free per-response offsets so that projection becomes exact. The two projections (onto the BT family in label space; onto the expressible set in reward space) are different objects, and H3 tests that AuxDPO only moves the second one.

For the online variant we write $\theta_{n+1} = \theta_n - a_n\, g(\theta_n, \xi_n)$ under Robbins–Monro conditions $\sum a_n = \infty$, $\sum a_n^2 < \infty$, with $b_n / a_n \to 0$ in the two-timescale version so the reward model tracks its fixed point for the current policy. The ODE limit $\dot\theta = -\nabla \mathcal{L}(\theta)$ has its stable equilibrium shifted by the misspecification; under a Lipschitz gradient and a Polyak–Łojasiewicz-type condition the expected squared error decomposes as a vanishing $O(a_n)$ variance term plus a bias term of order $\varepsilon$. Whether the empirical bias term scales linearly in $\varepsilon$ is the stretch figure.

---

## 5. Compute budget (hard cap)

Sized for a 16 GB T4 with LoRA rank 16 on Qwen2.5-0.5B-Instruct in fp32. **The 2026-10-18 review found the original per-run estimates about 2× optimistic**; the numbers below are revised. A 1.5B run costs roughly 3× a 0.5B run. Kaggle's free quota is commonly reported as ~30 GPU-hours per week (not an official figure); Colab free is unpublished and interruptible.

| Experiment | Model | Runs × seeds | GPU-h (revised) | Kaggle weeks |
|---|---|---|---|---|
| E1 Pair generation (10k prompts × 2 samples) + reward scoring | 0.5B | 1 | 3 | — |
| E2 Reference log-prob cache (computed once, reused by all runs) | 0.5B | 1 | 0.5 | — |
| E3 $\gamma$ sweep: 5 $\gamma$ × 3 $\beta$ × 3 seeds | 0.5B | 45 × 0.8 h | 36 | 1.2 |
| E4 $\varepsilon$ (mixture) sweep: 5 $\varepsilon$ × 3 seeds | 0.5B | 15 × 0.8 h | 12 | 0.4 |
| E5 Length-normalised baseline + AuxDPO: 2 × 5 $\varepsilon$ × 3 seeds, plus the policy-class cell | 0.5B | 36 × 0.8 h | 29 | 1 |
| E6 Online two-timescale variant (stretch) | 0.5B | 5 × 1.5 h | 7.5 | 0.25 |
| E8 1.5B replication of headline cells | 1.5B | 10 × 2.5 h | 25 | 0.8 |
| E9 Evaluation (generation + reward scoring, every checkpoint) | 0.5B/1.5B | all runs | 12 | 0.4 |
| **Total full version** | | | **~125** | **~4** |

**Minimum viable version (~25 GPU-hours, one Kaggle week).** Keep E1 and E2 unchanged (they are the cache that makes everything else cheap). E3: 3 $\gamma$-levels × $\beta = 0.1$ × 3 seeds (9 runs). E4: 3 $\varepsilon$-levels × 3 seeds (9 runs). E5: AuxDPO in the policy-class cell and the largest-$\varepsilon$ cell only, 3 seeds (6 runs). Drop E6 and E8. Use 5k prompts, 1 epoch, 300 evaluation prompts.

*What survives at minimum scale:* H1 intact (length absorption is a large effect); H2's monotonicity and margin-compression tests, with the $R^2$ test under-powered and reported as a trend; H3's two-cell contrast. The full version is required for the $\varepsilon$-scaling $R^2$ test, the length-normalised baseline and any 1.5B claim.

**Caching rules.** Generate pairs once; store responses plus reference log-probs in one parquet file. Store LoRA adapters, never merged models. Score win-rate with one cached reward-model pass over fixed generations. Same 3 seeds in every cell so every comparison is paired.

---


## 6. How to run (Kaggle)

Every experiment is one notebook in `notebooks/` (generated by `tools/make_notebooks.py`) that clones this repo and calls `python -m src.run`. Kaggle settings: Accelerator **GPU T4 x2**, Internet **On**. Results land in `/kaggle/working/results`; save that as a Kaggle dataset and attach it to the next notebook (set `PREV` in the setup cell).

| Order | Notebook | Command it runs | GPU-h (0.5B, minimum) |
|---|---|---|---|
| 0 | `00_smoke_test` | `python -m src.run smoke` — every code path on 24 prompts | 0.1 |
| 1 | `01_E1_E2_build_cache` | `build` (K=4 samples, r*, axis, ref log-probs) then `epsilon` | 2–3 |
| 2 | `02_E0_control_eta` | `sweep --config configs/control_eta.yaml` | 2.5 |
| 3 | `03_E3_gamma_sweep` | `sweep --config configs/gamma_sweep_0p5b.yaml --minimum` | 7 |
| 4 | `04_E4_epsilon_sweep` | `sweep --config configs/epsilon_sweep_0p5b.yaml --minimum` | 7 |
| 5 | `05_E5_auxdpo` | `sweep --config configs/auxdpo_cells.yaml --minimum` | 19 |
| 6 | `06_E6_online_two_timescale` | `online --cell ...` (stretch) | 6 |
| 7 | `07_analysis_and_figures` | `analyze` — pre-registered tests + Figures 1–4 | 0 |

Locally: `pip install -r requirements.txt && python tests/test_epsilon_and_stats.py` runs the numpy-only tests (ε, annotator model, statistics, config hashing) without a GPU.

**Status of the code (2026-10-20).** All modules are written and compile; the numpy/scipy parts are unit-tested; the torch parts (generation, training, evaluation) have not yet been executed on a GPU. The smoke test is the first thing to run, and the expected-to-fail control is the second.

## 6b. Plan

The week-by-week plan with task lists lives in [`ROADMAP.md`](ROADMAP.md). Summary: W1–2 reading and rewrite of the pre-registration in your own words; W3–4 run the smoke test and the expected-to-fail control, fix what breaks; W5–7 minimum-viable sweeps; W8–10 analysis, 4-page write-up, one-page summary for Gopalan.

---

## 7. Deliverables

1. **Preprint** (4-page workshop version first, then 6–8 pages): *"DPO under misspecified preferences: a calibrated testbed with known ground truth."*
2. **This repository**, with the synthetic preference generator as a reusable package, a config file for every figure, and a one-command reproduction of the main plot.
3. **One-page summaries** for two readers only. Gopalan: the $\varepsilon$-scaling figure, the calibration result and the AuxDPO decomposition. Thoppe or Bhatnagar: the two-timescale analysis, if the stretch arm is run.

---

## 8. Research protocol

- **Pre-register before running.** Hypotheses, tests, models, seeds and date live in `PREREGISTRATION.md`, frozen before the first sweep; changes go in its Amendments appendix with a date and a reason.
- **Experiment log.** Every run in `EXPERIMENT_LOG.md` records config hash, git commit, expectation, observation and next change; it feeds the negative-results section.
- **Seeds.** Fixed for data sampling, LoRA init, dropout and generation; at least three seeds (0, 1, 2) behind any number in a figure.
- **Uncertainty.** Mean plus 95% bootstrap CI, never a single run; paired comparisons over the same prompts.
- **Negative results.** The preprint states which hypotheses were refuted and what we now believe instead; a refuted hypothesis with a mechanism beats a confirmed one without.
- **Expected-to-fail control.** The $\eta = 0.1$ cell must show no effect, or the harness is debugged before anything else is reported.
- **Figures** are frozen only from runs reproducible with one command. A 0.5B result is reported as a 0.5B result.
- **"First" claims** are made only after a Semantic Scholar sweep recorded in `REVIEW.md`.

---

## 9. Repository layout

```
dpo-misspecification-study/
├── README.md
├── PREREGISTRATION.md            # frozen 2026-10-13; amendments appendix
├── REVIEW.md                     # independent review, 2026-10-18
├── ROADMAP.md                    # milestones as task lists (progress tracker)
├── EXPERIMENT_LOG.md             # append-only run log
├── READING.md                    # ordered reading list
├── requirements.txt
├── configs/                      # one YAML per experiment (_base_0p5b.yaml is copied into each)
├── src/                          # common/, prefgen/, train/, eval/, analysis/, run.py, smoke.py (see src/README.md)
├── tests/                        # numpy-only tests: epsilon, annotator model, statistics
├── tools/make_notebooks.py       # regenerates notebooks/
├── notebooks/                    # one Kaggle notebook per experiment, 00_smoke_test first
├── results/                      # parquet caches, LoRA adapters, figure CSVs (git-ignored)
└── paper/                        # preprint source, one-pagers/
```

---

## 10. Related work to read first

See [`READING.md`](READING.md) for the ordered list with reasons. The must-reads: Rafailov et al. (DPO), Azar et al. (IPO), Park et al. (R-DPO), Siththaranjan et al. (hidden context), Chakraborty et al. (MaxMin-RLHF), Ray Chowdhury et al. (Provably Robust DPO), Razin et al. (likelihood displacement), Gopalan et al. (AuxDPO), Borkar (ch. 1–2, 6).

---

## 11. Faculty this work is aimed at

| Faculty | Dept | Paper this extends | Role |
|---|---|---|---|
| Aditya Gopalan | ECE | *Why DPO is a Misspecified Estimator and How to Fix It* (ICLR 2026 oral) | Primary reader |
| Gugan Thoppe / Shalabh Bhatnagar | CSA | Stochastic approximation, finite-time rates | Second reader, only if the online arm is run |

---

## 12. License and citation

Code: MIT. Data and figures: CC BY 4.0.

```bibtex
@misc{dpo-misspecification-2026,
  title  = {DPO under misspecified preferences: a calibrated testbed with known ground truth},
  author = {Arush Badhe},
  year   = {2026},
  note   = {Pre-registered 2026-10-13, amended 2026-10-18. https://github.com/Arubad/dpo-misspecification-study}
}
```
