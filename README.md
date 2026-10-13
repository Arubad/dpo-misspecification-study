# When preferences lie: DPO under Bradley–Terry misspecification, analysed as stochastic approximation

**Status:** Pre-registration · Week 0 of 5 · Pre-registered 2026-10-13 · Models: Qwen2.5-0.5B-Instruct (primary), Qwen2.5-1.5B-Instruct (replication) · Seeds: 0, 1, 2

**Problem in one sentence.** We measure how, and explain why, DPO's learned policy degrades when the preference data violate the Bradley–Terry (BT) assumption that DPO encodes.

**Why this matters.** DPO is the default way to align small open models, and it assumes the annotator's preferences come from a single Bradley–Terry reward. Real preference data do not: annotators disagree, and they favour longer answers. There is no controlled study on a small open LLM where the ground-truth reward is known exactly, the misspecification is dialled in by hand, and the degradation is compared against what stochastic-approximation theory predicts. Without one we cannot say whether DPO fails gracefully or catastrophically, or whether proposed fixes recover what was lost.

---

## 1. Research proposition

### The gap

Three strands of work at IISc point at this question from different directions, and none of them closes it.

The **estimation-theoretic strand** is Aditya Gopalan's. His ICLR 2026 oral, *"Why DPO is a Misspecified Estimator and How to Fix It"*, argues that DPO is misspecified as an estimator of the reward and proposes a correction; his AISTATS 2026 paper *"Bad Values but Good Behavior"* shows that highly misspecified bandits with function approximation can still act well even when their value estimates are badly wrong ([Gopalan research page](https://ece.iisc.ac.in/~aditya/research.html); [Gopalan homepage](https://ece.iisc.ac.in/~aditya/)). Put together, these raise an uncomfortable empirical question: when DPO's implicit reward is wrong, is the *policy* also wrong, or does it keep behaving well regardless?

The **stochastic-approximation strand** supplies the analytical tools but has not been applied to preference optimisation. Gugan Thoppe gives parameter-free, finite-time rates for TD and Q-learning (AAAI 2025), studies federated TD and policy gradient under heterogeneity and adversaries (CDC 2024; TMLR 2024), and with Gopalan asks "Does DQN Learn?" (IEEE TAC 2025) ([Thoppe research page](https://sites.google.com/site/gugancth/research)). Shalabh Bhatnagar's stochastic-approximation lineage (off-policy actor-critic, ICML 2023; policy Newton, AISTATS 2023; SPSA) and Anant Raj's mean-field and mirror-descent optimisation (COLT 2025; policy gradient, NeurIPS 2024 and RLC 2024) give the ODE and two-timescale machinery ([Bhatnagar homepage](https://www.csa.iisc.ac.in/~shalabh/); [Raj site](https://anantrajml.github.io/)).

The **applied trustworthy-fine-tuning strand** asks what to do with alignment data of uncertain quality. Prathosh A.P.'s ICLR 2026 paper *"Enhancing Trustworthiness of Fine-Tuned LLMs via Regularized Subset Selection"* asks which fine-tuning data to trust ([prathosh.in/publications](https://prathosh.in/publications.html)); Danish Pruthi's evaluation line (abstention, NAACL 2025; presupposition-laden queries for reasoning models, ACL 2026 Findings) asks how to measure whether an aligned model behaves well ([Pruthi publications](https://danishpruthi.com/publications/)).

Nobody at IISc has published a controlled empirical study that connects these three: what actually happens to a small open LLM when the preference data violate the BT assumption, and does the degradation follow the shape a projected-fixed-point analysis predicts?

### The research question

Under what kinds and magnitudes of preference misspecification does DPO's learned policy degrade, and is the degradation **graceful** (bias proportional to the misspecification level, as a projected-fixed-point analysis predicts) or **catastrophic** (reward hacking such as verbosity growth that compounds with training)?

### Why synthetic preferences, and why a sub-2B model

Because the ground-truth reward $r^*$ must be known exactly to measure anything. Our generator scores $r^*$ with a small reward model and injects nuisance terms with known coefficients: a length coefficient $\gamma$, a mixture over $K$ annotator rewards with weights $w_k$, and BT-consistent label-flip noise $\eta$ as the control that should not matter. A 0.5B model with LoRA fits a single T4 at under 2 GPU-hours per run; a 1.5B replication checks that the shape is not a small-model artefact.

---

## 2. Pre-registered hypotheses

Full text with dates and seeds is in [`HYPOTHESES.md`](HYPOTHESES.md). The numerical tests are copied verbatim here.

**H1 — length is absorbed into the implicit reward.** If the annotator model is $p^*(y_w \succ y_l \mid x) = \sigma\big(r^*(y_w) - r^*(y_l) + \gamma(|y_w| - |y_l|)\big)$, the BT class still contains the data-generating reward $r^* + \gamma|y|$, so DPO converges to a policy that pays exactly $\gamma/\beta$ nats per extra token.
*Confirmed* if mean response length at $\gamma = 0.3$ exceeds $\gamma = 0$ by more than 25% in at least 4 of 5 ($\gamma$-level × seed) comparisons and KL-to-reference rises more than 2× over the same range. *Refuted* if the length change is under 5% or non-monotone (Spearman $\rho < 0.6$).

**H2 — mixture misspecification degrades gracefully, then breaks.** If $p^* = \sum_k w_k\, \sigma\big(r_k(y_w) - r_k(y_l)\big)$, no single BT reward reproduces $p^*$; DPO converges to the BT information projection, implicit-reward margins compress, and win-rate loss against the majority annotator grows approximately linearly in $\varepsilon = \min_r \mathbb{E}_x\, \mathrm{KL}\big(p^*(\cdot \mid x)\,\|\,\mathrm{BT}_r(\cdot \mid x)\big)$ for small $\varepsilon$, then nonlinearly.
*Confirmed* if win-rate loss is monotone in $\varepsilon$ with a linear fit $R^2 > 0.8$ for $\varepsilon \le 0.1$ and the implicit-reward margin IQR shrinks by more than 30% at the largest $\varepsilon$. *Refuted* if win-rate loss at the largest $\varepsilon$ is within the 95% CI of zero.

**H3 — the fix works.** The ICLR 2026 correction recovers most of the lost win-rate in the mixture setting at the cost of higher variance, which should appear as a larger constant in the finite-time error bound.
*Confirmed* if the correction recovers at least 50% of the win-rate gap at the largest $\varepsilon$ in at least 2 of 3 seeds. *Refuted* if recovery is under 20% or the fix loses more than 3 win-rate points at $\varepsilon = 0$.

---

## 3. Experimental design

| Role | Variable | Levels |
|---|---|---|
| Independent | Length-contamination coefficient $\gamma$ | {0, 0.05, 0.1, 0.2, 0.3} |
| Independent | Annotator-mixture misspecification $\varepsilon$ | $K$ = 2–3 reward models; 5 mixture weights from 0 to 0.5 minority mass |
| Independent | Algorithm | DPO, length-normalised DPO, ICLR-2026 fix, online two-timescale DPO |
| Dependent | Win-rate vs true majority reward | 500 held-out prompts |
| Dependent | Response length | mean and 90th percentile |
| Dependent | $\mathrm{KL}(\pi \,\|\, \pi_{\mathrm{ref}})$ | per prompt, averaged |
| Dependent | Implicit-reward margin distribution | IQR |
| Dependent | Calibration of implicit vs true reward | Spearman $\rho$ |
| Dependent | Convergence curve $\|\theta_n - \theta_T\|$ | checkpoints along one long run |
| Controlled | Base model | Qwen2.5-0.5B-Instruct, LoRA rank 16 |
| Controlled | Prompt set | 10k fixed prompts from a public instruction set |
| Controlled | $\beta$ | 0.1 unless swept ({0.05, 0.1, 0.5}) |
| Controlled | Training | 1 epoch, same pair-sampling temperature, same 3 seeds in every cell |
| Controlled | Evaluation decoding | greedy |
| Control (expected to fail) | BT-consistent label-flip noise $\eta = 0.1$ | should produce no systematic change |

**Models.** Qwen2.5-0.5B-Instruct for all sweeps; Qwen2.5-1.5B-Instruct for headline cells (optionally Llama-3.2-1B/3B). LoRA rank 16 throughout.

**Data.** 10k–20k prompts from a public instruction set; two responses per prompt sampled from the reference model; pairs labelled by the chosen $p^*$. The $K$ annotator rewards disagree on a known axis (formality versus brevity).

**Baselines.** DPO at $\beta \in \{0.05, 0.1, 0.5\}$; length-normalised DPO; the ICLR 2026 fix; an online two-timescale variant (reward model on the fast timescale, policy on the slow one) that makes the stochastic-approximation analysis literal.

**Ablations.** $\gamma$, mixture entropy, dataset size, $\beta$, and a federated-preference variant in which each of $C$ clients holds one annotator's data and the server aggregates DPO updates (the bridge to Thoppe's federated TD/PG work and Chakraborty's NeurIPS 2026 rational-client aggregation).

**The expected-to-fail control.** BT-consistent label flips at $\eta = 0.1$ keep the data inside the BT class. If this cell shows an effect of the size we see under $\gamma$ or $\varepsilon$, the harness is broken and nothing else is reported until it is fixed.

---

## 4. The reasoning component

We derive first and test second. The DPO loss is

$$\mathcal{L}(\theta) = -\mathbb{E}\big[\log \sigma(\beta h_\theta)\big], \qquad h_\theta = \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)},$$

with gradient

$$\nabla \mathcal{L} = -\beta\, \mathbb{E}\Big[\sigma(-\beta h_\theta)\big(\nabla \log \pi_\theta(y_w \mid x) - \nabla \log \pi_\theta(y_l \mid x)\big)\Big].$$

The weight $\sigma(-\beta h_\theta)$ is largest on pairs the model currently gets wrong. Under annotator disagreement the model keeps pushing on pairs that no BT reward can explain; that is the mechanism behind margin compression in H2.

At the population level, if the data follow BT with reward $r^*$, the minimiser satisfies $\beta \log \pi_\theta / \pi_{\mathrm{ref}} = r^* + c(x)$, so $\pi_\theta \propto \pi_{\mathrm{ref}} \exp(r^*/\beta)$. Under length contamination this becomes $\pi_\theta \propto \pi_{\mathrm{ref}} \exp\big((r^* + \gamma|y|)/\beta\big)$, the quantitative prediction behind H1. Under a mixture the minimiser is the KL projection of $p^*$ onto the BT family; we compute $\varepsilon$ numerically on the synthetic data and plot win-rate loss against it.

For the online variant we write $\theta_{n+1} = \theta_n - a_n\, g(\theta_n, \xi_n)$ under Robbins–Monro conditions $\sum a_n = \infty$, $\sum a_n^2 < \infty$, with $b_n / a_n \to 0$ in the two-timescale version so the reward model tracks its fixed point for the current policy. The ODE limit $\dot\theta = -\nabla \mathcal{L}(\theta)$ has its stable equilibrium shifted by the misspecification; under a Lipschitz gradient and a Polyak–Łojasiewicz-type condition the expected squared error decomposes as a vanishing $O(a_n)$ variance term plus a bias term of order $\varepsilon$. Whether the empirical bias term scales linearly in $\varepsilon$ is the paper's central figure: Thoppe's parameter-free, finite-time question for TD learning, transplanted to preference optimisation.

---

## 5. Compute budget (hard cap)

Sized for a 16 GB T4 with LoRA rank 16 on Qwen2.5-0.5B-Instruct; a 1.5B run costs roughly 3× a 0.5B run. Cost columns assume Colab Pro at about US$0.20 per T4-hour, Kaggle's free quota of about 30 GPU-hours per week, a rented 3090/4090 at US$0.30–0.50 per hour, and Lightning AI's free monthly credit.

| Experiment | Model | Runs × seeds | GPU-h (full) | Colab Pro | Kaggle free | Rented 3090/4090 | Lightning free |
|---|---|---|---|---|---|---|---|
| E1 Pair generation (10k prompts × 2 samples) + reward scoring | 0.5B | 1 | 2.5 | ~$0.50 | $0 | ~$1 | within quota |
| E2 Reference log-prob cache (computed once, reused by all runs) | 0.5B | 1 | 0.3 | ~$0.06 | $0 | ~$0.15 | within quota |
| E3 $\gamma$ sweep: 5 $\gamma$ × 3 $\beta$ × 3 seeds | 0.5B | 45 × 0.4 h | 18 | ~$3.60 | $0 | ~$7 | partial |
| E4 $\varepsilon$ (mixture) sweep: 5 $\varepsilon$ × 3 seeds | 0.5B | 15 × 0.4 h | 6 | ~$1.20 | $0 | ~$2.50 | within quota |
| E5 Length-normalised baseline + ICLR-2026 fix: 2 × 5 $\varepsilon$ × 3 seeds | 0.5B | 30 × 0.4 h | 12 | ~$2.40 | $0 | ~$5 | partial |
| E6 Online two-timescale variant | 0.5B | 5 × 1 h | 5 | ~$1 | $0 | ~$2 | within quota |
| E7 Federated-preference ablation: 3 configs × 3 seeds | 0.5B | 9 × 0.5 h | 4.5 | ~$0.90 | $0 | ~$2 | within quota |
| E8 1.5B replication of headline cells | 1.5B | 10 × 1.5 h | 15 | ~$3 | $0 (slow) | ~$6 | no |
| E9 Evaluation (win-rate, KL, calibration) | 0.5B/1.5B | all runs | 3 | ~$0.60 | $0 | ~$1.20 | within quota |
| **Total full version** | | | **~66** | **~$13** | **$0 over ~3 weeks** | **~$27** | — |

**Minimum viable version (~15 GPU-hours, ≤US$3 on Colab Pro or free on Kaggle).** Keep E1 and E2 unchanged (they are the cache that makes everything else cheap). Shrink E3 to 3 $\gamma$-levels × $\beta = 0.1$ × 3 seeds (9 runs, 2.7 h); E4 to 3 $\varepsilon$-levels × 3 seeds (9 runs, 2.7 h); E5 to the length-normalised baseline only, 3 $\varepsilon$-levels × 2 seeds (6 runs, 1.8 h); E6 to 2 runs (2 h). Drop E7 and E8. Use 5k prompts, 1 epoch, 300 evaluation prompts. SmolLM-360M or Pythia-410M halves every row again if needed.

*What survives at minimum scale:* H1 intact (length absorption is a large effect); H2's monotonicity and margin-compression tests, with the $R^2$ test under-powered and reported as a trend; H3 only via the length-normalised baseline; the convergence-curve fit, which needs one long checkpointed run. The full version is required for the $\varepsilon$-scaling $R^2$ test, the ICLR-2026 fix, the federated ablation and any 1.5B claim.

**Caching rules.** Generate pairs once; store responses plus reference log-probs in one parquet file. Store LoRA adapters, never merged models. Score win-rate with one cached reward-model pass over fixed generations. Same 3 seeds in every cell so every comparison is paired.

---

## 6. Week-by-week plan

- [ ] **W1 — Preference generator + evaluation harness.** Artifact: notebook showing 20 labelled pairs and the ground-truth reward histogram; this pre-registered README committed.
- [ ] **W2 — $\gamma$ sweep on 0.5B.** Artifact: length-vs-$\gamma/\beta$ plot with CIs and the KL plot.
- [ ] **W3 — $\varepsilon$ sweep.** Artifact: win-rate-vs-$\varepsilon$ figure and the margin-IQR table.
- [ ] **W4 — Fix, length-normalised baseline, online variant.** Artifact: recovery table and the convergence-curve fit ($O(1/n)$ + bias).
- [ ] **W5 — 1.5B replication (full budget only), negative-results section, preprint v0.** Artifact: the paper's Figure 1 and a one-command reproduction.

---

## 7. Expected results and how to read either outcome

The most likely outcome is that H1 holds cleanly ($\gamma/\beta$ explains most of the variance in mean length), H2 holds for small $\varepsilon$ with a knee where the mixture becomes bimodal, and H3 holds partially.

If instead win-rate loss is **flat** in $\varepsilon$ while implicit-reward calibration collapses, that is a direct empirical instance of "bad values but good behaviour" and is just as publishable: DPO's policy would be robust to misspecification even when its implicit reward is useless as a reward model, which matters to anyone reusing DPO implicit rewards for data selection (Prathosh's subset-selection line).

If the federated variant shows the server policy drifting to a minority annotator, the explanation is the non-convexity of the aggregated objective; we test it by checking whether weighting clients by margin variance fixes the drift.

---

## 8. Deliverables

1. **Preprint** (6–8 pages, arXiv style): *"DPO under misspecified preferences: an empirical stochastic-approximation study on sub-2B models."*
2. **This repository**, with the synthetic preference generator as a reusable package, a config file for every figure, and a one-command reproduction of the main plot.
3. **Per-professor one-pagers.** Gopalan: the $\varepsilon$-scaling figure and the calibration result. Thoppe and Bhatnagar: the two-timescale analysis and step-size ablation. Raj: the mirror-descent view of the KL-regularised objective. Prathosh: the implicit-reward-as-data-selector finding. Pruthi: the evaluation harness and verbosity audit.

---

## 9. Research protocol

- **Pre-register before running.** Hypotheses, tests, models, seeds and date live in `HYPOTHESES.md`, committed before the first sweep.
- **Experiment log.** Every run in `logs/EXPERIMENT_LOG.md` records config hash, git commit, expectation, observation and next change; it feeds the negative-results section.
- **Seeds.** Fixed for data sampling, LoRA init, dropout and generation; at least three seeds (0, 1, 2) behind any number in a figure.
- **Uncertainty.** Mean plus 95% bootstrap CI, never a single run; paired comparisons over the same prompts.
- **Negative results.** The preprint states which hypotheses were refuted and what we now believe instead; a refuted hypothesis with a mechanism beats a confirmed one without.
- **Expected-to-fail control.** The $\eta = 0.1$ cell must show no effect, or the harness is debugged before anything else is reported.
- **Figures** are frozen only from runs reproducible with one command. A 0.5B result is reported as a 0.5B result.

---

## 10. Repository layout (planned)

```
dpo-misspecification-study/
├── README.md
├── HYPOTHESES.md                 # pre-registration, frozen 2026-10-13
├── configs/                      # one YAML per figure cell (gamma/, epsilon/, fix/, online/, federated/)
├── src/
│   ├── prefgen/                  # synthetic preference generator (reusable package)
│   ├── train/                    # DPO, length-normalised DPO, ICLR-2026 fix, two-timescale online DPO
│   ├── eval/                     # win-rate, length, KL, margin IQR, calibration, convergence curve
│   └── analysis/                 # epsilon computation, O(1/n)+bias fit, bootstrap CIs
├── notebooks/                    # W1 pair inspection, per-week artifacts
├── results/                      # parquet caches, LoRA adapters, figure CSVs
├── logs/EXPERIMENT_LOG.md
└── paper/                        # preprint source, one-pagers/
```

---

## 11. Related work to read first

**IISc papers this study extends** (read in this order)

- A. Gopalan et al., *Why DPO is a Misspecified Estimator and How to Fix It*, ICLR 2026 (oral); *Bad Values but Good Behavior*, AISTATS 2026. [Gopalan research page](https://ece.iisc.ac.in/~aditya/research.html)
- G. Thoppe et al., parameter-free finite-time rates for Q-learning, AAAI 2025; federated TD, CDC 2024; federated policy gradient with adversaries, TMLR 2024; *Does DQN Learn?* (with Gopalan), IEEE TAC 2025. [Thoppe research page](https://sites.google.com/site/gugancth/research)
- S. Bhatnagar et al., off-policy average-reward actor-critic, ICML 2023; cubic-regularised policy Newton, AISTATS 2023. [Bhatnagar homepage](https://www.csa.iisc.ac.in/~shalabh/)
- A. Raj et al., policy gradient, NeurIPS 2024 and RLC 2024. [Raj site](https://anantrajml.github.io/)
- Prathosh A.P. et al., *Enhancing Trustworthiness of Fine-Tuned LLMs via Regularized Subset Selection*, ICLR 2026. [prathosh.in/publications](https://prathosh.in/publications.html)
- D. Pruthi et al., abstention evaluation, NAACL 2025; presupposition evaluation, ACL 2026 Findings. [Pruthi publications](https://danishpruthi.com/publications/)

**Canonical external papers**

- R. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023, arXiv:2305.18290.
- P. Singhal et al., *A Long Way to Go: Investigating Length Correlations in RLHF*, 2023, arXiv:2310.03716.
- R. Park et al., *Disentangling Length from Quality in Direct Preference Optimization*, 2024, arXiv:2403.19159.
- M. G. Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences* (IPO), 2023, arXiv:2310.12036.
- V. S. Borkar, *Stochastic Approximation: A Dynamical Systems Viewpoint*, 2008 (book) — the ODE method and two-timescale analysis we use.

---

## 12. Faculty this work is aimed at

| Faculty | Dept | Paper this extends |
|---|---|---|
| Aditya Gopalan | ECE | *Why DPO is a Misspecified Estimator and How to Fix It* (ICLR 2026); *Bad Values but Good Behavior* (AISTATS 2026) |
| Gugan Thoppe | CSA | Parameter-free Q-learning rates (AAAI 2025); federated TD (CDC 2024); federated PG (TMLR 2024) |
| Shalabh Bhatnagar | CSA | Off-policy actor-critic (ICML 2023); policy Newton (AISTATS 2023) |
| Anant Raj | CSA | Policy gradient (NeurIPS 2024, RLC 2024); mirror-descent variational principles |
| Prathosh A.P. | ECE | Regularized subset selection for trustworthy fine-tuning (ICLR 2026) |
| Danish Pruthi | CDS | Abstention evaluation (NAACL 2025); presupposition evaluation (ACL 2026 Findings) |
| Anirban Chakraborty | CDS | FL aggregation with rational clients (NeurIPS 2026), via the federated ablation |
| Siddharth Barman | CSA | Limited-adaptivity GLM bandits (NeurIPS 2024), via the online variant |

---

## 13. License and citation

Code: MIT. Data and figures: CC BY 4.0.

```bibtex
@misc{dpo-misspecification-2026,
  title  = {DPO under misspecified preferences: an empirical stochastic-approximation study on sub-2B models},
  author = {<Your Name>},
  year   = {2026},
  note   = {Pre-registered 2026-10-13. https://github.com/Arubad/dpo-misspecification-study}
}
```
