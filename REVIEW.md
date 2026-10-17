# Review of the pre-registration (2026-10-18)

Independent review of the 2026-10-13 pre-registration, done before any experiment was run. Literature and author pages were opened on 17 October 2026. Every claim below links to the page it came from. The changes this review caused are logged in the Amendments appendix of [`PREREGISTRATION.md`](PREREGISTRATION.md).

**Verdict.** The best of the three projects for learning how theory meets experiment, and the one most aligned with a specific IISc professor (Aditya Gopalan), but the original H3 was built on a misreading of his paper and has been rewritten. Novelty is moderate: each arm (length, mixture, noise) has prior work, but nobody has put all three in one calibrated testbed with a known ground-truth reward.

## 1. Citation check

| Pre-registration said | Reality | Source |
|---|---|---|
| Gopalan, *Why DPO is a Misspecified Estimator and How to Fix It*, ICLR 2026 oral | Real, and an oral. **But the misspecification studied is of the policy class, not of the Bradley–Terry model.** DPO is shown to be a reverse-KL projection of r* onto the rewards the policy class can express (Prop. 1); when r* is outside that set, preference order can reverse (Prop. 3). The fix, AuxDPO, adds a free offset per response in the dataset with a soft null-space penalty. Experiments: Llama-3.1-8B, Llama-3.2-1B, Qwen3-0.6B on UltraFeedback. Length bias is not analysed. | [ICLR page](https://iclr.cc/virtual/2026/poster/10008563), [arXiv 2510.20413](https://arxiv.org/abs/2510.20413), [Gopalan research page](https://ece.iisc.ac.in/~aditya/research.html) |
| *Bad Values but Good Behavior*, AISTATS 2026 | Real; ε-greedy and LinUCB bandits, not DPO; motivation only | [PMLR v300](https://proceedings.mlr.press/v300/banerjee26b.html) |
| Gopalan and Thoppe, *Does DQN Learn?*, IEEE TAC 2025 | Listed as TAC 2025 on Gopalan's page; arXiv shows no journal reference; treat as probable | [arXiv 2205.13617](https://arxiv.org/abs/2205.13617) |
| Thoppe, parameter-free finite-time rates, AAAI 2025 | **Could not verify** an AAAI 2025 paper; related work is on arXiv without a venue. Cite the arXiv versions or check his CV. | [arXiv 2508.05984](https://arxiv.org/abs/2508.05984), [arXiv 2510.07436](https://arxiv.org/abs/2510.07436) |
| Prathosh, *Regularized Subset Selection*, ICLR 2026 | Real (poster), but about SFT data repair via DPP subset selection, not preferences. **Removed.** | [ICLR page](https://iclr.cc/virtual/2026/poster/10008494) |
| Pruthi, abstention (NAACL 2025), presupposition (ACL 2026 Findings) | Not about preference learning. **Removed.** | [Pruthi publications](https://danishpruthi.com/publications/) |

## 2. The H3 problem (fixed)

AuxDPO assumes the labels *are* Bradley–Terry over some latent r*; what it repairs is the policy's inability to express r*. Under length contamination or an annotator mixture, the BT-fitted target itself is biased, so AuxDPO faithfully learns the biased target. "The fix recovers win-rate" was therefore most likely false by construction, and claiming it to Gopalan would have signalled that the paper had not been read.

The corrected H3 is a decomposition: AuxDPO should close the gap caused by policy-class misspecification (a new cell with a deliberately too-small policy) and leave the gap caused by preference-model misspecification untouched. A predicted null result that separates two kinds of misspecification is a clean contribution and exactly the conversation Gopalan would want to have.

## 3. Novelty against the literature

- **Length arm.** [R-DPO (Park et al. 2024)](https://arxiv.org/abs/2403.19159) shows DPO exploits length and regularises it; [CoDPO (NeurIPS 2026)](https://neurips.cc/virtual/2026/poster/151343) defines a "constraint drift" diagnostic for pairwise labels that conflict with pointwise criteria such as length; [down-sampled-KL DPO](https://arxiv.org/pdf/2406.10957) is another length fix. H1 is a quantitative calibration of a known effect; the γ/β-nats-per-token prediction is a closed-form corollary of DPO's optimum, so it is a sanity check, not a finding. With LoRA and one epoch the population optimum is not reached; test the slope qualitatively.
- **Mixture arm.** [MaxMin-RLHF (ICML 2024)](https://arxiv.org/abs/2402.08925) proves a single reward cannot represent diverse preferences and fits an EM mixture; [EM-DPO](https://arxiv.org/abs/2510.15716) shows pairwise data cannot identify latent annotator types; [MixDPO (2026)](https://arxiv.org/abs/2601.06180) models preference-strength variation; [Siththaranjan et al.](https://arxiv.org/abs/2312.08358) show DPO under hidden context aggregates by Borda count; [Shen et al.](https://arxiv.org/abs/2408.03459) give a generalisation bound. None sweeps a KL-projection distance ε against win-rate on a known reward, so the dose–response curve of H2 is open.
- **Noise control.** [Provably Robust DPO (ICML 2024)](https://proceedings.mlr.press/v235/ray-chowdhury24a.html) studies DPO under flip noise; it confirms the control should behave as expected.
- **Other DPO failure modes to rule out.** [Razin et al., likelihood displacement (ICLR 2025)](https://arxiv.org/abs/2410.08847); [Chen et al., DPO ranking accuracy (NeurIPS 2024)](https://arxiv.org/abs/2405.19534); [Rafailov et al., overoptimisation in direct alignment (NeurIPS 2024)](https://arxiv.org/abs/2406.02900); [Im and Li, learning dynamics (ICML 2024)](https://arxiv.org/abs/2403.18742); [Sun et al., Rethinking BT](https://arxiv.org/abs/2411.04991); [general preference models](https://arxiv.org/abs/2410.02197).
- **Federated arm (dropped).** [Distributed DPO (2026)](https://arxiv.org/abs/2605.20696) and FedPDPO ([arXiv 2603.19741](https://arxiv.org/abs/2603.19741)) already cover convergence of federated DPO with heterogeneous users.
- **Stochastic-approximation framing.** No DPO-specific SA analysis turned up in search (a negative result, not proof). Worth keeping only if it yields a testable prediction.
- **Combination.** No prior paper combines known reward, swept γ, K-annotator mixture, BT-consistent control and an ε distance in one testbed. That combination is the contribution. **Run a Semantic Scholar sweep and record it here before writing "first".**

## 4. Feasibility on free Kaggle/Colab

Directionally right, about 2× optimistic. 10k pairs at ~300 tokens is ~6M tokens through policy forward+backward plus the reference pass: ~0.3–0.5 h at T4 peak, 0.5–1.5 h with TRL/HF overhead, gradient checkpointing and AMP. Win-rate evaluation needs generation plus reward scoring per checkpoint and can cost as much as training. T4 has no bf16, and Qwen in fp16 can NaN; use fp32 for the 0.5B model. The minimum version (~25 h) is one Kaggle week (quota commonly reported as ~30 h/week, not official: [gpuperhour, Sep 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)); the full version is about four.

## 5. What changed (summary; details in the Amendments appendix)

1. H3 rewritten as the AuxDPO decomposition; policy-class-capacity cell added.
2. Likelihood displacement added as a dependent variable.
3. H1 reframed as a calibration check.
4. Federated ablation dropped; online two-timescale arm demoted to stretch.
5. fp32 on T4; GPU-hour estimates doubled; evaluation budget quadrupled.
6. Citations to Prathosh and Pruthi removed; Thoppe AAAI 2025 flagged as unverified.
7. Target readers reduced from eight faculty to one primary (Gopalan) and one secondary (Thoppe or Bhatnagar).

## 6. Open items

- [ ] Semantic Scholar sweep for "synthetic preferences known reward DPO misspecification" before any "first" claim.
- [ ] Decide in W1 between Qwen2.5-0.5B-Instruct and Qwen3-0.6B (Gopalan et al.'s small model); record in the log.
- [ ] Check whether AuxDPO code is public; if not, implement from the paper's definition and validate on their Qwen3-0.6B setting.
- [ ] Verify the Thoppe AAAI 2025 citation or replace it with the arXiv versions.
