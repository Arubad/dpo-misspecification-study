# Reading list

Ordered so that each item uses the one before it. For each paper write a half-page note in `paper/notes/`: the claim, the one experiment that supports it, and what would have refuted it. The notes become the related-work section.

## Foundations (shared across the three projects)

| Order | Item | Why it matters |
|---|---|---|
| 1 | Keshav, [How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) | The three-pass method; use it on everything below |
| 2 | Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Attention, KV cache and the softmax, before any project makes sense |
| 3 | Lones, [How to avoid machine learning pitfalls](https://arxiv.org/abs/2108.02497) | The checklist the pre-registration template implements |
| 4 | Bouthillier et al., [Accounting for Variance in ML Benchmarks](https://arxiv.org/abs/2103.03098) | Why seeds, bootstrap CIs and paired comparisons are non-negotiable |
| 5 | Sutton and Barto, [Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html), ch. 1–3, 9, 13 | Policy gradient and function approximation; the language the IISc theory groups speak |
| 6 | Peyton Jones, [How to write a great research paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/) | Read before writing the 4-page version |

## This project

| Order | Item | Why it matters |
|---|---|---|
| 1 | Christiano et al., [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | Where the Bradley–Terry assumption enters RLHF |
| 2 | Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155) | The RLHF pipeline DPO replaces |
| 3 | Rafailov et al., [DPO](https://arxiv.org/abs/2305.18290) | Derive the closed-form optimum yourself (Section 4); H1 is a corollary of it |
| 4 | Azar et al., [IPO](https://arxiv.org/abs/2310.12036) | The first paper to say DPO's BT assumption is the weak point |
| 5 | Singhal et al., [A Long Way to Go: length in RLHF](https://arxiv.org/abs/2310.03716); Park et al., [R-DPO](https://arxiv.org/abs/2403.19159) | The length arm's prior work and baseline |
| 6 | Siththaranjan et al., [Distributional Preference Learning: Hidden Context in RLHF](https://arxiv.org/abs/2312.08358) | DPO under annotator mixtures implicitly aggregates by Borda count; H2 is a dose–response version of this |
| 7 | Chakraborty et al., [MaxMin-RLHF](https://arxiv.org/abs/2402.08925) | Impossibility of one reward for diverse preferences; the mixture arm's main competitor |
| 8 | Ray Chowdhury et al., [Provably Robust DPO](https://proceedings.mlr.press/v235/ray-chowdhury24a.html) | The flip-noise control, with theory |
| 9 | Razin et al., [Unintentional Unalignment / likelihood displacement](https://arxiv.org/abs/2410.08847) | The other way DPO fails; must be ruled out as the cause of the effects |
| 10 | Gopalan, Chowdhury, Banerjee, [Why DPO is a Misspecified Estimator and How to Fix It](https://arxiv.org/abs/2510.20413) | Read Prop. 1, Prop. 3 and the AuxDPO definition closely; the paper the original H3 misread |
| 11 | Borkar, *Stochastic Approximation: A Dynamical Systems Viewpoint*, ch. 1–2 and 6 | The ODE method and two timescales; only needed for the stretch arm, but what Thoppe and Bhatnagar will ask about |

## Related, for the related-work section

CoDPO ([NeurIPS 2026](https://neurips.cc/virtual/2026/poster/151343)); EM-DPO ([arXiv 2510.15716](https://arxiv.org/abs/2510.15716)); MixDPO ([arXiv 2601.06180](https://arxiv.org/abs/2601.06180)); Shen et al. ([arXiv 2408.03459](https://arxiv.org/abs/2408.03459)); Chen et al. on DPO ranking accuracy ([NeurIPS 2024](https://arxiv.org/abs/2405.19534)); Rafailov et al. on overoptimisation ([NeurIPS 2024](https://arxiv.org/abs/2406.02900)); Im and Li on learning dynamics ([ICML 2024](https://arxiv.org/abs/2403.18742)); Sun et al., Rethinking BT ([arXiv 2411.04991](https://arxiv.org/abs/2411.04991)); SimPO; Distributed DPO ([arXiv 2605.20696](https://arxiv.org/abs/2605.20696)).
