# src/ — planned modules

- `prefgen/` — synthetic preference generator: prompts, paired sampling from the reference model, exact ground-truth reward r*, nuisance terms (γ length contamination, K-annotator mixtures, BT-consistent label flips η); writes the single parquet cache with reference log-probs (E1 + E2).
- `train/` — DPO trainer on LoRA with per-step checkpoints; variants `dpo`, `dpo_lennorm`, `auxdpo` (Gopalan et al. ICLR 2026: per-response auxiliary offsets with null-space penalty), and `dpo_two_timescale` (stretch; fast reward model, slow policy). The federated variant was dropped on 2026-10-18 (see PREREGISTRATION.md amendments).
- `eval/` — win-rate against the true majority reward on held-out prompts, mean/p90 length, KL(π‖π_ref), implicit-reward margin IQR, Spearman calibration; paired bootstrap CIs.
- `analysis/` — numerical ε = min_r E_x KL(p* ‖ BT_r), convergence-curve ‖θ_n − θ_T‖ fits of C·a_n + b(ε), all paper figures.
- `utils/` — config hashing, seeding (data, LoRA init, dropout, generation), W&B wrappers.
- `reproduce.py` — `python -m src.reproduce --figure 1` regenerates the headline figure from cached results.
