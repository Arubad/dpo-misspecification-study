# src/ — module map

| Package | Files | What it does |
|---|---|---|
| `common/` | `config.py`, `seed.py`, `explog.py`, `stats.py`, `kaggle_env.py` | YAML configs + cell overrides + config hash; seeding; append-only experiment log; bootstrap/Spearman/partial-Spearman/linear fits; Kaggle paths and dtype choice |
| `prefgen/` | `prompts.py`, `sample.py`, `reward.py`, `annotate.py`, `refprobs.py`, `build.py` | E1+E2: fixed prompts, K=4 samples per prompt from the reference model, r* (DeBERTa reward model) and the sentiment axis f, the exact annotator p* (gamma, mixture, eta), reference log-probs; one parquet cache |
| `train/` | `dataset.py`, `pairs.py`, `dpo_loss.py`, `trainer.py`, `two_timescale.py` | labelled pairs per cell; losses `dpo`, `rdpo`, `auxdpo`, `soft`; minimal LoRA DPO trainer with checkpoints and likelihood-displacement check; E6 online two-timescale variant |
| `eval/` | `evaluate.py` | E9: win-rate vs the majority reward, length, KL to reference, margin IQR, calibration Spearman, paired bootstrap CIs |
| `analysis/` | `epsilon.py`, `hypotheses.py`, `figures.py` | numerical epsilon (BT projection per prompt), the pre-registered pass/fail tests applied mechanically, Figures 1–4 |
| root | `run.py`, `smoke.py` | CLI (`build`, `epsilon`, `train`, `eval`, `sweep`, `online`, `analyze`, `smoke`) |

Design notes:
- The trainer is our own (plain PyTorch + PEFT), not TRL, so that AuxDPO's per-response offsets, R-DPO's length term, the restricted-capacity policy and the online variant share one loss file (`dpo_loss.py`).
- The reference policy is never a second model: it is the PEFT model with adapters disabled. Reference log-probs for the training cache are computed once in E2.
- `pairs_per_prompt` subsamples the 6 labelled pairs per prompt so training cost does not scale with K; epsilon is still computed on all pairs from the exact p*.
