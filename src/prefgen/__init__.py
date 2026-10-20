"""Synthetic preference generator with an exactly known ground-truth reward.

Pipeline (E1 + E2, run once and cached):
  prompts.py   -> fixed prompt set (train / held-out split, seeded)
  sample.py    -> K responses per prompt from the reference model
  reward.py    -> r*(x, y) from a small public reward model; a second axis f(x, y)
  annotate.py  -> p*(y_i > y_j | x) under length contamination gamma, a K-annotator
                  mixture, and BT-consistent flip noise eta; sampled labels
  refprobs.py  -> reference log-probabilities of every response (cached once)
  build.py     -> runs everything and writes results/cache/pairs.parquet
"""
