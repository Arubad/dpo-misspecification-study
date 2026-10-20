"""Fixed prompt set from a public instruction dataset, with a seeded train / held-out split."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

SOURCES = {
    # name -> (hf dataset id, split, field holding the instruction, optional input field)
    "alpaca_cleaned": ("yahma/alpaca-cleaned", "train", "instruction", "input"),
    "dolly": ("databricks/databricks-dolly-15k", "train", "instruction", "context"),
}


def load_prompts(source: str, n_train: int, n_eval: int, seed: int, min_chars: int = 20, max_chars: int = 600) -> Dict[str, List[str]]:
    """Return ``{"train": [...], "eval": [...]}`` prompts, deduplicated and length-filtered."""
    from datasets import load_dataset

    ds_id, split, field, extra = SOURCES[source]
    ds = load_dataset(ds_id, split=split)
    prompts: List[str] = []
    seen = set()
    for row in ds:
        p = (row.get(field) or "").strip()
        ctx = (row.get(extra) or "").strip() if extra else ""
        if ctx:
            p = f"{p}\n\n{ctx}"
        if not (min_chars <= len(p) <= max_chars):
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(p)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(prompts))
    need = n_train + n_eval
    if len(prompts) < need:
        raise ValueError(f"only {len(prompts)} usable prompts in {source}, need {need}")
    chosen = [prompts[i] for i in idx[:need]]
    return {"train": chosen[:n_train], "eval": chosen[n_train:need]}


def chat_format(tokenizer, prompt: str, system: str | None = None) -> str:
    """Apply the model's chat template so sampling and training see identical prompts."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
