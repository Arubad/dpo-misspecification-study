"""Reference log-probabilities log pi_ref(y | x) for every cached response (E2)."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

from ..train.dataset import collate, encode_prompt_response, sequence_logprobs


@torch.no_grad()
def reference_logprobs(model, tokenizer, prompts: List[str], responses: List[str], max_prompt_len: int, max_len: int, batch_size: int = 8, system: str | None = None) -> np.ndarray:
    device = next(model.parameters()).device
    enc = [encode_prompt_response(tokenizer, p, r, max_prompt_len, max_len, system) for p, r in zip(prompts, responses)]
    # sort by length for efficient batching, then unsort
    order = np.argsort([len(e.input_ids) for e in enc])
    out = np.zeros(len(enc), dtype=np.float64)
    for s in tqdm(range(0, len(enc), batch_size), desc="ref logprobs"):
        idx = order[s : s + batch_size]
        batch = collate([enc[i] for i in idx], tokenizer.pad_token_id, device)
        lp = sequence_logprobs(model, batch).cpu().numpy()
        out[idx] = lp
    return out
