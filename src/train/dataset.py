"""Tokenisation shared by the reference-logprob cache, training and evaluation.

The response span is defined once here so that log p(y | x) means the same thing
everywhere: the tokens of ``response + eos`` appended to the chat-formatted prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..prefgen.prompts import chat_format


@dataclass
class Encoded:
    input_ids: List[int]
    response_mask: List[int]  # 1 on response tokens (positions whose *prediction* is a response token)


def encode_prompt_response(tokenizer, prompt: str, response: str, max_prompt_len: int, max_len: int, system: Optional[str] = None) -> Encoded:
    prompt_text = chat_format(tokenizer, prompt, system)
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"][-max_prompt_len:]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        r_ids = r_ids + [eos]
    r_ids = r_ids[: max(1, max_len - len(p_ids))]
    ids = p_ids + r_ids
    mask = [0] * len(p_ids) + [1] * len(r_ids)
    return Encoded(ids, mask)


def collate(batch: List[Encoded], pad_id: int, device) -> Dict[str, torch.Tensor]:
    n = max(len(b.input_ids) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    rmask = torch.zeros((len(batch), n), dtype=torch.float32)
    for k, b in enumerate(batch):
        L = len(b.input_ids)
        ids[k, :L] = torch.tensor(b.input_ids)
        att[k, :L] = 1
        rmask[k, :L] = torch.tensor(b.response_mask, dtype=torch.float32)
    return {"input_ids": ids.to(device), "attention_mask": att.to(device), "response_mask": rmask.to(device)}


def sequence_logprobs(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum of log p(token_t | tokens_<t) over response tokens, per sequence."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    targets = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:]
    # gather - logsumexp avoids materialising a full log_softmax copy (Qwen's vocab is 151k)
    tgt = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    lse = torch.logsumexp(logits.float(), dim=-1)
    logp = tgt - lse
    return (logp * mask).sum(dim=-1)


def response_lengths(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["response_mask"][:, 1:].sum(dim=-1)
