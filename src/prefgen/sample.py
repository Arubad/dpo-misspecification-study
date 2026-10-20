"""Sample K responses per prompt from the reference policy (batched, seeded)."""
from __future__ import annotations

from typing import Dict, List

import torch
from tqdm.auto import tqdm

from .prompts import chat_format


def load_causal_lm(name: str, dtype, device, lora_adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    if lora_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_adapter)
        model = model.merge_and_unload()
    model.to(device).eval()
    return tok, model


@torch.no_grad()
def sample_responses(
    model,
    tokenizer,
    prompts: List[str],
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    system: str | None = None,
    greedy: bool = False,
) -> List[Dict]:
    """Return a list of ``{"prompt_idx", "sample_idx", "prompt_text", "response", "n_tokens"}``."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    out: List[Dict] = []
    formatted = [chat_format(tokenizer, p, system) for p in prompts]
    for start in tqdm(range(0, len(formatted), batch_size), desc="sampling"):
        chunk = formatted[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        if greedy:
            kwargs.update(do_sample=False, num_return_sequences=1)
            reps = 1
        else:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=k)
            reps = k
        torch.manual_seed(seed + start)  # generation uses the global RNG
        ids = model.generate(**enc, **kwargs)
        new = ids[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new, skip_special_tokens=True)
        for j, text in enumerate(texts):
            p_local = j // reps
            n_tok = int((new[j] != tokenizer.pad_token_id).sum().item())
            out.append(
                {
                    "prompt_idx": start + p_local,
                    "sample_idx": j % reps,
                    "prompt_text": prompts[start + p_local],
                    "response": text.strip(),
                    "n_tokens": n_tok,
                }
            )
    del gen
    return out
