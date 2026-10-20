"""Ground-truth reward r*(x, y) and the second annotator axis f(x, y).

r* comes from a small public pairwise reward model (sequence classifier that
scores a (prompt, response) pair).  Both r* and f are z-scored over the whole
generated set so that the annotator model in annotate.py works in comparable
units.  The *known* axis f is a sentiment score from an SST-2 classifier: the
second annotator rewards cheerful answers, which is independent of helpfulness
and therefore cannot be absorbed into a single BT reward across prompts.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

DEFAULT_REWARD_MODEL = "OpenAssistant/reward-model-deberta-v3-base"
DEFAULT_AXIS_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


@torch.no_grad()
def score_reward_model(model_name: str, prompts: List[str], responses: List[str], device, batch_size: int = 16, max_length: int = 1024) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    scores = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"scoring {model_name.split('/')[-1]}"):
        enc = tok(prompts[i : i + batch_size], responses[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        logits = model(**enc).logits
        scores.append(logits[:, 0].float().cpu().numpy())
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.concatenate(scores)


@torch.no_grad()
def score_sentiment_axis(model_name: str, responses: List[str], device, batch_size: int = 32, max_length: int = 512) -> np.ndarray:
    """Positive-class log-odds from an SST-2 classifier, used as the second annotator's axis."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    pos_idx = 1
    for k, v in model.config.id2label.items():
        if str(v).upper().startswith("POS"):
            pos_idx = int(k)
    out = []
    for i in tqdm(range(0, len(responses), batch_size), desc="sentiment axis"):
        enc = tok(responses[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        logits = model(**enc).logits.float()
        logodds = logits[:, pos_idx] - logits[:, 1 - pos_idx]
        out.append(logodds.cpu().numpy())
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.concatenate(out)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)
