"""Turn the response cache into labelled training pairs for one cell."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..common.seed import derive_seed
from ..prefgen.annotate import annotate_prompt_group


def build_pairs(responses: pd.DataFrame, annotator_cfg: Dict, seed: int, split: str = "train", pairs_per_prompt: int | None = None) -> pd.DataFrame:
    """One row per labelled pair, joined with both responses' text, length and ref log-prob.

    ``responses`` must carry a ``row_id`` column (its index in the cache) so that
    AuxDPO can allocate one offset per response.
    """
    df = responses[responses["split"] == split].copy()
    if "row_id" not in df.columns:
        df["row_id"] = df.index
    rng = np.random.default_rng(derive_seed(seed, f"labels_{split}"))
    rows: List[Dict] = []
    for pid, g in df.groupby("prompt_idx", sort=True):
        g = g.sort_values("sample_idx")
        if len(g) < 2:
            continue
        group = g.to_dict("records")
        pairs = annotate_prompt_group(group, annotator_cfg, rng)
        if pairs_per_prompt is not None and len(pairs) > pairs_per_prompt:
            keep = rng.choice(len(pairs), size=pairs_per_prompt, replace=False)
            pairs = [pairs[k] for k in sorted(keep)]
        for p in pairs:
            w, l = group[p["chosen"]], group[p["rejected"]]
            rows.append(
                {
                    "prompt_idx": pid,
                    "prompt_text": w["prompt_text"],
                    "chosen": w["response"],
                    "rejected": l["response"],
                    "chosen_id": int(w["row_id"]),
                    "rejected_id": int(l["row_id"]),
                    "len_w": int(w["n_tokens"]),
                    "len_l": int(l["n_tokens"]),
                    "ref_w": float(w["ref_logp"]),
                    "ref_l": float(l["ref_logp"]),
                    "p_exact": p["p_exact"],
                    "p_noisy": p["p_noisy"],
                    "rstar_w": float(w["rstar_z"]),
                    "rstar_l": float(l["rstar_z"]),
                    "true_majority_margin": p["true_majority_margin"],
                }
            )
    out = pd.DataFrame(rows)
    return out.sample(frac=1.0, random_state=int(derive_seed(seed, "shuffle") % (2**32 - 1))).reset_index(drop=True)


def label_sanity(pairs: pd.DataFrame) -> Dict[str, float]:
    """Numbers to print after building pairs: how often the sampled label agrees with the majority annotator, etc."""
    return {
        "n_pairs": int(len(pairs)),
        "frac_label_matches_majority": float((pairs["true_majority_margin"] > 0).mean()),
        "mean_p_exact": float(pairs["p_exact"].mean()),
        "mean_abs_len_diff": float((pairs["len_w"] - pairs["len_l"]).abs().mean()),
        "frac_chosen_longer": float((pairs["len_w"] > pairs["len_l"]).mean()),
    }
