"""numpy/scipy-only tests; run with  python -m pytest tests  or  python tests/test_epsilon_and_stats.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.analysis.epsilon import bt_projection_one_prompt, epsilon_from_groups
from src.prefgen.annotate import annotator_rewards, exact_pair_probs, mixture_weights, apply_flip_noise, annotate_prompt_group
from src.common.stats import bootstrap_ci, paired_bootstrap_diff, spearman, partial_spearman, linear_fit, win_rate, iqr
from src.common.config import resolve, config_hash, parse_cell, iter_cells, cell_name


def _group(K, seed, minority, gamma=0.0, lam=1.0):
    rng = np.random.default_rng(seed)
    rz, fz, ln = rng.normal(size=K), rng.normal(size=K), rng.integers(20, 200, size=K).astype(float)
    cfg = {"reward_scale": 2.0, "axis_weight": lam, "gamma": gamma, "mixture": {"K": 2, "minority_mass": minority}}
    R = annotator_rewards(rz, fz, cfg)
    return exact_pair_probs(R, ln, gamma, mixture_weights(cfg))


def test_single_annotator_is_bt():
    for gamma in (0.0, 0.1, 0.3):
        groups = [_group(4, s, minority=0.0, gamma=gamma) for s in range(30)]
        e = epsilon_from_groups(groups, 4)
        assert e["epsilon"] < 1e-6, (gamma, e)


def test_mixture_is_not_bt_and_grows_with_mass():
    eps = []
    for m in (0.0, 0.1, 0.3, 0.5):
        groups = [_group(4, s, minority=m, lam=2.0) for s in range(60)]
        eps.append(epsilon_from_groups(groups, 4)["epsilon"])
    assert eps[0] < 1e-6 and eps[1] > 1e-5 and eps[2] > eps[1] and eps[3] > eps[2], eps


def test_two_items_always_bt():
    groups = [_group(2, s, minority=0.5, lam=3.0) for s in range(20)]
    assert epsilon_from_groups(groups, 2)["epsilon"] < 1e-8


def test_flip_noise_is_nearly_bt():
    groups = []
    for s in range(60):
        g = _group(4, s, minority=0.0)
        groups.append({k: apply_flip_noise(v, 0.1) for k, v in g.items()})
    e = epsilon_from_groups(groups, 4)["epsilon"]
    em = epsilon_from_groups([_group(4, s, minority=0.5, lam=2.0) for s in range(60)], 4)["epsilon"]
    assert e < em / 2, (e, em)  # control is much closer to BT than the mixture


def test_projection_recovers_rewards():
    r = np.array([0.0, 1.0, -0.5, 2.0])
    probs = {(i, j): 1 / (1 + np.exp(-(r[i] - r[j]))) for i in range(4) for j in range(i + 1, 4)}
    kl, rhat = bt_projection_one_prompt(probs, 4)
    assert kl < 1e-9
    assert np.allclose(rhat - rhat[0], r - r[0], atol=1e-3)


def test_annotate_group_labels():
    rng = np.random.default_rng(0)
    group = [{"prompt_idx": 7, "sample_idx": k, "rstar_z": float(k), "f_z": 0.0, "n_tokens": 50} for k in range(3)]
    cfg = {"reward_scale": 5.0, "axis_weight": 1.0, "gamma": 0.0, "mixture": {"K": 1, "minority_mass": 0.0}, "label_flip_eta": 0.0}
    pairs = annotate_prompt_group(group, cfg, rng)
    assert len(pairs) == 3
    for p in pairs:  # with a huge scale the higher r* response should win
        assert p["chosen"] > p["rejected"] or p["p_exact"] > 0.99


def test_length_contamination_prefers_long():
    cfg = {"reward_scale": 1.0, "axis_weight": 1.0, "gamma": 0.3, "mixture": {"K": 1, "minority_mass": 0.0}}
    R = annotator_rewards(np.array([1.0, 0.0]), np.zeros(2), cfg)
    p = exact_pair_probs(R, np.array([50.0, 150.0]), 0.3, mixture_weights(cfg))[(0, 1)]
    assert p < 0.5  # the shorter, better answer loses under strong length contamination


def test_stats():
    x = np.arange(100, dtype=float)
    ci = bootstrap_ci(x, n_boot=500)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    d = paired_bootstrap_diff(x + 1, x, n_boot=500)
    assert abs(d["mean"] - 1) < 1e-9 and d["lo"] > 0.99
    assert abs(spearman(x, x**3) - 1) < 1e-12 and abs(spearman(x, -x) + 1) < 1e-12
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000)
    a, b = z + 0.3 * rng.normal(size=2000), z + 0.3 * rng.normal(size=2000)
    assert spearman(a, b) > 0.8 and abs(partial_spearman(a, b, z)) < 0.1  # shared cause removed
    assert np.isnan(partial_spearman(z, z, z))
    f = linear_fit(x, 2 * x + 3)
    assert abs(f["slope"] - 2) < 1e-9 and abs(f["r2"] - 1) < 1e-9
    assert win_rate([1, 2, 3], [0, 2, 4]) == 0.5
    assert iqr(np.arange(101)) == 50


def test_config_resolve_and_hash():
    base = {"experiment_id": "X", "annotator": {"gamma": 0.0, "mixture": {"K": 1, "minority_mass": 0.0}}, "train": {"beta": 0.1, "seed": 0}, "model": {"capacity": "full"}, "sweep": {"gamma": [0, 0.3], "seed": [0, 1]}}
    cells = list(iter_cells(base))
    assert len(cells) == 4
    c = resolve(base, parse_cell("gamma=0.3,seed=1,capacity=restricted"))
    assert c["annotator"]["gamma"] == 0.3 and c["train"]["seed"] == 1 and c["model"]["capacity"] == "restricted" and "sweep" not in c
    assert config_hash(c) != config_hash(resolve(base, parse_cell("gamma=0.3,seed=0")))
    assert cell_name({"gamma": 0.3, "seed": 1}) == "gamma0p3_seed1"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
