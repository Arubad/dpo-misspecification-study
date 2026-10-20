"""Command-line entry point.  Every notebook is a thin wrapper around these commands.

  python -m src.run build    --config configs/cache_0p5b.yaml                 # E1 + E2, once
  python -m src.run epsilon  --config configs/epsilon_sweep_0p5b.yaml         # numerical epsilon per minority_mass
  python -m src.run train    --config configs/gamma_sweep_0p5b.yaml --cell gamma=0.3,beta=0.1,seed=0
  python -m src.run eval     --config configs/gamma_sweep_0p5b.yaml --cell gamma=0.3,beta=0.1,seed=0
  python -m src.run sweep    --config configs/gamma_sweep_0p5b.yaml [--minimum] [--skip-existing]   # train + eval every cell
  python -m src.run online   --config configs/online_two_timescale.yaml --cell seed=0
  python -m src.run analyze  [--results results]
  python -m src.run smoke                                                      # tiny CPU/GPU end-to-end check
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .common.config import cell_name, config_hash, iter_cells, load_yaml, parse_cell, resolve, run_dir
from .common.explog import finish_entry, start_entry
from .common.kaggle_env import hf_cache_dir, import_previous_results, results_root


def _expected(cfg) -> str:
    c = cfg.get("_cell", {})
    e = cfg.get("expected", {})
    if isinstance(e, dict):
        for k, v in c.items():
            key = f"{k}={v}"
            if key in e:
                return e[key]
        return e.get("default", "")
    return str(e)


def cmd_build(args):
    from .prefgen.build import build_cache

    cfg = load_yaml(args.config)
    build_cache(cfg, force=args.force)


def cmd_epsilon(args):
    from .analysis.epsilon import epsilon_for_cell
    from .prefgen.build import load_cache

    cfg = load_yaml(args.config)
    responses = load_cache(cfg)
    masses = cfg.get("sweep", {}).get("minority_mass", [cfg["annotator"]["mixture"].get("minority_mass", 0.0)])
    out = {}
    for m in masses:
        a = resolve(cfg, {"minority_mass": m})["annotator"]
        out[str(m)] = epsilon_for_cell(responses, a, max_prompts=int(args.max_prompts))
        print(m, out[str(m)])
    # the flip-noise control is *approximately* BT-consistent; report its epsilon too
    if "control" in cfg:
        a = resolve(cfg, {"label_flip_eta": cfg["control"]["label_flip_eta"], "minority_mass": 0.0})["annotator"]
        out["control_eta"] = epsilon_for_cell(responses, a, max_prompts=int(args.max_prompts))
    path = os.path.join(results_root(), "cache", "epsilon_by_minority_mass.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


def _train_eval(cfg, do_train=True, do_eval=True, skip_existing=False):
    from .eval.evaluate import evaluate_cell
    from .train.trainer import train_cell

    d = run_dir(cfg)
    if skip_existing and os.path.exists(os.path.join(d, "metrics.json")):
        print("skip existing", d)
        return
    meta = start_entry(cfg, d, _expected(cfg))
    metrics = {}
    if do_train:
        metrics.update(train_cell(cfg))
    if do_eval:
        metrics.update(evaluate_cell(cfg))
    finish_entry(meta, metrics)
    return metrics


def cmd_train(args):
    cfg = resolve(load_yaml(args.config), parse_cell(args.cell))
    cfg["logging"]["output_dir"] = results_root(cfg["logging"].get("output_dir", "results"))
    _train_eval(cfg, do_train=True, do_eval=False)


def cmd_eval(args):
    cfg = resolve(load_yaml(args.config), parse_cell(args.cell))
    cfg["logging"]["output_dir"] = results_root(cfg["logging"].get("output_dir", "results"))
    _train_eval(cfg, do_train=False, do_eval=True)


def cmd_sweep(args):
    base = load_yaml(args.config)
    for cell in iter_cells(base, minimum=args.minimum):
        cfg = resolve(base, cell)
        cfg["logging"]["output_dir"] = results_root(cfg["logging"].get("output_dir", "results"))
        print(f"\n===== cell {cell_name(cell)}  hash {config_hash(cfg)} =====")
        _train_eval(cfg, skip_existing=args.skip_existing)


def cmd_online(args):
    from .eval.evaluate import evaluate_cell
    from .train.two_timescale import train_two_timescale

    cfg = resolve(load_yaml(args.config), parse_cell(args.cell))
    cfg["logging"]["output_dir"] = results_root(cfg["logging"].get("output_dir", "results"))
    d = run_dir(cfg)
    meta = start_entry(cfg, d, _expected(cfg))
    m = train_two_timescale(cfg)
    m.update(evaluate_cell(cfg))
    finish_entry(meta, m)


def cmd_analyze(args):
    from .analysis.figures import analyze

    analyze(args.results or results_root())


def cmd_smoke(args):
    """End-to-end on a tiny model and 24 prompts; proves the plumbing before spending GPU hours."""
    from .smoke import run_smoke

    run_smoke(model=args.model)


def main(argv=None):
    hf_cache_dir()
    import_previous_results()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build"); s.add_argument("--config", required=True); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_build)
    s = sub.add_parser("epsilon"); s.add_argument("--config", required=True); s.add_argument("--max-prompts", default=2000); s.set_defaults(fn=cmd_epsilon)
    for name, fn in (("train", cmd_train), ("eval", cmd_eval), ("online", cmd_online)):
        s = sub.add_parser(name); s.add_argument("--config", required=True); s.add_argument("--cell", default=""); s.set_defaults(fn=fn)
    s = sub.add_parser("sweep"); s.add_argument("--config", required=True); s.add_argument("--minimum", action="store_true"); s.add_argument("--skip-existing", action="store_true"); s.set_defaults(fn=cmd_sweep)
    s = sub.add_parser("analyze"); s.add_argument("--results", default=None); s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("smoke"); s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); s.set_defaults(fn=cmd_smoke)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
