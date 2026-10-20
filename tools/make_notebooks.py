"""Generate the Kaggle notebooks in notebooks/ from one spec (run: python tools/make_notebooks.py).

Each notebook: clone the repo, install requirements, optionally import a previous
notebook's output as the cache, run one experiment through ``python -m src.run``,
then leave results/ in /kaggle/working so it can be attached as a dataset input
to the next notebook.  Kaggle settings to use: Accelerator = GPU T4 x2 (or P100),
Internet = On, Persistence = Files only.
"""
from __future__ import annotations

import os

import json

REPO = "https://github.com/Arubad/dpo-misspecification-study.git"
NAME = "dpo-misspecification-study"

SETUP = f"""# --- setup (identical in every notebook) -------------------------------------------------
import os, subprocess, sys
WORK = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
os.chdir(WORK)
if not os.path.exists('{NAME}'):
    subprocess.run(['git', 'clone', '--depth', '1', '{REPO}'], check=True)
os.chdir('{NAME}')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
os.environ['RESULTS_ROOT'] = os.path.join(WORK, 'results')
# To reuse caches from an earlier notebook: Add Input -> that notebook's output, then set:
PREV = ''   # e.g. '/kaggle/input/dpo-e1e2-cache/results'
if PREV:
    os.environ['PREV_RESULTS'] = PREV
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"""

NOTEBOOKS = [
    ("00_smoke_test", "Smoke test", "Runs every code path on 24 prompts in a few minutes. **Run this first.** Nothing it produces is a result.",
     ["!python -m src.run smoke"]),
    ("01_E1_E2_build_cache", "E1 + E2: response cache", "Samples K=4 responses per prompt from the reference model, scores r* and the sentiment axis, caches reference log-probs. ~2-3 T4-hours for 5k prompts. Output: `results/cache/`. Save this notebook's output as a Kaggle dataset; every later notebook imports it.",
     ["!python -m src.run build --config configs/cache_0p5b.yaml", "!python -m src.run epsilon --config configs/epsilon_sweep_0p5b.yaml",
      "import pandas as pd, json\ndf = pd.read_parquet(os.path.join(os.environ['RESULTS_ROOT'],'cache','responses.parquet'))\nprint(df.groupby('split').size()); print(df[['n_tokens','rstar_z','f_z','ref_logp']].describe())\nprint(json.load(open(os.path.join(os.environ['RESULTS_ROOT'],'cache','epsilon_by_minority_mass.json'))))\n# W3 artifact: 20 labelled pairs and the r* histogram\nfrom src.train.pairs import build_pairs\nimport numpy as np\ndf['row_id']=np.arange(len(df))\npairs = build_pairs(df, {'reward_scale':2.0,'axis_weight':1.0,'gamma':0.0,'mixture':{'K':2,'minority_mass':0.3},'label_flip_eta':0.0}, seed=0)\nprint(pairs[['prompt_text','chosen','rejected','p_exact','len_w','len_l']].head(20).to_string())\nimport matplotlib.pyplot as plt; plt.hist(df['rstar_z'], bins=50); plt.title('r* (z-scored)'); plt.show()"]),
    ("02_E0_control_eta", "E0: expected-to-fail control", "BT-consistent label flips at eta=0.1 must change nothing. If this cell shows an effect, the harness is broken: fix it before running any sweep.",
     ["!python -m src.run sweep --config configs/control_eta.yaml --skip-existing"]),
    ("03_E3_gamma_sweep", "E3: gamma sweep (H1)", "Minimum version: 3 gamma levels x beta 0.1 x 3 seeds = 9 runs (~0.8 h each). Drop `--minimum` for the full 45-run grid.",
     ["!python -m src.run sweep --config configs/gamma_sweep_0p5b.yaml --minimum --skip-existing", "!python -m src.run analyze"]),
    ("04_E4_epsilon_sweep", "E4: epsilon sweep (H2)", "Minimum version: 3 minority-mass levels x 3 seeds. Requires `epsilon_by_minority_mass.json` from notebook 01.",
     ["!python -m src.run sweep --config configs/epsilon_sweep_0p5b.yaml --minimum --skip-existing", "!python -m src.run analyze"]),
    ("05_E5_auxdpo", "E5: AuxDPO decomposition (H3)", "Policy-class cell (restricted capacity, well-specified labels) vs mixture cell (full capacity, minority mass 0.5), DPO vs AuxDPO, 3 seeds.",
     ["!python -m src.run sweep --config configs/auxdpo_cells.yaml --minimum --skip-existing", "!python -m src.run analyze"]),
    ("06_E6_online_two_timescale", "E6 (stretch): online two-timescale DPO", "Only after milestone 4. Produces `convergence_curve.csv` per cell for the C*a_n + b fit.",
     ["!python -m src.run online --config configs/online_two_timescale.yaml --cell minority_mass=0.0,seed=0", "!python -m src.run online --config configs/online_two_timescale.yaml --cell minority_mass=0.3,seed=0", "!python -m src.run analyze"]),
    ("07_analysis_and_figures", "Analysis and figures", "Collects every `metrics.json`, applies the pre-registered tests mechanically and draws Figures 1-4 into `results/figures/`.",
     ["!python -m src.run analyze", "import json; print(json.dumps(json.load(open(os.path.join(os.environ['RESULTS_ROOT'],'figures','hypothesis_report.json'))), indent=2)[:4000])",
      "from IPython.display import Image, display\nimport glob\nfor f in sorted(glob.glob(os.path.join(os.environ['RESULTS_ROOT'],'figures','*.png'))): display(Image(f))"]),
]


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}


def build(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for fname, title, desc, cells in NOTEBOOKS:
        nb_cells = [_md(f"# {title}\n\n{desc}\n\nRepo: {REPO}  \nKaggle: Accelerator GPU T4 x2, Internet On. Results land in `/kaggle/working/results`."), _code(SETUP)]
        nb_cells += [_code(c) for c in cells]
        nb_cells.append(_code("# keep the results as this notebook's output (attach as a dataset to the next notebook)\n!du -sh $RESULTS_ROOT && ls $RESULTS_ROOT"))
        nb = {"cells": nb_cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        with open(os.path.join(outdir, f"{fname}.ipynb"), "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", fname)


if __name__ == "__main__":
    build(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks"))
