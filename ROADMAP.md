# Roadmap and progress tracker

Ten weeks, one project at a time, minimum-viable version first. Tick boxes as you go; GitHub renders the counts in the file view. Every GPU run also gets an entry in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

**Start date:** _fill in_ · **Target venue:** ICLR 2027 or NeurIPS 2027 workshop (4 pages); IndoML / CODS-COMAD as the Indian venue.

## Milestone 0 — Pre-registration v0.2 (done 2026-10-18)
- [x] Independent review ([`REVIEW.md`](REVIEW.md))
- [x] H3 rewritten; amendments logged in [`PREREGISTRATION.md`](PREREGISTRATION.md)
- [x] Scaffold committed (configs, requirements, src layout, log template)

## Milestone 1 — Weeks 1–2: reading and ownership (no GPU)
- [ ] Foundations block of [`READING.md`](READING.md) read, with a half-page note per paper in `paper/notes/`
- [ ] Project block of [`READING.md`](READING.md) read, same notes
- [ ] Derive the DPO population optimum and the γ/β-per-token prediction by hand (`paper/notes/derivation_h1.md`)
- [ ] Read Gopalan et al. Prop. 1, Prop. 3 and the AuxDPO definition; write one paragraph in your own words on why AuxDPO should not fix label misspecification
- [ ] Rewrite the three hypotheses in your own words in a new amendment (same tests; your wording)
- [ ] Semantic Scholar sweep for prior controlled synthetic-preference studies; record in `REVIEW.md` §6

## Milestone 2 — Weeks 3–4: harness and control
- [ ] `src/prefgen`: prompts, paired sampling, r* scoring, γ / mixture / η nuisance terms; one parquet cache (E1 + E2)
- [ ] Notebook showing 20 labelled pairs and the r* histogram
- [ ] `src/train`: DPO on LoRA via TRL, fp32, per-step checkpoints; config hash + git commit tagging
- [ ] `src/eval`: win-rate vs r*, length, KL, margin IQR, calibration, likelihood displacement; paired bootstrap
- [ ] Smallest end-to-end run: 1 config, 1 seed, 200 prompts, 50 steps
- [ ] **Expected-to-fail control passes:** η = 0.1 cell shows no effect (logged)
- [ ] Decide Qwen2.5-0.5B vs Qwen3-0.6B; log the decision

## Milestone 3 — Weeks 5–7: minimum-viable sweeps (~25 GPU-h)
- [ ] E3 γ sweep: 3 levels × β 0.1 × 3 seeds
- [ ] E4 ε sweep: 3 levels × 3 seeds; ε computed numerically
- [ ] E5 AuxDPO: policy-class cell + largest-ε cell × 3 seeds
- [ ] Length-vs-γ/β plot with CIs; KL plot
- [ ] Win-rate-vs-ε figure; margin-IQR table
- [ ] AuxDPO recovery table (two cells)
- [ ] Every run has a complete `EXPERIMENT_LOG.md` entry

## Milestone 4 — Weeks 8–10: analysis and write-up
- [ ] H1, H2, H3 each marked confirmed / refuted / under-powered against the pre-registered thresholds
- [ ] Negative-results section drafted
- [ ] 4-page workshop paper in `paper/`
- [ ] One-page summary for Aditya Gopalan (ε-scaling figure, calibration result, AuxDPO decomposition)
- [ ] `python -m src.reproduce --figure 1` works from cached results
- [ ] Decide whether the full version (E3 full grid, E8 1.5B replication, E6 stretch) is worth ~100 more GPU-hours

## Stretch (only after Milestone 4)
- [ ] E6 online two-timescale variant with a stated bias-floor prediction
- [ ] E8 1.5B replication of headline cells
- [ ] Length-normalised DPO baseline across all ε
