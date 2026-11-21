# Data Directory Policy

Per Section 5.3 of `NEURAL_HESTON_SPEC.md`:

- **No vendor market data is committed to this repository.** Almost all option data licenses prohibit redistribution.
- Everything inside `data/` is git-ignored except `data/README.md` and `data/mini_synthetic/` (<= 1 MB synthetic fixture used by CI tests and `make repro-small`).
- Generated datasets (120k+ synthetic surfaces) and any locally downloaded SPX snapshots live under `data/` locally and remain untracked.
