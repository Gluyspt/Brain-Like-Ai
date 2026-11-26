# Project rules (all agents)
- Goal: goal.md. Rules: docs/rules.md. Results log: docs/iteration-log.md. Benchmark: docs/benchmark.md. Only the main session edits these.
- SPARC stays local: no autograd, no backprop through layers, batch 1, single pass, no raw-image storage.
- Tune on the validation split only (--split val). The test set is for logged results.
- Touch only the files your task names. Report numbers copied from results JSON, never estimated.
- Environment: 4 CPU cores, numpy only (no torch). Datasets are cached in data_cache/ (gitignored).
- **Brain-first rule.** Every change to the model or its training must be justified as something a real brain does (name the mechanism). A change that only raises an accuracy number, with no brain analogue, does not belong in this project, even if it scores better. When a brain-like and a score-only option compete, pick the brain-like one and log the cost.
- Background research: research/ (one .md per brain system; index in research/README.md). Name the file a mechanism comes from when logging it.
- Keep the work clean: goal.md holds only the goal; findings go in research/, rules and protocols in docs/, run lists in runs/, helper scripts in tools/, outputs in results/.
- Git: commit on the working branch; when a full task or key piece of work is complete, push it and fast-forward `main` to it (docs/rules.md). Never force-push `main`.
