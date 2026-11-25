# Project rules (all agents)
- Source of truth: goal.md. Only the main session edits it.
- SPARC stays local: no autograd, no backprop through layers, batch 1, single pass, no raw-image storage.
- Tune on the validation split only (--split val). The test set is for logged results.
- Touch only the files your task names. Report numbers copied from results JSON, never estimated.
- Environment: 4 CPU cores, numpy only (no torch). Datasets are cached in data_cache/ (gitignored).
- **Brain-first rule.** Every change to the model or its training must be justified as something a real brain does (name the mechanism). A change that only raises an accuracy number, with no brain analogue, does not belong in this project, even if it scores better. When a brain-like and a score-only option compete, pick the brain-like one and log the cost.
