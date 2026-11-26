# Working rules

## Brain-first rule (user directive, 2026-11-25)
Every step or decision about the model or its training must make it act more like a real brain. Name the brain mechanism behind each change in docs/iteration-log.md. A change that only raises an accuracy number with no brain analogue does not belong here, even if it scores better; when a brain-like and a score-only option compete, pick the brain-like one and log the cost. Accuracy targets (goal.md §4) are checks that the brain-like machinery works, not the objective.
Name the mechanism and the `research/` file it comes from in docs/iteration-log.md.

## Evidence and evaluation
- Tune on the validation split only (`--split val`); the test split is used only for logged final runs.
- Every reported number is copied from a `results/*.json` file, never estimated. Re-run seed 0 yourself before a number changes a decision.
- Every change is logged in docs/iteration-log.md, failures included, with its brain mechanism and cost.
- Claims follow research/12-prior-attempts-and-warnings.md: claim only where measured, against strong baselines (incl. GDumb).

## Model boundaries (what may live inside SPARC)
Three layers, kept separate (decided 2026-11-26):

| Layer | What | Rule |
|---|---|---|
| **World** | `bench.py`, `data.py`, baselines: the data stream and the scoring | stays **outside** the model; the model never sees test data, task labels at test, or its own score |
| **Body: senses** | sensory front-ends that turn raw input into sparse codes (V1 today; later hearing, …) | part of the model; must learn by local rules, or be a declared, ablated **innate** prior (e.g. the Gabor bank). No pretrained backprop encoders |
| **Brain** | IT, hippocampus, sleep, readouts, and control loops such as **rethink** (ACC conflict detection → PFC top-down gain → second look) | part of the model; local plasticity; inference-time loops have a fixed budget |

New senses (e.g. spoken digits bound to the same IT/hippocampal codes as images) are roadmap docs/roadmap.md #5 and need their own benchmark first.

## Code and files
- SPARC stays local: no autograd, no backprop through layers, batch 1, single pass, no raw-image storage.
- New mechanisms go behind a flag that is off by default until adopted.
- One writer per file: only the main session edits goal.md, docs/rules.md, docs/iteration-log.md and merges into sparc.py.
- Commit after every iteration (message = the log line).
