# Working rules

## Brain-first rule (user directive, 2026-11-25)
Every step or decision about the model or its training must make it act more like a real brain. Name the brain mechanism behind each change in docs/iteration-log.md. A change that only raises an accuracy number with no brain analogue does not belong here, even if it scores better; when a brain-like and a score-only option compete, pick the brain-like one and log the cost. Accuracy targets (goal.md §4) are checks that the brain-like machinery works, not the objective.
Name the mechanism and the `research/` file it comes from in docs/iteration-log.md.

**Copy the function, not the limitation** (user directive, 2026-11-27). Adopt a brain mechanism when it does a job (learning, protecting memories, generalising, thinking). Do not copy a weakness that exists only because of biology's costs (small capacity, slowness, forgetting with no benefit). When a brain limit is kept on purpose (e.g. fading memories), name the job it does (e.g. clearing noise, less interference, better generalisation; Richards & Frankland 2017 [known]) and show it measurably helps; otherwise drop it and log why. This refines the brain-first rule, it does not replace it: every mechanism must still come from the brain, and a change that only raises a score is still out.

## Reporting comparisons (user directive, 2026-11-27)
Every comparison, in chat and in docs, shows **old (baseline) vs new** side by side, each labelled with what it is (config name, split, seeds), with the metric name and its direction (higher / lower is better), and the change.

## Research → test rule (user directive, 2026-11-27)
Every research finding that suggests something worth trying **gets tested**, unless one of two exceptions applies:
1. **More important work comes first**: something already in progress, or a fix for a failing criterion of the current stage, has priority. The idea waits in the queue, then runs.
2. **It belongs to a later stage**: it is parked under that stage in `research/10-experiment-backlog.md` (stage tag) and tested when that stage starts.
Each research note ends with its "try" ideas, and each one gets a backlog row with a status (queued / running / tested → result / parked for stage N) and a reason. Nothing worth trying is left without a decision. When a test finishes, its result goes into the research note and docs/iteration-log.md.

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
- **Push and merge to `main` every time a full task or key piece of work is complete** (user directive, 2026-11-26): push the working branch, then fast-forward `main` to it (`git push origin HEAD:main`). Work in progress stays on the working branch. If `main` has moved, merge it into the branch first; never force-push `main`.
