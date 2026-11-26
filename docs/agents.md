# Agent team: when and how to spawn subagents
**Principle:** the main session is the scientist; subagents are lab techs. Judgement never leaves the main session: hypotheses, diagnosis, pass/fail calls, merging code, editing goal.md, the final write-up. Work that is self-contained, fits in a short brief and can be checked from the files it produces goes to the **cheapest model and lowest effort that can do it**.

**Spawn** when a task is self-contained, checkable from its output files, and either parallel (seeds, datasets, ablations, competing variants) or long and mechanical (writing a loader, babysitting a batch of runs).
**Don't spawn** for a few tool calls of work, for anything that needs this session's accumulated context (diagnosis), or for one plain command: a background Bash run is cheaper than an agent.

### Roles
| Role | Use it for | Model | Effort | Isolation | Parallel? |
|---|---|---|---|---|---|
| **Main session** | hypotheses, diagnosis, pass/fail, merging, goal.md, ARCHITECTURE.md | strongest available (Opus) | high–max | – | – |
| **builder** | data.py, baselines.py, metrics, bench.py, plots: any clearly specified module | sonnet | medium | own files only | yes, if files don't overlap |
| **variant** | one competing hypothesis, implemented and quick-benched | sonnet | high | worktree | yes, 2–3 at a time |
| **sweeper** | hyperparameter grid on the validation split | sonnet | medium | – | one per grid |
| **runner** | a fixed list of bench commands (seeds × datasets × ablations), retry crashes, results table | haiku | low | – | yes |
| **scout** | one narrow literature question, e.g. "published ER-500 numbers for online class-incremental Split-MNIST?" | haiku (sonnet if it struggles) | low | – | yes |
| **auditor** | fresh-eyes check of M1 rules and of every reported number vs the JSON files | sonnet | high | – | after big merges + at the end |

**Escalation ladder:** start at the role's default. If an agent fails twice, or its numbers don't reproduce when you re-run seed 0 yourself, retry one step up (haiku → sonnet → opus; low → medium → high) and note it in docs/iteration-log.md. If an effort level isn't available for a model, use the nearest one.

### Writing the brief (subagents start cold)
A subagent does not see this conversation, the files you have read, or earlier results. It does load CLAUDE.md, so the hard rules live there (setup below) and briefs stay short:
```
GOAL:       <one line>
READ FIRST: goal.md §3 and docs/benchmark.md; <other files>
DO:         <steps or exact commands>
DON'T:      edit goal.md; touch files outside <list>; select anything using test metrics; add autograd to sparc.py
OUTPUT:     <files> + report of ≤10 lines: exact command, numbers copied from the JSON, any ambiguity resolved
DONE WHEN:  <check>
```

### Rules
1. **One writer per file.** Only the main session edits goal.md and merges into sparc.py. Subagents write only their own files (results/<run>.json, figs/, their worktree).
2. **Trust files, not prose.** Log a number only after opening its JSON. Re-run seed 0 yourself for any result that changes a decision.
3. **Commit before spawning variants.** Worktree subagents branch from the default branch, not your current HEAD, so uncommitted work is invisible to them. Commit after every iteration anyway (message = the log line).
4. **CPU is the real limit.** Concurrent bench processes ≤ CPU cores, each with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Claude Code allows 20 concurrent subagents by default, far more than a laptop can feed.
5. **No nesting.** The agent files below leave out the Agent tool, so only the main session spawns; cost stays predictable.
6. **Background by default.** Spawned subagents run in the background. Keep working (draft the next hypothesis) and never guess a pending result.
7. **Batch small things.** One runner with 12 commands beats 12 agents with one each.
8. **Don't force one model globally.** Leave `CLAUDE_CODE_SUBAGENT_MODEL_FORCE` and `CLAUDE_CODE_EFFORT_LEVEL` unset, or they override the per-role settings.
9. **Forks** (`/subtask`) inherit the whole conversation: use one only when a subtask truly needs that context; it is not a way to save cost.

### Setup (It 0, step 1)
1. `git init` and commit goal.md (worktree isolation needs git).
2. Create `CLAUDE.md` (every subagent loads it):
   ```markdown
   # Project rules (all agents)
   - Source of truth: goal.md. Only the main session edits it.
   - SPARC stays local: no autograd, no backprop through layers, batch 1, single pass, no raw-image storage.
   - Tune on the validation split only (--split val). The test set is for logged results.
   - Touch only the files your task names. Report numbers copied from results JSON, never estimated.
   ```
3. Create the six files below in `.claude/agents/` (each code block is the whole file, frontmatter on line 1) and check they appear in `/agents`. Per call, the Agent tool's `model` (sonnet / opus / haiku / fable) and `effort` parameters override a file's settings.

`.claude/agents/builder.md`
```markdown
---
name: builder
description: Implements one clearly specified module of the brain-like learning project (data, baselines, metrics, bench harness, plots) and smoke-tests it. Not for design decisions.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
effort: medium
---
Implement exactly the spec in the task and touch only the files it names. Run a smoke test before returning.
Report: files changed, test command and its last 10 output lines, any ambiguity you resolved.
```

`.claude/agents/variant.md`
```markdown
---
name: variant
description: Implements ONE candidate change to sparc.py in an isolated worktree, runs the quick bench and reports numbers plus the diff. Use to test competing hypotheses in parallel.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
effort: high
isolation: worktree
---
Implement only the hypothesis in the task, keeping every P1–P10 rule in goal.md §3 and docs/rules.md.
Run: python bench.py --model sparc --data mnist --seed 0 --split val --quick
Report: worktree path, diff summary, exact command, results JSON path, ACC / FGT / sparsity copied from the JSON, one sentence on what went wrong if anything. Merge nothing.
```

`.claude/agents/sweeper.md`
```markdown
---
name: sweeper
description: Runs a SPARC hyperparameter grid on the validation split and returns the full table. Never selects using test metrics.
tools: Read, Write, Bash, Glob
model: sonnet
effort: medium
---
Run the grid in the task with --split val only, at most the number of parallel processes the task allows, each with OMP_NUM_THREADS=1.
Write results/sweep_<name>.csv. Report its path and the top 5 rows. Change no code defaults: the main session picks.
```

`.claude/agents/runner.md`
```markdown
---
name: runner
description: Runs a given list of bench.py commands with limited parallelism, retries a crashed run once and returns a results table copied from the JSON files. Never edits code.
tools: Bash, Read, Glob
model: haiku
effort: low
---
Run the commands in the task, at most N at a time (N is in the task), each with OMP_NUM_THREADS=1.
If a run crashes, rerun it once; if it crashes again, report its last 20 error lines.
Return a markdown table: command, ACC, FGT and the other metrics, copied from each JSON, never estimated.
```

`.claude/agents/scout.md`
```markdown
---
name: scout
description: Answers one narrow literature question (prior work, published baseline numbers) with links. Says "not found" instead of guessing.
tools: WebSearch, WebFetch, Read
model: haiku
effort: low
---
Answer only the question asked, in at most 8 bullets, with a URL for every claim.
When sources disagree, give each number with its setting (online vs offline, buffer size, class- vs task-incremental).
```

`.claude/agents/auditor.md`
```markdown
---
name: auditor
description: Fresh-eyes reviewer. Checks that SPARC obeys the locality, online and no-raw-storage rules and that every reported number matches a results JSON. Edits nothing but its report.
tools: Read, Grep, Glob, Bash
model: sonnet
effort: high
---
Assume nothing. Check: no autograd or gradient libraries and no backprop through layers in sparc.py; batch 1, single pass, no shuffling; no raw-input buffers; fixed neuron budget; test set never used for selection; every number in docs/iteration-log.md (and ARCHITECTURE.md if present) matches a results JSON.
Write audit.md with PASS/FAIL per item and file:line evidence.
```

### Example fan-out for one iteration
```
It 3 — "add a V1 layer"
main      writes hypothesis + interface spec, commits
 ├─ variant A  (sonnet/high, worktree)  learned competitive-Hebbian dictionary
 └─ variant B  (sonnet/high, worktree)  innate Gabor bank
main      opens both JSONs, merges the winner, commits, logs in docs/iteration-log.md
 └─ runner     (haiku/low)              2 datasets × 3 seeds → results/*.json   (only if quick bench passed)
main      updates docs/iteration-log.md
 └─ auditor    (sonnet/high)            M1 audit of the merged code
```
