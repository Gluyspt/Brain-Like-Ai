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
