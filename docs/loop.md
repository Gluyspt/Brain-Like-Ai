# Loop protocol
Each iteration:
1. **Hypothesis**: which brain mechanism should fix which observed failure (one sentence). Main session only.
2. **Implement** the smallest change. One change per iteration. When two or three fixes compete, give each to a `variant` agent in parallel (docs/agents.md) and keep the winner.
3. **Run** quick bench (MNIST, seed 0). Full bench (2 datasets × 3 seeds) only once quick passes, handed to a `runner` agent.
4. **Log** the numbers against the success criteria (goal.md §4) in docs/iteration-log.md, after opening the JSON files yourself.
5. **Diagnose** → next hypothesis. Main session only.

Suggested order (deviate when the data says so):
- **It 0** — setup from docs/agents.md (git, CLAUDE.md, agent files); a `builder` writes data.py / baselines.py / bench.py; a `runner` runs B1–B4; check the sanity ranges in docs/benchmark.md.
- **It 1** — IT layer + fast readout on raw pixels (no V1): tests the core anti-forgetting mechanism.
- **It 2** — metaplasticity + vigilance recruitment tuning if FGT is high.
- **It 3** — add V1 if ACC is below target (learned vs innate-Gabor ablation).
- **It 4** — hippocampus + sleep; measure S2.
- **It 5** — few-shot, few-label, novelty metrics; fix whichever fails.
- **It 5b** — self-correction (P10): confident-mistake depression of active→wrong-class links (three-factor rule, M = confident error), reopen plasticity of the units active in the mistake (reconsolidation), evidence-weighted links so one bad label only nudges, sleep cleanup of conflicting hippocampal memories. Test with REVERSAL and NOISE. Known weakness it fixes: the fast readout only ever strengthens the labelled class, so a wrong link is only diluted, never removed.
- **It 6+** — full run, ablations (S3), then Fashion-specific fixes.

**Stop** when every MUST passes on the full bench → run ablations → write `ARCHITECTURE.md` (design, equations, brain mapping, results table, limits). Or stop after **12 iterations** and report the best version plus an honest gap analysis.
