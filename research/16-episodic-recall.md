# 16 — Episodic recall at answer time

## What the brain does
The hippocampus stores specific episodes and can bring one back from a partial cue (pattern completion in CA3). Recent memories, not yet consolidated into cortex, depend on this recall: you recognise the new colleague because you remember meeting them, not because your cortex has a category for them yet (complementary learning systems).

## Relation to RAG (user question, 2026-11-26)
RAG looks up raw documents outside the model and pastes them into the prompt; the model learns nothing and nothing is forgotten. Episodic recall in SPARC is the brain-like part of that idea:
- it uses compressed codes the model itself stored (15 numbers per trace, P7 holds);
- the store is bounded and forgets;
- the same traces also train the cortex during sleep (consolidation).

## How it works in SPARC (flag `epi`)
- The cue is the image's sparse IT code (15 of 3000 units).
- Cosine with every stored trace (sparse: ~15 × 1000 operations per image).
- For each class, the best-matching trace of that class is its recall strength; a softmax with gain `epi_beta` turns these into a vote p_e.
- The vote joins the existing fast/slow mix with weight `epi · max(p_e)`: confident recall counts more.

## Expected effect
Few-shot classes have their 5 shots in the hippocampus, so a test image can match "that 8 I saw" directly. Targets: MNIST FEW-5 (42.7 vs 70), and the relearning test (research/15).

## Status
Implemented (flag off; MNIST predictions unchanged). Screen queued: strength 0.5 / 1 / 2, quick, 3 datasets.
