# 21 — MeMo ("Memory as a Model") vs SPARC

User question (2026-11-27): an Instagram reel ("rag is dead. mit built memo — a second brain for ai", labelled AI-generated) sounded like our project.

## What it actually is [abstract / secondary sources; arXiv not reachable from this environment]
- **MeMo: Memory as a Model**, arXiv 2605.15156 (May 2026), authors from NUS, A*STAR, MIT CSAIL and SMART: https://arxiv.org/html/2605.15156 · code https://github.com/arunv3rma/MeMo · project page https://arunv3rma.github.io/blogs/memo/
- A small **Memory model** (reportedly initialised from Qwen2.5-14B-Instruct, https://beta.hyper.ai/en/papers/2605.15156) is trained, with ordinary fine-tuning, on **"reflection" question-answer data** that a frozen generator model creates from the document corpus.
- The main LLM (**Executive**) stays frozen. At inference it decides whether memory is needed, splits the question into sub-questions and **interrogates** the Memory model over several turns (https://www.alphaxiv.org/abs/2605.15156).
- Reported: NarrativeQA 53.58 % vs 23.21 % for HippoRAG2 (Gemini-3.0-Flash); robustness to retrieval noise; the Executive can be swapped without retraining the memory.
- Costs (authors): upfront compute to build the reflection data and train the memory; multi-turn inference. Critique: the evaluation does not cleanly separate the memory's contribution from the frozen LLM's (https://pith.science/paper/2605.15156). A streaming setting with a second corpus arriving later is mentioned (https://smallaimodel.substack.com/p/2026-05-14-memo-memory-as-a-model); results not found.
- Reel claims vs paper: "RAG is dead" (no: an alternative that wins on some long-document benchmarks), "no retrieval" (it still decides when to query and decomposes questions), "perfect recall" (not claimed), "MIT built" (one of several affiliations).

## Shared idea with SPARC
| | Brain | SPARC | MeMo |
|---|---|---|---|
| memory separate from the reasoning system | hippocampus vs cortex (CLS) | hippocampus + slow cortical readout | Memory model + frozen Executive |
| store digested knowledge, not raw copies | consolidation keeps the gist | compressed codes, no raw images (P7) | reflections instead of text chunks |
| recall by asking | PFC cue → hippocampal recall | episodic recall (research/16) | Executive interrogates Memory |
| protect existing knowledge | consolidation without overwriting | sparse local updates | frozen LLM |

## Key difference
MeMo is a **harness around a frozen LLM** (goal.md §2), trained offline with backprop; new knowledge means retraining a 14B-parameter model. SPARC learns **online, sample by sample, with local rules**, and keeps learning. MeMo is the practical route at language scale today; SPARC tests the brain-like learning route.

## Use for this project
1. Evidence that the hippocampus/cortex split pays off even at LLM scale.
2. The interrogation loop is a concrete model of **PFC–hippocampus interaction** for stages 4–5: a reasoning system asks memory focused sub-questions over several steps.
3. Stage-6 comparison (docs/roadmap.md): continual document learning, i.e. learning new documents online without retraining or forgetting, SPARC-style memory vs MeMo vs RAG.
