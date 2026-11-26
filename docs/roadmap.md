# Roadmap after the loop
1. **Time**: add temporal prediction (predict the next input from the current sparse code with dendrite-like context), test on sequence streams, then character-level text.
2. **Depth**: stack local layers (SoftHebb-style) for CIFAR-10/100.
3. **Reward**: dopamine-like reward-prediction error as the modulator M for reinforcement learning.
4. **Language**: compare a SPARC-style continual learner with BDH and Nested-Learning/Hope on continual language modelling.
5. **More senses**: a hearing front-end (e.g. spoken digits) learning by the same local rules, bound to the visual codes in IT and the hippocampus (cross-modal recall, multimodal few-shot). Requires a new benchmark in docs/benchmark.md first.
