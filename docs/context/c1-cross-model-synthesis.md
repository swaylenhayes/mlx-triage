# C1 Cross-Model Research Synthesis

This document synthesizes findings from multiple independent research analyses conducted on the confirmatory leg (C1) of Variant C — examining whether infrastructure defects can produce symptoms indistinguishable from model failure.

## C1 Cross-Analysis Agreement

The confirmatory leg came back substantially stronger than the disconfirming leg. Every analysis found concrete, specific, citable evidence. But the quality and depth varied dramatically.

**The highest-value individual finding** was the baa.ai engineering postmortem on quantizing Qwen3.5-397B on a Mac Studio, which documents four distinct MLX bugs — two of which produce total model collapse from infrastructure causes alone. The bfloat16 serialization corruption bug (structurally valid safetensors file, numerically garbage values, `What is 2+2?` → `!!!!!!!!!!!!!!!!!!`) is exactly the kind of case the hypothesis predicts. Anthropic's September 2025 postmortem was also surfaced — TPU misconfig producing stray Chinese characters in English responses, initially attributed to model changes — a production-scale confirmation of the hypothesis from the other side of the stack.

**The densest collection of MLX-specific GitHub issues** included the PaliGemma GELU approximation mismatch (vision encoder activations diverging >100k sum-abs-diff), the Llama-3 70B 4-bit spiky weights bug fixed by bias adjustment in MLX PR #1016, KV-cache trim wiping entire caches, and LM Studio's MLX engine reprocessing full prompts each turn. These are all real postmortems with before/after diagnostics.

**One analysis produced particularly operationally useful output** — a structured table mapping case type → what it looked like → what it actually was → how it was isolated → fix. That table format is directly usable as a diagnostic reference. It also surfaced the mlx-lm `cache_limit` issue where memory growth during inference manifested as progressive quality degradation.

**The most thorough methodological coverage** addressed cross-runtime validation, precision sensitivity testing, memory pressure correlation, and tokenizer configuration verification, connecting a high volume of sources coherently.

**Unique coverage areas** included continuous batching fairness issues (stall-free batching resource allocation between decode and prefill), Non-Halting Queries research (cyclic fixed points where EOS is never sampled), the vLLM guided generation parser bug (JSON structures failing to close), and the clearest explanation of the MLX 4-bit vs llama.cpp Q4_K_M bits-per-weight discrepancy (4.0 vs 4.7 BPW).

**Additional contributions** included the vLLM scheduling overhead analysis from mlsys.wuklab.io, though this had the most overlap with other analyses.

## C1 vs C2 Synthesis: The Full Picture

Now here's where it gets really interesting, because putting both legs together paints a nuanced and genuinely useful picture.

**The disconfirming evidence (C2) says:** At practical quantization levels (Q4-Q8), MLX fidelity is equivalent to CUDA, quality ceilings are weight-encoded, and diagnostic effort has diminishing returns relative to simply upgrading weights.

**The confirmatory evidence (C1) says:** Infrastructure defects routinely produce symptoms indistinguishable from model failure — including complete collapse, runaway generation, progressive degradation under memory pressure, and cross-runtime output divergence on identical weights. These are real, documented, and some are MLX-specific.

**These are not contradictory.** They're describing different failure regimes. The synthesis is:

**When the system is working correctly** (proper quantization scheme, correct tokenizer config, adequate memory headroom, appropriate dtype for the architecture), MLX inference is faithful to the weights and quality issues are model-level. The disconfirming evidence is right: in steady-state, "upgrade weights" is almost always the better investment than infrastructure debugging.

**When the system is misconfigured or hitting edge cases** (wrong dtype for recurrent architectures, serialization corruption, memory pressure forcing cache drops, EOS misconfiguration, 4-bit quantization on architectures with spiky weight distributions), infrastructure produces catastrophic failures that look exactly like model failures. The confirmatory evidence is right: the ability to detect and diagnose these is essential.

**The practical resolution** isn't "always do decoupled testing" (too expensive, as C2 correctly notes) or "never do it" (too risky, as C1 demonstrates). It's a **tiered diagnostic protocol**:

1. **First response**: Try better weights or higher-precision quantization. This is cheap and addresses the majority case (C2 evidence).
2. **If quality issues persist**: Run a quick infrastructure triage — check dtype compatibility with architecture type, verify tokenizer/EOS config, check memory pressure indicators, compare output against a reference stack. This catches the C1 failure modes.
3. **Deep decoupled testing** (layer-wise activation diffing, logit comparison, shard-level checksumming) is reserved for cases that survive steps 1-2. This is expensive but necessary when you're seeing the kind of total-collapse or progressive-degradation patterns C1 documents.

## Source Overlap and Convergence

The source overlap and convergence patterns across all analyses are informative:

**High-overlap sources (appearing in 4+ analyses across both legs):**
- The MLX non-determinism blog post (adityakarnam.com) — appeared in both C1 and C2
- The Red Hat half-million quantized LLM evaluations study
- The Apple Silicon runtime comparison paper (arxiv 2511.05502)
- The PaliGemma MLX quantization issue (mlx-examples #783)
- The MLX vs GGUF community comparison thread

That non-determinism blog post is a **dual-relevance source** — it appears in both legs because it supports both sides. It says quantized integer models achieve perfect reproducibility (disconfirming the hypothesis) but float16 is catastrophically unstable (confirming it).

**Unique high-value sources (appeared in only 1-2 analyses):**
- The baa.ai Qwen3.5-397B MLX postmortem — arguably the single most valuable source in the entire set
- Anthropic's production postmortem
- vLLM issue #5898 batch-size accuracy drop
- The vLLM guided generation JSON parser bug
- The mlx-lm cache_limit memory growth issue

## Recommendation for Variant C

Evidence sufficiency gate: **passes comfortably**. Across both legs, there are well over 3 includable sources (estimated 20-25 high-quality sources after deduplication), counterexample documented (the entire C2 leg), and the inter-analysis agreement band is high on the core claims.

The evidence is concrete, the failure modes are well-documented with before/after diagnostics, and the practical implications are directly transferable to the MLX / Apple Silicon stack.

The actionable output for the project: build the tiered diagnostic protocol described above, and specifically add checksum validation to any MLX serialization workflow (the bfloat16 corruption bug is a silent killer), dtype checks for any non-transformer architecture, and memory pressure monitoring during long-context inference.

The complete Variant C evidence set — all sources across both legs, all tiers — yields the following final synthesis.

## What the Deep Research Tier Adds to C1

### In-Depth Analysis (C1) — Outstanding

This is the single most valuable document in the entire Variant C corpus. Several unique findings:

**Token-DiFR** (November 2025) — a statistical framework for detecting infrastructure infidelity by comparing token-level outputs against a trusted reference. It detected KV cache quantization with ~10,000 tokens, 4-bit model quantization with ~1,000 tokens, and incorrect sampling seeds with ~100 tokens. This is the closest thing to an automated diagnostic tool for the exact problem the hypothesis describes. None of the other analyses found this.

**Thinking Machines Lab batch invariance analysis** — quantified the problem precisely. Three categories of non-batch-invariant GPU kernels (RMSNorm, matmul, attention), with a `batch_invariant_ops` library providing drop-in replacements at ~34% performance overhead. SGLang integrated this as a first-class feature. This is a concrete, deployable solution.

**The Ollama template disaster** — over 400 GitHub issues from users attributing "model forgetting" to poor model quality, when Ollama's default 2048-token context was silently truncating. An entire community repository (`ktsaou/ollama-templates`) was created to fix incorrect templates across official models. This is the highest-volume example of infrastructure-as-model-failure in the evidence set.

**BF16 RoPE precision loss** — HuggingFace PR #29285 showing that at 8,192 tokens, BF16 thinks positions 8188-8191 are all position 8192 because 7 mantissa bits can't distinguish adjacent large integers. This is a concrete, non-obvious infrastructure failure that would present as "model can't do long context."

**Meta's HawkEye toolkit** — 50% reduction in time-to-root-cause for ML prediction issues using decision-tree-based debugging. This is a production-validated diagnostic methodology.

### Complementary Deep Analysis (C1) — Strong Structural Contribution

The standout contribution from this analysis is the **929-bug empirical analysis** across five inference engines, providing a taxonomic classification of infrastructure defects. The defect category table (State Corruption, Arithmetic Drift, Tokenization Desync, Scheduling Paradoxes, Quantization Faults) with root causes, symptoms, and affected frameworks is directly usable as a diagnostic reference.

The **constrained decoding quality degradation** finding is unique and important — grammar constraints caused a 3.2% Pass@1 drop for C++ generation in Deepseek-Coder and 4.2% for Phi-3.5. This is a category of infrastructure-induced degradation that no other output surfaced: the serving stack's grammar state machine is "suffocating the model's latent capabilities."

The **ragged tensor problem in speculative decoding** is another unique find — different sequences accepting different numbers of drafted tokens creates asymmetric tensor shapes that cause token alignment errors under load.

### Systematic Review (C1) — Methodological Complement

This systematic review mirrors the C2 structure: PRISMA flow (1,099 → 885 → 405 → 50), claims table with strength ratings, research gaps matrix. Its unique contribution is the gaps matrix identifying "Batch/Scheduler Bugs × Unified Memory/Metal Shader Effects" and "XAI Diagnostics × Unified Memory/Metal Shader Effects" as complete gaps — exactly the intersection this project occupies.

## Final Synthesis: Complete Variant C

With all sources now in hand, the evidence resolves into a remarkably clean picture. Both legs are strong, and they're not in tension — they're describing **different failure regimes** of the same system.

### Five validated claims (supported by both legs):

1. **At practical quantization levels (Q4-Q8), MLX output fidelity is statistically indistinguishable from CUDA for standard transformer architectures.** The MMLU Pro benchmark (36.15% vs 36.10%), the 500K-evaluation study, and the community convergence on <3% quality band all support this. The C1 evidence doesn't contradict it — the infrastructure failures it documents are mostly edge cases, misconfigurations, or specific architecture incompatibilities.

2. **Infrastructure failures, when they occur, produce symptoms indistinguishable from model failure.** Both legs agree on this. C2 just argues it's rare; C1 argues it's common enough to matter. The 929-bug analysis, the Anthropic postmortem, and the Ollama 400+ issue pattern all confirm the mechanism is real and recurring.

3. **"Try better weights first" is the correct default diagnostic action.** Even C1's strongest advocates for infrastructure testing acknowledge that model upgrades resolve the majority of perceived quality issues. The 5-10x gap between cross-generation model improvements and quantization penalties is undisputed.

4. **Float16 on MLX is genuinely dangerous; quantized integer models are safe.** Both legs converge here. C2 treats this as "the problem resolves itself at practical quantization levels." C1 treats it as "but you need to know this or you'll waste weeks debugging the wrong thing." Both are correct.

5. **No rigorous MLX-vs-CUDA output fidelity study exists.** The C2 deep research identified this gap; C1's evidence implicitly confirms it by relying on GitHub issues and community benchmarks rather than peer-reviewed comparisons. The MMLU Pro benchmark is the closest thing available and it's a single community test.

### Three contested claims (where legs disagree):

1. **Is decoupled testing worth the cost?** C2 says no for most cases; C1 says yes because the failure modes are severe enough to justify it. Resolution: **it depends on where you are in the deployment lifecycle.** During initial setup and configuration, decoupled testing is essential. In steady-state production, it's overkill — but you need continuous monitoring (Token-DiFR style) to detect when you've slipped back into the danger zone.

2. **Are infrastructure failures "edge cases" or "common"?** C2 frames them as rare; C1 documents hundreds of instances. Resolution: **they're common during setup/migration/upgrade and rare in steady state.** The Ollama template disaster (400+ issues) is "common" but it's also a one-time configuration problem, not an ongoing production failure.

3. **Should you invest in infrastructure diagnostics or model upgrades?** Both are right for their regime. Resolution: **the tiered protocol I proposed earlier, now strengthened by the specific tools C1 surfaced** (Token-DiFR, batch invariance testing, HawkEye-style decision trees).

### The contribution opportunity:

The clearest gap across all sources, confirmed by both systematic review gap matrices and the deep research finding about the missing MLX-CUDA fidelity study, is: **nobody has built a systematic, practitioner-oriented triage framework for determining whether a quality issue on MLX is infrastructure or model.** The pieces exist — Token-DiFR for statistical detection, batch invariance ops for determinism, activation diffing for layer-level isolation, the diagnostic divergence heuristic (crashes = infra, quality degradation = model) — but they haven't been assembled into a coherent protocol specifically for the Apple Silicon / MLX stack.

A tiered diagnostic protocol, tested against the specific failure modes documented in this evidence set, would be a genuine contribution. Both legs of this research independently validate the need for it — C2 by showing the boundary conditions are real even if the steady-state is safe, C1 by documenting the severity when the system is in the wrong regime — giving it strong evidentiary backing.

